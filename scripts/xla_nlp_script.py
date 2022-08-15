# This script is based on the NLP example script available in 
# accelerate: https://github.com/huggingface/accelerate/blob/main/examples/nlp_example.py

from fastcore.script import call_parse

from aim import Run

import yaml
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from huggingface_hub import Repository

from datasets import load_dataset
from evaluate import load as load_metric 
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from accelerate.utils import set_seed, extract_model_from_parallel, wait_for_everyone, save
from accelerate.utils.operations import _tpu_gather 

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.serialization as xser


"""
Configuration Parameters:

- Learning rate
- Mixed precision type
- Downcast
- Default tensor type

How each one relates:

- Learning Rate:
    - 1x lr
    - (n)x lr (n = num_processes)

- Mixed Precision Type:
    - no
    - bf16 (normal)
    - bf16 (downcast)
    - fp16

- Default Tensor Type:
    - May set it to `"torch.FloatTensor"`
"""


def get_dataloaders(batch_size:int=16, eval_batch_size:int=32):
    """Creates a set of `Dataloader`s for the `glue` dataset,
    using "bert-base-cased" as the tokenizer

    Args:
        batch_size (`int`, *optional*, defaults to 16):
            Training dataloader batch size
        eval_batch_size (`int`, *optional*, defaults to 32):
            Eval dataloader batch size
    """
    MODEL = "bert-base-cased"
    DATASET = "mrpc"
    METRIC = "glue"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    datasets = load_dataset(DATASET, METRIC)

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence1"], 
            examples["sentence2"],
            truncation=True,
            max_length=None
        )

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence1", "sentence2"]
    )

    def collate_fn(examples):
        # On TPU you should pad everything to be the same length
        return tokenizer.pad(
            examples, 
            padding="max_length", 
            max_length=128, 
            return_tensors="pt"
        )
    
    train_sampler = DistributedSampler(
        tokenized_datasets["train"],
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )

    eval_sampler = DistributedSampler(
        tokenized_datasets["validation"],
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False
    )

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        collate_fn=collate_fn,
        batch_size=batch_size,
        sampler=train_sampler
    )

    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        collate_fn=collate_fn,
        batch_size=eval_batch_size,
        sampler=eval_sampler
    )

    return train_dataloader, eval_dataloader, tokenizer

""" 
Example config:

{
    "lr":1e-4,
    "mixed_precision":"no",
    "set_default":0
}
"""


@call_parse
def main(
    config_file:str, # Location of the config file
    num_iterations:int = 3, # Number of times to run the benchmark
):
    MODEL = "bert-base-cased"
    DATASET = "mrpc"
    METRIC = "glue"
    SEED = 108

    HUB_STR_TEMPLATE = "muellerzr/bert-base-cased-tpu-accelerate-experiments"
    BASE_DIR = HUB_STR_TEMPLATE.split("/")[1].replace("-", "_")
    with open(config_file, "r") as stream:
        config = yaml.safe_load(stream)
    for key, value in config.items():
        if key == "lr":
            config[key] = float(config[key])
        if key == "set_default":
            config[key] = bool(config[key])

    if config["set_default"]:
        torch.set_default_tensor_type("torch.FloatTensor")
    
    metric = load_metric(METRIC, DATASET)
    device = xm.xla_device()
    IS_LOCAL_PROCESS = xm.get_local_ordinal()

    for iteration in range(num_iterations):
        save_dir = f'{BASE_DIR}_{Path(config_file).name}_{iteration}'
        if IS_LOCAL_PROCESS:
            run = Run(repo=f"~/{save_dir}")
            run['hparams'] = {
                **config,
                "iteration":iteration,
                "seed":SEED,
            }
        set_seed(SEED)

        train_dataloader, eval_dataloader, tokenizer = get_dataloaders()
        train_dataloader = pl.MpDeviceLoader(train_dataloader, device) 
        eval_dataloader = pl.MpDeviceLoader(eval_dataloader, device)

        model = AutoModelForSequenceClassification.from_pretrained(MODEL, return_dict=True)
        model = xmp.MpModelWrapper(model).to(device)

        optimizer = AdamW(params=model.parameters(), lr=config["lr"])

        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=100,
            num_training_steps=(len(train_dataloader) * 3)
        )

        xm.master_print(f'Starting training...')

        for epoch in range(3):
            model.train()
            for step, batch in enumerate(train_dataloader):
                batch.to(device)
                outputs = model(**batch)
                loss = outputs.loss
                if IS_LOCAL_PROCESS:
                    run.track(loss.item(), name="train_loss", epoch=epoch, context={"subset":"train"})
                loss.backward()
                xm.optimizer_step(optimizer)
                scheduler.step() 
                optimizer.zero_grad()
            
            model.eval()
            samples_seen = 0
            for step,batch in enumerate(eval_dataloader):
                batch.to(device)
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                predictions, references = _tpu_gather((predictions, batch["labels"]), name="accelerate.utils._tpu_gather")
                if step == (len(eval_dataloader) - 1):
                    predictions = predictions[: len(eval_dataloader._loader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader._loader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
                metric.add_batch(
                    predictions=predictions,
                    references=references
                )
            eval_metrics = metric.compute()
            if IS_LOCAL_PROCESS:
                for metric, value in eval_metrics.items():
                    run.track(value, name=metric, epoch=epoch, context={"subset":"validation"})
                xm.master_print(f'Epoch {epoch} complete...')

        SEED += 100*iteration
        wait_for_everyone()
        repo = Repository(
            local_dir=save_dir,
            clone_from=f'{HUB_STR_TEMPLATE}',
            revision=f"{Path(config_file).name}-{iteration}",
            use_auth_token=True
        )
        with repo.commit(commit_message=f"Uploading experiment {Path(config_file).name}"):
            unwrapped_model = extract_model_from_parallel(model)
            unwrapped_model.save_pretrained(
                save_dir, is_main_process=IS_LOCAL_PROCESS, save_function=save
            )
            if IS_LOCAL_PROCESS:
                tokenizer.save_pretrained(save_dir)
        repo.push_to_hub(commit_message="End of training, uploading logs", auto_lfs_prune=True)