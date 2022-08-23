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

from accelerate import Accelerator, DistributedType

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
    datasets = load_dataset(METRIC, DATASET)

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

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

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

    HUB_STR_TEMPLATE = "muellerzr/bert-base-cased-tpu-experiments-accelerate"
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
    accelerator = Accelerator(
        mixed_precision="no" if not config["bf16"] else "bf16"
    )
    if accelerator.is_local_main_process:
        repo = Repository(
            "bert_base_cased_tpu_accelerate_experiments",
            HUB_STR_TEMPLATE,
            use_auth_token=True,
        )
        repo.git_checkout("main")
    xm.rendezvous("creating repo")

    for iteration in range(num_iterations):
        SEED += (1000*iteration)
        set_seed(SEED)
        if accelerator.is_local_main_process:
            experiment = f'{Path(config_file).name.split(".")[0]}_dataloaders'
            repo.git_checkout(experiment, create_branch_ok=True)
            run = Run(repo=".", experiment=experiment)
            run['hparams'] = {
                **config,
                "iteration":iteration,
                "seed":SEED,
                "script":experiment,
                "xla_dataloaders":True
            }
        wait_for_everyone()
        
        train_dataloader, eval_dataloader, tokenizer = get_dataloaders()

        model = AutoModelForSequenceClassification.from_pretrained(MODEL, return_dict=True)
        model = model.to(accelerator.device)

        optimizer = AdamW(params=model.parameters(), lr=config["lr"])

        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=100,
            num_training_steps=(len(train_dataloader) * 3)
        )

        model, optimizer, scheduler = accelerator.prepare(
            model, optimizer, scheduler
        )

        if hasattr(model, "tie_weights"):
            model.tie_weights()

        xm.master_print(f'Starting training...')

        for epoch in range(3):
            model.train()
            for batch in train_dataloader:
                outputs = model(**batch)
                loss = outputs.loss
                if accelerator.is_local_main_process:
                    run.track(loss.item(), name="train_loss", epoch=epoch, context={"subset":"train", "script":experiment})
                wait_for_everyone()
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step() 
                optimizer.zero_grad()
            
            model.eval()
            for batch in eval_dataloader:
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
                metric.add_batch(
                    predictions=predictions,
                    references=references
                )
            eval_metrics = metric.compute()
            if accelerator.is_local_main_process:
                for met, value in eval_metrics.items():
                    run.track(value, name=met, epoch=epoch, context={"subset":"validation", "script":experiment})
                xm.master_print(f'Epoch {epoch} complete...')
                xm.master_print(f'Metrics: {eval_metrics}')
            wait_for_everyone()

        unwrapped_model = extract_model_from_parallel(model)
        # wait for everyone TPU specific
        xm.rendezvous("accelerate.utils.wait_for_everyone")
        unwrapped_model.save_pretrained(
            "bert_base_cased_tpu_accelerate_experiments", is_main_process=accelerator.is_local_main_process, save_function=save
        )
        if accelerator.is_local_main_process:
            tokenizer.save_pretrained("bert_base_cased_tpu_accelerate_experiments")
        wait_for_everyone()
        if accelerator.is_local_main_process:
            repo.git_add(auto_lfs_track=True)
            repo.git_commit(f'{experiment}_iteration_{iteration}')
            repo.git_push(upstream=f'origin {experiment}')
        wait_for_everyone()
        xm.rendezvous("upload to git")