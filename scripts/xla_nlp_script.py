# This script is based on the NLP example script available in 
# accelerate: https://github.com/huggingface/accelerate/blob/main/examples/nlp_example.py
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from fastcore.script import call_parse

import evaluate
from pytorch_benchmark import prepare_modules, get_device, is_tpu_available, get_process_index, num_processes
from accelerate.utils import gather, convert_outputs_to_fp32
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

import time
import json
import os
import yaml
import statistics as stats
from pathlib import Path

if is_tpu_available(False):
  import torch_xla.core.xla_model as xm
  import torch_xla.distributed.parallel_loader as pl
  import torch_xla.distributed.xla_multiprocessing as xmp
  import torch_xla.utils.serialization as xser

def clear_memory():
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# For gather
_ = Accelerator()


def get_dataloaders(batch_size: int = 16, eval_batch_size:int = 32):
    """
    Creates a set of `DataLoader`s for the `glue` dataset,
    using "bert-base-cased" as the tokenizer.
    Args:
        batch_size (`int`, *optional*):
            The batch size for the trainining DataLoader.
        batch_size (`int`, *optional*):
            The batch size for the validation DataLoader.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    datasets = load_dataset("glue", "mrpc")

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
        return outputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence1", "sentence2"],
    )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    def collate_fn(examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        if is_tpu_available():
            return tokenizer.pad(examples, padding="max_length", max_length=128, return_tensors="pt")
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    train_sampler = torch.utils.data.distributed.DistributedSampler(
          tokenized_datasets["train"],
          num_replicas=xm.xrt_world_size(), #divide dataset among this many replicas
          rank=xm.get_ordinal(), #which replica/device/core
          shuffle=True
    )

    eval_sampler = torch.utils.data.distributed.DistributedSampler(
          tokenized_datasets["validation"],
          num_replicas=xm.xrt_world_size(), #divide dataset among this many replicas
          rank=xm.get_ordinal(), #which replica/device/core
          shuffle=False
    )

    # Instantiate dataloaders.
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

    return train_dataloader, eval_dataloader

@call_parse
def main(
    config_file:str, # Location of the config file
    iteration:int = 1, # Number of times to run benchmark
    outfile:str = None, # Location of output file
    set_default:bool =False # Whether to set default tensor type
):
    if set_default:
      torch.set_default_tensor_type('torch.FloatTensor')
    iteration = iteration - 1
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    for k,v in config.items():
        if k != "mixed_precision":
            config[k] = float(v) if k == "lr" else int(v)

    fname = Path(config_file).name.split('.')[0]
    typ = Path(config_file).parent.name

    Path(f'reports/nlp_script/{typ}_{fname}').mkdir(exist_ok=True)
    lr, num_epochs, seed, batch_size, eval_batch_size = (
            config["lr"], config["num_epochs"], config["seed"], config["train_batch_size"], config["eval_batch_size"]
    )
    fname = Path(config_file).name.split('.')[0]
    
    device = xm.xla_device()
    metric = evaluate.load("glue", "mrpc")

    seed += 100*iteration

    set_seed(seed)
    train_dataloader, eval_dataloader = get_dataloaders(batch_size, eval_batch_size)
    train_dataloader = pl.MpDeviceLoader(train_dataloader, device)
    eval_dataloader = pl.MpDeviceLoader(eval_dataloader, device)
    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", return_dict=True)
    model = xmp.MpModelWrapper(model).to(device)

    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=lr)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )
    # Now we train the model
    train_times = []
    epoch_train_times = []
    validation_times = []
    epoch_validation_times = []
    metrics = []
    total_time_start = time.perf_counter()
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            start_time = time.perf_counter()
            batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            xm.optimizer_step(optimizer)
            lr_scheduler.step()
            optimizer.zero_grad()
            end_time = time.perf_counter()
            train_times.append(end_time - start_time)
        epoch_train_times.append(stats.mean(train_times))
        train_times = []

        model.eval()
        samples_seen = 0
        for step, batch in enumerate(eval_dataloader):
            start_time = time.perf_counter()
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            end_time = time.perf_counter()
            validation_times.append(end_time - start_time)
            predictions, references = gather((predictions, batch["labels"]))
            if num_processes() > 1:
                # Then see if we're on the last batch of our eval dataloader
                if step == len(eval_dataloader) - 1:
                    # Last batch needs to be truncated on distributed systems as it contains additional samples
                    predictions = predictions[: len(eval_dataloader._loader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader._loader.dataset) - samples_seen]
                else:
                    # Otherwise we add the number of samples seen
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
        epoch_validation_times.append(stats.mean(validation_times))
        validation_times = []

        eval_metric = metric.compute()
        metrics.append(eval_metric)
    total_time = time.perf_counter() - total_time_start
    if get_process_index() == 0:
        print('-----------------------------------------------------')
        print(f'----- Training Report for Iteration {iteration} ----')
        print('-----------------------------------------------------')
        if torch.cuda.is_available(): 
            num_devices = torch.cuda.device_count()
        elif is_tpu_available():
            num_devices = 8
        else:
            num_devices = 1
        print(f'Number of devices: {num_devices}')
        print(f'Training device: {"cuda" if not is_tpu_available() else "tpu"}')
        print('Metric by epoch:')
        for i,met in enumerate(metrics):
            print(f'Epoch {i}:')
            for key,val in met.items():
                print(f'\t{key}: {val:.2f}')
        print('---------------------------')
        print(f'Total training time: {total_time} (s)')
        print('Per batch speeds:')
        print('Training:')
        print(f'Mean: {stats.mean(epoch_train_times):.3f} (ms/batch)')
        for i, t in enumerate(epoch_train_times):
            print(f'Epoch {i}: {t:.3f} (ms/batch)')
        print(f'Evaluation:')
        print(f'Mean: {stats.mean(epoch_validation_times):.3f} (ms/batch)')
        for i, t in enumerate(epoch_validation_times):
            print(f'Epoch {i}: {t:.3f} (ms/batch)')

        report = {
            "dataset":"mrpc",
            "model":"bert-base-cased",
            "lr":lr,
            "num_epochs":num_epochs,
            "seed":seed,
            "train_batch_size":batch_size,
            "eval_batch_size":eval_batch_size,
            "num_devices":num_devices, 
            "training_device": "cuda" if not is_tpu_available() else "tpu",
            "metrics": metrics,
            "speeds": {
                "total": total_time,
                "training": {
                    "mean": stats.mean(epoch_train_times),
                    "times": epoch_train_times
                },
                "evaluation":{
                    "mean": stats.mean(epoch_validation_times),
                    "times": epoch_validation_times
                }
            }
        }
        if outfile is None:
          outfile = f'reports/nlp_script/{typ}_{fname}/run_{iteration}.json'
        with open(outfile, "w") as of:
            json.dump(report, of, indent=4)

        print(f'Report saved to reports/nlp_script/{typ}_{fname}/run_{iteration}.json')