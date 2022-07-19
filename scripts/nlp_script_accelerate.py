# This script is based on the NLP example script available in 
# accelerate: https://github.com/huggingface/accelerate/blob/main/examples/nlp_example.py
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from fastcore.script import call_parse

import evaluate
from accelerate import Accelerator
from accelerate.utils import DistributedType, is_tpu_available
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

import time
import json
import os
import yaml
import statistics as stats
from pathlib import Path

def clear_memory():
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_dataloaders(accelerator, batch_size: int = 16, eval_batch_size:int = 32):
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
        if accelerator.distributed_type == DistributedType.TPU:
            return tokenizer.pad(examples, padding="max_length", max_length=128, return_tensors="pt")
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=eval_batch_size
    )

    return train_dataloader, eval_dataloader

@call_parse
def main(
    config_file:str, # Location of the config file
    num_iterations:int = 3, # Number of times to run benchmark
):
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    for k,v in config.items():
        if k != "mixed_precision":
            config[k] = float(v) if k == "lr" else int(v)
    accelerator = Accelerator(mixed_precision=config.get("mixed_precision", None))

    fname = Path(config_file).name.split('.')[0]
    typ = Path(config_file).parent.name

    if num_iterations < 1:
        num_iterations = 1
    Path(f'reports/nlp_script_accelerate/{typ}_{fname}').mkdir(exist_ok=True)
    lr, num_epochs, seed, batch_size, eval_batch_size = (
            config["lr"], config["num_epochs"], config["seed"], config["train_batch_size"], config["eval_batch_size"]
    )
    fname = Path(config_file).name.split('.')[0]
    
    for iteration in range(num_iterations):
        metric = evaluate.load("glue", "mrpc")

        set_seed(seed)
        train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size, eval_batch_size)
        # Instantiate the model (we build the model here so that the seed also control new weights initialization)
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", return_dict=True)

        # Instantiate optimizer
        optimizer = AdamW(params=model.parameters(), lr=lr)

        # Instantiate scheduler
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=(len(train_dataloader) * num_epochs),
        )

        # Prepare everything
        # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
        # prepare method.
        model, optimizer, lr_scheduler, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, lr_scheduler, train_dataloader, eval_dataloader
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
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
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
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                end_time = time.perf_counter()
                validation_times.append(end_time - start_time)
                predictions, references = accelerator.gather((predictions, batch["labels"]))
                if accelerator.use_distributed:
                    # Then see if we're on the last batch of our eval dataloader
                    if step == len(eval_dataloader) - 1:
                        # Last batch needs to be truncated on distributed systems as it contains additional samples
                        if hasattr(eval_dataloader, "dataset"):
                            dataset_length = len(eval_dataloader.dataset)
                        elif hasattr(eval_dataloader, "_loader"):
                            dataset_length = len(eval_dataloader._loader.dataset)
                        else:
                            raise ValueError("Unsupported dataloader type")
                        predictions = predictions[: dataset_length - samples_seen]
                        references = references[: dataset_length - samples_seen]
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
        if accelerator.local_process_index == 0:
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
            with open(f'reports/nlp_script_accelerate/{typ}_{fname}/run_{iteration}.json', "w") as outfile:
                json.dump(report, outfile, indent=4)

            print(f'Report saved to reports/nlp_script_accelerate/{typ}_{fname}/run_{iteration}.json')
        del model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        clear_memory()
        seed += 100