# This script is based on the NLP example script available in 
# accelerate: https://github.com/huggingface/accelerate/blob/main/examples/nlp_example.py
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from fastcore.script import call_parse

import evaluate
from pytorch_benchmark import prepare_modules, get_device, is_tpu_available, get_process_index
from accelerate.utils import gather
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

import time
import json
import yaml
import statistics as stats
from pathlib import Path

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
    change_seed:bool = True, # Whether to change the seed each iteration
):
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    for k,v in config.items():
        config[k] = float(v) if k == "lr" else int(v)
    fname = Path(config_file).name.split('.')[0]
    if num_iterations < 1:
        num_iterations = 1
    Path(f'reports/nlp_script/{fname}').mkdir(exist_ok=True)
    lr, num_epochs, seed, batch_size, eval_batch_size = (
            config["lr"], config["num_epochs"], config["seed"], config["train_batch_size"], config["eval_batch_size"]
        )
    fname = Path(config_file).name.split('.')[0]
    for iteration in range(num_iterations):
        device = get_device()
        metric = evaluate.load("glue", "mrpc")

        set_seed(seed)
        if change_seed:
            seed += 1
        train_dataloader, eval_dataloader = get_dataloaders(batch_size, eval_batch_size)
        # Instantiate the model (we build the model here so that the seed also control new weights initialization)
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", return_dict=True)
        model = model.to(device)

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
        model, optimizer, lr_scheduler = prepare_modules(
            model, optimizer, lr_scheduler
        )

        # Now we train the model
        train_times = []
        epoch_train_times = []
        validation_times = []
        epoch_validation_times = []
        metrics = []
        for epoch in range(num_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                start_time = time.perf_counter()
                batch.to(device)
                outputs = model(**batch)
                outputs.loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                end_time = time.perf_counter()
                train_times.append(end_time - start_time)
            epoch_train_times.append(stats.mean(train_times))
            train_times = []

            model.eval()
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
                metric.add_batch(
                    predictions=predictions,
                    references=references,
                )
            epoch_validation_times.append(stats.mean(validation_times))
            validation_times = []

            eval_metric = metric.compute()
            metrics.append(eval_metric)
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
            with open(f'reports/nlp_script/{fname}/run_{iteration}.json', "w") as outfile:
                json.dump(report, outfile)

            print(f'Report saved to reports/nlp_script/{fname}/run_{iteration}.json')
        del model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        clear_memory()