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
import statistics as stats

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
    lr:float = 2e-5, # A learning rate
    num_epochs:int = 5, # The number of epochs to train for
    seed:int = 42, # A seed
    batch_size:int = 32, # The minibatch size per device during training
    eval_batch_size:int = 64, # The minibatch size per device on eval
):
    device = get_device()
    metric = evaluate.load("glue", "mrpc")

    set_seed(seed)
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
            optimizer.zero_grad(0)
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
        # Print only on the main process.
        if get_process_index() == 0:
            print(f"epoch {epoch}:", eval_metric)
    if get_process_index() == 0:
        print('---------------------------')
        print('----- Training Report -----')
        print('---------------------------')
        print('Metric by epoch:')
        for i,met in enumerate(metrics):
            print(f'Epoch {i}: {met}')
        print('---------------------------')
        print('Per batch speeds:')
        print('Training:')
        print(f'Mean: {stats.mean(epoch_train_times):.3f} (ms)')
        for i, t in enumerate(epoch_train_times):
            print(f'Epoch {i}: {t:.3f} (ms)')
        print(f'Evaluation:')
        print(f'Mean: {stats.mean(epoch_validation_times):.3f} (ms)')
        for i, t in enumerate(epoch_validation_times):
            print(f'Epoch {i}: {t:.3f} (ms)')
        