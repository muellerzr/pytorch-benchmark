# PyTorch Benchmarks
> A small repo comparing XLA to distributed to single GPUs with PyTorch


## Install

The main library contains a smaller version of [Accelerate](https://github.com/huggingface/accelerate) aimed at only wrapping the bare minimum needed to note performance gains from each of the three distributed platforms (GPU, multi-GPU, and TPU). 

Given this is a small benchmark library, I will not be releasing it on pypi and instead you should install from main:

`pip install git+https://github.com/muellerzr/pytorch-benchmark`

It uses barebones dependencies, and relies on Accelerate only for basic utility functions (such as `gather` and `accelerate launch`). I implement my own small version of the main wrapper classes for the sake of simplicity. 

## How to use

TODO
