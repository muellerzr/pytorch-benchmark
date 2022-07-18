# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01_prepare.ipynb (unless otherwise specified).

__all__ = ['get_device', 'prepare_model', 'OptimizerInterface', 'prepare_optimizer', 'SchedulerInterface',
           'prepare_scheduler', 'prepare_modules']

# Cell
from .imports import is_tpu_available, is_multigpu_available

# Cell
import os, torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

if is_tpu_available(check_device=False):
    import torch_xla.distributed.xla_multiprocessing as xmp

# Cell
_device_rank = int(os.environ.get("LOCAL_RANK", -1))

# Cell
def get_device():
    if is_tpu_available(): return xm.xla_device()
    elif is_multigpu_available(): return torch.device("cuda", _device_rank)
    else: return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cell
def prepare_model(
    model:nn.Module, # A PyTorch model to wrap
    **kwargs
):
    "Prepares a model for distributed training. kwargs are sent to DDP"
    if is_tpu_available():
        return xmp.MpModelWrapper(model)
    elif is_multigpu_available():
        return DDP(model, device_ids=[_device_rank], output_device=_device_rank)
    return model

# Cell
class OptimizerInterface(torch.optim.Optimizer):
    "Basic optimizer wrapper that performs the right step call for TPU"
    def __init__(self, optimizer):
        self.opt = optimizer

    @property
    def state(self): return self.opt.state

    @state.setter
    def state(self, state): self.opt.state = state

    @property
    def defaults(self): return self.opt.defaults

    @defaults.setter
    def defaults(self, defaults): self.opt.defaults = defaults

    def state_dict(self): return self.opt.state_dict()

    def zero_grad(self): return self.opt.zero_grad()

    def step(self, closure=None):
        if is_tpu_available():
            xm.optimizer_step(self.opt, {})
        self.opt.step(closure)

# Cell
def prepare_optimizer(
    opt:torch.optim.Optimizer
):
    return OptimizerInterface(opt)

# Cell
class SchedulerInterface:
    "Wrapper to step the scheduler the right number of times"
    def __init__(self, scheduler, num_processes):
        self.scheduler = scheduler
        self.num_processes = num_processes

    def step(self, *args, **kwargs):
        for _ in range(self.num_processes):
            if getattr(self.scheduler, "total_steps", 0) <= self.scheduler.last_epoch:
                self.scheduler.step(*args, **kwargs)

# Cell
def prepare_scheduler(
    sched:torch.optim.lr_scheduler._LRScheduler
):
    if is_tpu_available():
        num_processes = 8 # hard coded for my tests
    elif is_multigpu_available():
        num_processes = torch.cuda.device_count()
    else:
        num_processes = 1
    return SchedulerInterface(sched, num_processes)

# Cell
def _prepare_one(obj, first_pass=False):
    # first pass on preperation: DataLoader, model, optimizer
    if first_pass:
        if isinstance(obj, torch.nn.Module):
            return prepare_model(obj)
        elif isinstance(obj, torch.optim.Optimizer):
            return prepare_optimizer(obj)
    elif isinstance(obj, torch.optim.lr_scheduler._LRScheduler):
        return prepare_scheduler(obj)
    return obj

# Cell
def prepare_modules(*modules):
    result = tuple(_prepare_one(obj, first_pass=True) for obj in modules)
    return tuple(_prepare_one(obj) for obj in result)