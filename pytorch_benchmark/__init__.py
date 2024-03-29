__version__ = "0.0.1"

from .imports import is_tpu_available, is_multigpu_available
from .prepare import prepare_modules
from .utils import get_process_index, num_processes, get_device, get_rank