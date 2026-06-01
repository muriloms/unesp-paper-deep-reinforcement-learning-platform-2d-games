"""Helpers genéricos: seed, formatação, validação de ambiente."""
from __future__ import annotations
import random
import sys
import warnings

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Reprodutibilidade: NumPy, Python, PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """torch.device — usa CUDA se disponível."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_warnings() -> None:
    """Silencia warnings irrelevantes do gym/numpy."""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def check_python_version(min_minor: int = 10, max_minor: int = 11) -> None:
    """Aborta se Python não está na faixa suportada (nes-py quebra em 3.12+)."""
    v = sys.version_info
    if v.major != 3 or not (min_minor <= v.minor <= max_minor):
        raise RuntimeError(
            f"Python {v.major}.{v.minor} não é suportado. "
            f"Use Python 3.{min_minor}–3.{max_minor} (nes-py 8.2.1 quebra em Py 3.12+)."
        )


def print_env_info() -> None:
    """Imprime versão do Python, PyTorch e GPU."""
    print(f"Python : {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
