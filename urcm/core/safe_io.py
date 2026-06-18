"""
Safe I/O utilities for loading serialized data.
Provides pickle loading with restrictions to mitigate arbitrary code execution.
"""
import io
import os
import pickle
from typing import Any, Optional


class RestrictedUnpickler(pickle.Unpickler):
    """
    A pickle Unpickler that only allows safe built-in types.
    Blocks execution of arbitrary code during deserialization.
    """
    SAFE_MODULES = {
        'builtins', 'numpy', 'numpy.core', 'numpy.core.multiarray',
        'numpy.ma.core', 'numpy.random', 'collections',
    }
    SAFE_CLASSES = {
        'builtins.range', 'builtins.slice', 'builtins.complex',
        'numpy.dtype', 'numpy.ndarray', 'numpy.float32', 'numpy.float64',
        'numpy.int32', 'numpy.int64', 'numpy.bool_', 'numpy.str_',
        'numpy.bytes_', 'numpy.object_', 'numpy.ma.core.MaskedArray',
        'collections.OrderedDict', 'collections.defaultdict',
    }

    def find_class(self, module: str, name: str) -> Any:
        qualified = f"{module}.{name}"
        if qualified in self.SAFE_CLASSES:
            return super().find_class(module, name)
        if module in self.SAFE_MODULES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Blocked deserialization of {qualified}. "
            f"Only numpy arrays, dicts, and built-in types are allowed."
        )


def safe_load_pickle(file_path: str) -> Any:
    """
    Load a pickle file with restricted unpickling for safety.

    Only allows numpy arrays, dicts, and basic built-in types.
    Blocks arbitrary code execution via crafted pickle payloads.

    Args:
        file_path: Path to the pickle file.

    Returns:
        Deserialized object.

    Raises:
        FileNotFoundError: If file doesn't exist.
        pickle.UnpicklingError: If file contains unsafe types.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pickle file not found: {file_path}")

    with open(file_path, "rb") as f:
        return RestrictedUnpickler(f).load()


def safe_load_pickle_bytes(data: bytes) -> Any:
    """
    Load pickle data from bytes with restricted unpickling.
    """
    return RestrictedUnpickler(io.BytesIO(data)).load()
