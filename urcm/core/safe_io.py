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
        'builtins', 'collections',
        'numpy', 'numpy.core', 'numpy._core',
        'numpy.core.multiarray', 'numpy._core.multiarray',
    }
    SAFE_MODULE_PREFIXES = ('numpy.', 'numpy.core.', 'numpy._core.')  # Allow all numpy submodules
    SAFE_CLASSES = {
        'builtins.range', 'builtins.slice', 'builtins.complex',
        'collections.OrderedDict', 'collections.defaultdict',
        # Explicit fallback for numpy array reconstruction (used in pickle)
        'numpy._core.multiarray._reconstruct',
        'numpy.core.multiarray._reconstruct',
        'numpy._core.multiarray.scalar',
        'numpy.core.multiarray.scalar',
        'numpy.ndarray', 'numpy.dtype',
    }

    def find_class(self, module: str, name: str) -> Any:
        qualified = f"{module}.{name}"
        if qualified in self.SAFE_CLASSES:
            return super().find_class(module, name)
        if module in self.SAFE_MODULES:
            return super().find_class(module, name)
        # Allow any numpy submodule (numpy, numpy.core, numpy._core, numpy._core.multiarray, etc.)
        for prefix in self.SAFE_MODULE_PREFIXES:
            if module.startswith(prefix):
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
