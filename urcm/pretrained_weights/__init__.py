"""
Pre-trained weight converters for URCM.

Maps HuggingFace transformer weights (BERT, GPT-2, DistilBERT) 
into URCM's echo state network compatible format.
"""

from .converter import (
    convert_transformer_to_urcm,
    list_available_models,
    get_model_info,
    download_and_convert,
    save_urcm_weights,
)

__all__ = [
    "convert_transformer_to_urcm",
    "list_available_models",
    "get_model_info",
    "download_and_convert",
    "save_urcm_weights",
]
