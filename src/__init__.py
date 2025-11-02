"""IntLLM - QINS Chat Demo

Interactive chat interface showcasing QINS compression on Phi-3.5-mini.
"""

__version__ = "1.1.0"
__author__ = "IntLLM Team"

from .projective_layer import ProjectiveLinear
from .converter import convert_model_to_projective, measure_conversion_error
from .compression import ProjectiveCompressor
from .model_loader import QINSModelLoader, save_compressed_model

__all__ = [
    "ProjectiveLinear",
    "convert_model_to_projective",
    "measure_conversion_error",
    "ProjectiveCompressor",
    "QINSModelLoader",
    "save_compressed_model",
]
