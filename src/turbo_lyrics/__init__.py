"""turbo-lyrics: word-timed lyrics from sung vocal stems on a CPU (WhisperX-compatible output)."""

__version__ = "0.1.0"

from .align import ALIGN_MODELS, Aligner
from .asr import AsrOptions
from .compare import compare
from .language import detect_text_language
from .transcriber import GateOptions, Result, Transcriber
from .vad import VadOptions

__all__ = [
    "ALIGN_MODELS",
    "Aligner",
    "AsrOptions",
    "GateOptions",
    "Result",
    "Transcriber",
    "VadOptions",
    "__version__",
    "compare",
    "detect_text_language",
]
