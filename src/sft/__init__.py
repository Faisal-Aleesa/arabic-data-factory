"""
SFT (Supervised Fine-Tuning) Sample Generation

Complete pipeline for generating high-quality Arabic SFT training data:
- Schema: Defines sample structure and validation
- Prompts: Detailed instructions for model
- Generator: API calls and normalization
- Validation: Schema and quality checks
"""

from .schema import (
    SFTSample,
    QuestionType,
    Difficulty,
    ReasoningMode,
    ReasoningDepth,
    ValidationStatus,
    Message,
    TeacherOnlyKnowledge,
    SourceMetadata,
    GenerationMetadata,
    QualityMetrics,
)
from .generator import SFTGenerator, load_chunks, save_samples
from .validation import SFTValidator, QualityScorer
from .prompts import SFTPrompts, DPOPrompts

__all__ = [
    "SFTSample",
    "QuestionType",
    "Difficulty",
    "ReasoningMode",
    "ReasoningDepth",
    "ValidationStatus",
    "Message",
    "TeacherOnlyKnowledge",
    "SourceMetadata",
    "GenerationMetadata",
    "QualityMetrics",
    "SFTGenerator",
    "load_chunks",
    "save_samples",
    "SFTValidator",
    "QualityScorer",
    "SFTPrompts",
    "DPOPrompts",
]
