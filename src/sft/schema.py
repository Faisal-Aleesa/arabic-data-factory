"""
SFT Sample Schema and Validation

Defines the complete data model for SFT samples with quality metrics and metadata.
All samples must conform to this schema.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from enum import Enum


class QuestionType(str, Enum):
    DEFINITION = "Definition"
    MEANING = "Meaning"
    EXPLANATION = "Explanation"
    CLARIFICATION = "Clarification"
    COMPARISON = "Comparison"
    DISTINCTION = "Distinction"
    CAUSE_WHY = "Cause/Why"
    REASONING = "Reasoning"
    INFERENCE = "Inference"
    APPLICATION = "Application"
    EXAMPLE = "Example"
    CONTEXTUAL_MEANING = "Contextual Meaning"
    RELATIONSHIP = "Relationship"
    MULTI_STEP = "Multi-step"
    CONCEPTUAL = "Conceptual"


class Difficulty(str, Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


class ReasoningMode(str, Enum):
    DIRECT_RECALL = "Direct Recall"
    CONTEXTUAL_INTERPRETATION = "Contextual Interpretation"
    DEFINITION_RESOLUTION = "Definition Resolution"
    DISAMBIGUATION = "Disambiguation"
    COMPARISON = "Comparison"
    CAUSE_EFFECT = "Cause → Effect"
    EVIDENCE_CONCLUSION = "Evidence → Conclusion"
    CONCEPT_APPLICATION = "Concept → Application"
    MULTI_STEP_INFERENCE = "Multi-step Inference"
    RELATIONSHIP_RESOLUTION = "Relationship Resolution"


class ReasoningDepth(int, Enum):
    DIRECT = 0
    SINGLE_STEP = 1
    MULTI_STEP = 2
    DEEP = 3


class ValidationStatus(str, Enum):
    UNVERIFIED = "Unverified"
    ACCEPTED = "Accepted"
    REVIEW = "Review"
    REJECTED = "Rejected"


class Split(str, Enum):
    TRAIN = "Train"
    VALIDATION = "Validation"
    TEST = "Test"


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class TeacherOnlyKnowledge:
    support_facts: List[str]  # Atomic facts that justify the answer
    knowledge_units: List[str]  # Minimal concepts the sample teaches
    required_facts: List[str]  # Info that must appear in answer
    optional_facts: List[str]  # Correct but not required info
    source_id: Optional[str] = None
    chunk_id: Optional[str] = None
    source_span: Optional[Dict[str, Any]] = None
    source_domain: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SourceMetadata:
    doc_id: Optional[str] = None
    chunk_id: Optional[str] = None
    region: Optional[str] = None
    format_type: Optional[str] = None
    source_pointer: Optional[Dict[str, Any]] = None
    license: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class GenerationMetadata:
    model: str
    prompt_version: str
    temperature: float = 0.2
    max_tokens: int = 12000

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QualityMetrics:
    """Quality scores for the sample (0.0–1.0 unless noted)"""
    support_coverage: float  # How well support_facts cover the answer
    answer_completeness: float  # Does answer fully address the question?
    factual_correctness: float  # Is the answer factually correct?
    reasoning_correctness: float  # Is thinking logically sound?
    question_naturalness: float  # Does question sound like natural Saudi Arabic?
    answer_naturalness: float  # Does answer sound natural?
    saudi_dialect_quality: float  # Quality of Saudi dialect usage
    instruction_quality: float  # Is question clear and unambiguous?
    response_quality: float  # Overall response quality
    complexity_score: float  # Question complexity (0.0–1.0)
    diversity_score: float  # Uniqueness vs similar samples
    information_density: float  # Useful info per word
    redundancy_score: float  # Verbosity score (lower is better)
    similarity_to_nearest_sample: float  # Similarity to nearest neighbor
    teacher_confidence: float  # Overall confidence in sample quality

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def average_score(self) -> float:
        """Weighted average of key quality metrics"""
        scores = [
            self.factual_correctness * 0.20,
            self.answer_completeness * 0.15,
            self.question_naturalness * 0.15,
            self.answer_naturalness * 0.15,
            self.instruction_quality * 0.10,
            self.support_coverage * 0.10,
            self.saudi_dialect_quality * 0.10,
            self.information_density * 0.05,
        ]
        return sum(scores)


@dataclass
class SFTSample:
    """Complete SFT sample according to spec"""
    # Core Content
    sample_id: str
    chunk_id: str
    question: str
    thinking: str  # Saudi Arabic thinking (main)
    thinking_arabic: str  # Saudi Arabic thinking (explicit)
    thinking_english: str  # English translation of thinking
    answer: str
    messages: List[Message]  # [{"role": "user", "content": question}, ...]

    # Classification
    question_type: QuestionType
    difficulty: Difficulty
    reasoning_mode: ReasoningMode
    reasoning_depth: ReasoningDepth

    # Knowledge
    teacher_only: TeacherOnlyKnowledge

    # Metadata
    language: str = "Arabic"
    dialect: str = "Saudi"
    domain: Optional[str] = None
    topic: Optional[str] = None
    subtopic: Optional[str] = None
    knowledge_dependency: str = "Medium"  # Low / Medium / High
    ambiguity_level: str = "None"  # None / Low / Medium / High
    question_length: str = "Medium"  # Short / Medium / Long
    thinking_length: str = "Medium"  # None / Short / Medium / Long
    answer_length: str = "Medium"  # Short / Medium / Long
    dialect_strength: str = "Natural"  # Light / Natural / Strong
    terminology_level: str = "General"  # General / Specialized / Technical

    # Source
    source: SourceMetadata = None
    generation: GenerationMetadata = None

    # Quality
    grounded: bool = True
    quality_metrics: QualityMetrics = None

    # Status
    validation_status: ValidationStatus = ValidationStatus.UNVERIFIED
    split: Optional[Split] = None

    def __post_init__(self):
        """Convert string enums to proper Enum types if needed"""
        if isinstance(self.question_type, str):
            self.question_type = QuestionType(self.question_type)
        if isinstance(self.difficulty, str):
            self.difficulty = Difficulty(self.difficulty)
        if isinstance(self.reasoning_mode, str):
            self.reasoning_mode = ReasoningMode(self.reasoning_mode)
        if isinstance(self.reasoning_depth, int):
            self.reasoning_depth = ReasoningDepth(self.reasoning_depth)
        if isinstance(self.validation_status, str):
            self.validation_status = ValidationStatus(self.validation_status)
        if self.source and not isinstance(self.source, SourceMetadata):
            self.source = SourceMetadata(**self.source)
        if self.generation and not isinstance(self.generation, GenerationMetadata):
            self.generation = GenerationMetadata(**self.generation)

    def to_dict(self, include_teacher_only: bool = True) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = {
            "sample_id": self.sample_id,
            "chunk_id": self.chunk_id,
            "question": self.question,
            "thinking": self.thinking,
            "thinking_arabic": self.thinking_arabic,
            "thinking_english": self.thinking_english,
            "answer": self.answer,
            "messages": [msg.to_dict() for msg in self.messages],
            "question_type": self.question_type.value,
            "difficulty": self.difficulty.value,
            "reasoning_mode": self.reasoning_mode.value,
            "reasoning_depth": self.reasoning_depth.value,
            "language": self.language,
            "dialect": self.dialect,
            "grounded": self.grounded,
            "validation_status": self.validation_status.value,
            "metadata": {
                "domain": self.domain,
                "topic": self.topic,
                "subtopic": self.subtopic,
                "knowledge_dependency": self.knowledge_dependency,
                "ambiguity_level": self.ambiguity_level,
                "question_length": self.question_length,
                "thinking_length": self.thinking_length,
                "answer_length": self.answer_length,
                "dialect_strength": self.dialect_strength,
                "terminology_level": self.terminology_level,
            },
        }

        if self.source:
            data["source"] = self.source.to_dict()

        if self.generation:
            data["generation"] = self.generation.to_dict()

        if include_teacher_only and self.teacher_only:
            data["teacher_only"] = self.teacher_only.to_dict()

        if self.quality_metrics:
            data["quality_metrics"] = self.quality_metrics.to_dict()

        if self.split:
            data["split"] = self.split.value

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SFTSample":
        """Create from dictionary (reverse of to_dict)"""
        teacher_only_data = data.get("teacher_only", {})
        teacher_only = TeacherOnlyKnowledge(**teacher_only_data) if teacher_only_data else None

        messages = [
            Message(role=m["role"], content=m["content"]) for m in data.get("messages", [])
        ]

        source_data = data.get("source", {})
        source = SourceMetadata(**source_data) if source_data else None

        generation_data = data.get("generation", {})
        generation = GenerationMetadata(**generation_data) if generation_data else None

        return cls(
            sample_id=data["sample_id"],
            chunk_id=data["chunk_id"],
            question=data["question"],
            thinking=data.get("thinking", ""),
            thinking_arabic=data.get("thinking_arabic", ""),
            thinking_english=data.get("thinking_english", ""),
            answer=data["answer"],
            messages=messages,
            question_type=data["question_type"],
            difficulty=data["difficulty"],
            reasoning_mode=data["reasoning_mode"],
            reasoning_depth=data["reasoning_depth"],
            teacher_only=teacher_only,
            language=data.get("language", "Arabic"),
            dialect=data.get("dialect", "Saudi"),
            domain=data.get("domain"),
            topic=data.get("topic"),
            subtopic=data.get("subtopic"),
            source=source,
            generation=generation,
            grounded=data.get("grounded", True),
            validation_status=data.get("validation_status", "Unverified"),
        )
