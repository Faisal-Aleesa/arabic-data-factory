"""
SFT Sample Validation and Quality Scoring

Validates samples against schema and computes quality metrics.
"""

from typing import List, Dict, Any, Tuple
import json
from .schema import SFTSample, ValidationStatus


class SFTValidator:
    """Validates SFT samples against schema and quality rules"""

    @staticmethod
    def validate_contract(samples: List[SFTSample]) -> Tuple[bool, List[str]]:
        """
        Validate that samples meet schema contract.
        Returns: (is_valid, list_of_errors)
        """
        errors = []
        required_fields = (
            "sample_id",
            "chunk_id",
            "question",
            "thinking",
            "answer",
            "messages",
            "question_type",
            "difficulty",
            "reasoning_mode",
            "reasoning_depth",
            "teacher_only",
            "validation_status",
        )

        seen_ids = set()

        for idx, sample in enumerate(samples, 1):
            # Check required fields
            for field in required_fields:
                if not hasattr(sample, field) or getattr(sample, field) is None:
                    errors.append(f"Sample {idx}: Missing {field}")

            # Check sample_id uniqueness
            if sample.sample_id in seen_ids:
                errors.append(f"Sample {idx}: Duplicate sample_id: {sample.sample_id}")
            seen_ids.add(sample.sample_id)

            # Check question and answer are non-empty strings
            if not isinstance(sample.question, str) or not sample.question.strip():
                errors.append(f"Sample {idx}: Question must be non-empty string")

            if not isinstance(sample.answer, str) or not sample.answer.strip():
                errors.append(f"Sample {idx}: Answer must be non-empty string")

            # Check thinking is string
            if not isinstance(sample.thinking, str):
                errors.append(f"Sample {idx}: Thinking must be string")

            # Check messages structure
            if not isinstance(sample.messages, list) or len(sample.messages) != 2:
                errors.append(f"Sample {idx}: Messages must be list with 2 elements")
            elif (
                sample.messages[0].role != "user" or sample.messages[1].role != "assistant"
            ):
                errors.append(
                    f"Sample {idx}: Messages must be [user, assistant], got [{sample.messages[0].role}, {sample.messages[1].role}]"
                )

            # Check validation status
            if sample.validation_status != ValidationStatus.UNVERIFIED:
                errors.append(
                    f"Sample {idx}: New samples must be UNVERIFIED, got {sample.validation_status}"
                )

            # Check teacher_only has support_facts
            if (
                not sample.teacher_only
                or not sample.teacher_only.support_facts
                or not isinstance(sample.teacher_only.support_facts, list)
            ):
                errors.append(f"Sample {idx}: Must have support_facts in teacher_only")

        return len(errors) == 0, errors

    @staticmethod
    def check_question_quality(question: str) -> Tuple[float, List[str]]:
        """
        Check question naturalness and quality.
        Returns: (score_0_to_1, list_of_issues)
        """
        issues = []
        score = 1.0

        # Check length
        word_count = len(question.split())
        if word_count < 3:
            issues.append("Too short (< 3 words)")
            score -= 0.2
        elif word_count > 40:
            issues.append("Too long (> 40 words)")
            score -= 0.1

        # Check for meta-language
        meta_markers = (
            "السؤال",
            "لازم",
            "المطلوب",
            "بناء",
            "حسب",
            "the question",
            "asking about",
        )
        if any(marker in question.lower() for marker in meta_markers):
            issues.append("Contains meta-language")
            score -= 0.3

        # Check for natural interrogatives
        natural_interrogatives = (
            "ما",
            "كيف",
            "متى",
            "أين",
            "من",
            "أي",
            "هل",
            "شنو",
            "إيش",
            "ليش",
        )
        if not any(interr in question for interr in natural_interrogatives):
            issues.append("Missing natural interrogative")
            score -= 0.1

        # Check Arabic text
        if not any(ord(c) >= 0x0600 and ord(c) <= 0x06FF for c in question):
            issues.append("Not in Arabic script")
            score -= 0.5

        return max(0.0, score), issues

    @staticmethod
    def check_answer_quality(answer: str, question: str) -> Tuple[float, List[str]]:
        """
        Check answer completeness and naturalness.
        Returns: (score_0_to_1, list_of_issues)
        """
        issues = []
        score = 1.0

        # Check length
        word_count = len(answer.split())
        if word_count < 5:
            issues.append("Too short (< 5 words)")
            score -= 0.2
        elif word_count > 250:
            issues.append("Too long (> 250 words)")
            score -= 0.1

        # Check for meta-language
        meta_markers = (
            "النص يقول",
            "حسب النص",
            "المصدر",
            "الكتاب",
            "البيانات",
            "بناءً على",
            "the text",
            "according to",
        )
        if any(marker in answer.lower() for marker in meta_markers):
            issues.append("Contains meta-language or source reference")
            score -= 0.3

        # Check for generic templates
        generic_templates = ("ببساطة", "المقصود", "باختصار", "يعني")
        template_count = sum(1 for tmpl in generic_templates if answer.startswith(tmpl))
        if template_count > 0:
            issues.append(f"Starts with generic template: {template_count}")
            score -= 0.1

        # Check Arabic text
        if not any(ord(c) >= 0x0600 and ord(c) <= 0x06FF for c in answer):
            issues.append("Not in Arabic script")
            score -= 0.5

        # Check if answer seems to address question
        # Simple heuristic: if question mentions a word and answer doesn't, might be misaligned
        q_words = set(question.split())
        a_words = set(answer.split())
        if len(q_words & a_words) == 0:
            issues.append("Question and answer may be misaligned")
            score -= 0.15

        return max(0.0, score), issues

    @staticmethod
    def check_thinking_quality(thinking: str, reasoning_depth: int) -> Tuple[float, List[str]]:
        """
        Check thinking (reasoning) quality.
        Returns: (score_0_to_1, list_of_issues)
        """
        issues = []
        score = 1.0

        if not thinking or not thinking.strip():
            # Direct recall should not have thinking
            if reasoning_depth == 0:
                return 1.0, []
            else:
                issues.append("Missing thinking for non-direct-recall sample")
                score -= 0.3
                return max(0.0, score), issues

        word_count = len(thinking.split())

        # Check length relative to reasoning_depth
        if reasoning_depth == 0:
            if word_count > 50:
                issues.append("Thinking too long for direct recall")
                score -= 0.2
        elif reasoning_depth == 1:
            if word_count > 100:
                issues.append("Thinking too long for single-step")
                score -= 0.1
        elif reasoning_depth >= 2:
            if word_count > 200:
                issues.append("Thinking too long even for multi-step")
                score -= 0.1

        # Check for meta-reasoning
        meta_markers = (
            "السؤال يسأل",
            "لازم",
            "المطلوب",
            "بناءً على",
            "حسب",
            "the question",
            "i need to",
        )
        if any(marker in thinking.lower() for marker in meta_markers):
            issues.append("Contains meta-reasoning")
            score -= 0.3

        return max(0.0, score), issues

    @staticmethod
    def check_grounding(answer: str, support_facts: List[str]) -> Tuple[float, List[str]]:
        """
        Check if answer is grounded in support_facts.
        Returns: (score_0_to_1, list_of_issues)
        """
        issues = []
        score = 1.0

        if not support_facts:
            issues.append("No support facts provided")
            score -= 0.5
            return max(0.0, score), issues

        # Simple heuristic: check if any support fact words appear in answer
        answer_words = set(answer.split())
        for fact in support_facts:
            fact_words = set(fact.split())
            if not (fact_words & answer_words):
                issues.append(f"Fact not reflected in answer: {fact[:50]}...")
                score -= 0.1

        return max(0.0, score), issues


class QualityScorer:
    """Computes quality metrics for samples"""

    @staticmethod
    def score_sample(sample: SFTSample) -> Dict[str, float]:
        """Compute quality scores for a sample"""
        scores = {}

        # Question naturalness
        q_score, _ = SFTValidator.check_question_quality(sample.question)
        scores["question_naturalness"] = q_score

        # Answer naturalness
        a_score, _ = SFTValidator.check_answer_quality(sample.answer, sample.question)
        scores["answer_naturalness"] = a_score

        # Thinking quality
        t_score, _ = SFTValidator.check_thinking_quality(
            sample.thinking, sample.reasoning_depth.value
        )
        scores["thinking_quality"] = t_score

        # Grounding
        g_score, _ = SFTValidator.check_grounding(
            sample.answer, sample.teacher_only.support_facts
        )
        scores["support_coverage"] = g_score

        # Derived scores
        scores["factual_correctness"] = 0.8  # Cannot verify without external source
        scores["answer_completeness"] = a_score
        scores["instruction_quality"] = q_score
        scores["response_quality"] = (a_score + t_score) / 2
        scores["saudi_dialect_quality"] = 0.7  # Would need linguistic evaluation
        scores["complexity_score"] = 0.5  # Context-dependent
        scores["diversity_score"] = 0.5  # Context-dependent (batch-level)
        scores["information_density"] = 0.7  # Heuristic estimate
        scores["redundancy_score"] = 0.2  # Heuristic (lower is better)
        scores["reasoning_correctness"] = t_score
        scores["similarity_to_nearest_sample"] = 0.5  # Context-dependent
        scores["teacher_confidence"] = (
            q_score * 0.2
            + a_score * 0.3
            + g_score * 0.2
            + t_score * 0.15
            + 0.15
        )

        return scores

    @staticmethod
    def score_batch(
        samples: List[SFTSample], verbose: bool = False
    ) -> Dict[str, Any]:
        """Score a batch of samples"""
        individual_scores = [QualityScorer.score_sample(sample) for sample in samples]

        # Compute batch statistics
        metric_names = set()
        for scores in individual_scores:
            metric_names.update(scores.keys())

        batch_stats = {}
        for metric in sorted(metric_names):
            values = [s.get(metric, 0.0) for s in individual_scores]
            batch_stats[metric] = {
                "mean": sum(values) / len(values) if values else 0.0,
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 0.0,
            }

        result = {
            "total_samples": len(samples),
            "individual_scores": individual_scores,
            "batch_statistics": batch_stats,
            "pass_rate": sum(
                1
                for s in individual_scores
                if s.get("teacher_confidence", 0.0) >= 0.6
            )
            / len(individual_scores)
            if samples
            else 0.0,
        }

        if verbose:
            print(f"Batch Quality Report")
            print(f"  Total samples: {result['total_samples']}")
            print(f"  Pass rate (confidence >= 0.6): {result['pass_rate']:.1%}")
            print(f"\nMetric Averages:")
            for metric in sorted(batch_stats.keys()):
                mean = batch_stats[metric]["mean"]
                print(f"  {metric}: {mean:.2f}")

        return result
