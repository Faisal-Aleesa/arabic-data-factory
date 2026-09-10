"""
SFT Sample Generator

Core generation logic for creating SFT samples via OpenRouter API.
Handles API calls, response parsing, normalization, and validation.
"""

import json
import re
import time
import urllib.error
import urllib.request
from typing import Dict, Any, List, Optional, Tuple

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
)
from .prompts import SFTPrompts


# Meta-thinking phrases to filter out
META_THINKING_MARKERS = (
    "السؤال يسأل",
    "السؤال هنا",
    "لازم نوضح",
    "لازم أجاوب",
    "الجواب الأفضل",
    "المطلوب هو",
    "بناءً على المعلومات",
    "حسب الحقائق",
    "بناءً على النص",
    "حسب النص",
    "the user is asking",
    "i need to find",
    "the text states",
    "i should explain",
)


def extract_answer_without_scaffold(text: str) -> str:
    """
    Extract answer without thinking scaffold.
    Handles both formats:
    - "Thinking: ... Answer: ..." → returns just the answer part
    - "Answer: ..." → returns text after "Answer: " header
    - Plain text → returns as-is
    """
    if not text:
        return ""

    text = text.strip()

    # Check for "Answer:" header (both at start and mid-text)
    if "Answer:" in text:
        # Find where "Answer:" appears
        idx = text.find("Answer:")
        # Extract everything after "Answer:" and optional newline
        answer = text[idx + 7:].strip()
        # Remove "Thinking:" section if it comes before
        if answer.startswith("\n"):
            answer = answer[1:].strip()
        return answer

    # No scaffold found, return as-is
    return text


class OpenRouterClient:
    """Client for OpenRouter API"""

    def __init__(self, api_key: str, model: str = "google/gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 12000,
        timeout: int = 180,
    ) -> Tuple[str, str]:
        """
        Call OpenRouter API.
        Returns: (response_content, finish_reason)
        """
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            self.base_url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail[:500]}") from exc

        content = body["choices"][0]["message"]["content"].strip()
        finish_reason = body.get("choices", [{}])[0].get("finish_reason", "unknown")
        return content, finish_reason


class SFTGenerator:
    """SFT Sample Generator"""

    def __init__(self, api_key: str, model: str = "google/gemini-2.5-flash"):
        self.client = OpenRouterClient(api_key, model)
        self.model = model

    def generate_batch(
        self,
        chunks: List[Dict[str, Any]],
        limit: Optional[int] = None,
        max_examples_per_chunk: int = 12,
        pause_seconds: float = 0.5,
        progress_callback=None,
        continue_on_error: bool = True,
    ) -> List[SFTSample]:
        """
        Generate SFT samples from chunks.

        Args:
            chunks: List of source chunks
            limit: Max number of chunks to process (None = all)
            max_examples_per_chunk: Safety ceiling per chunk
            pause_seconds: Delay between API calls
            progress_callback: Function called with (current, total, chunk_id)
            continue_on_error: If True, skip failed chunks; if False, raise on error

        Returns:
            List of SFTSample objects
        """
        chunks_to_process = chunks[: limit or len(chunks)]
        samples = []
        errors = []

        for idx, chunk in enumerate(chunks_to_process, 1):
            chunk_id = chunk.get("chunk_id", f"chunk_{idx}")

            if progress_callback:
                progress_callback(idx, len(chunks_to_process), chunk_id, "Generating...")

            try:
                generated_samples = self._generate_chunk(
                    chunk, max_examples_per_chunk
                )
                samples.extend(generated_samples)

                if progress_callback:
                    progress_callback(
                        idx,
                        len(chunks_to_process),
                        chunk_id,
                        f"✓ {len(generated_samples)} sample(s)",
                    )
            except Exception as e:
                error_msg = str(e)[:50]
                if progress_callback:
                    progress_callback(idx, len(chunks_to_process), chunk_id, f"✗ Error: {error_msg}")

                errors.append((chunk_id, str(e)))

                if not continue_on_error:
                    raise
                # Continue to next chunk if continue_on_error is True

            if idx < len(chunks_to_process):
                time.sleep(pause_seconds)

        # Log summary if there were errors
        if errors:
            print(f"\n⚠️  {len(errors)} chunk(s) had errors and were skipped:")
            for chunk_id, error in errors[:5]:
                print(f"  - {chunk_id}: {error[:80]}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more")

        return samples

    def _generate_chunk(
        self, chunk: Dict[str, Any], max_examples: int = 12
    ) -> List[SFTSample]:
        """Generate samples from a single chunk"""
        chunk_id = chunk.get("chunk_id", "unknown")

        # Primary prompt
        messages = SFTPrompts.build_messages(chunk, max_examples)
        content, finish_reason = self.client.call(messages)

        generated_data = self._parse_response(content)

        # If primary prompt returns zero, try fallback
        if not generated_data.get("examples"):
            # Fallback with simplified prompt
            fallback_messages = self._build_fallback_messages(chunk, max_examples)
            content, _ = self.client.call(fallback_messages)
            generated_data = self._parse_response(content)

        # Normalize and validate samples
        normalized = self._normalize_samples(chunk, generated_data.get("examples", []))
        return normalized

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON response from model"""
        # Remove markdown code blocks if present
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE
        )
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return {"examples": []}

        if isinstance(parsed, list):
            return {"examples": parsed}
        if isinstance(parsed, dict) and isinstance(parsed.get("examples"), list):
            return parsed
        return {"examples": []}

    def _normalize_question_type(self, value: str) -> str:
        """Normalize question type to match schema"""
        if not value:
            return "Meaning"

        value = str(value).strip()

        # Handle variations in spacing/formatting
        normalizations = {
            "cause / why": "Cause/Why",
            "cause/why": "Cause/Why",
            "cause - why": "Cause/Why",
            "contextual meaning": "Contextual Meaning",
            "contextual-meaning": "Contextual Meaning",
            "multi-step": "Multi-step",
            "multi_step": "Multi-step",
            "definition/meaning": "Definition",
            "definition / meaning": "Definition",
            "meaning/definition": "Meaning",
            "meaning / definition": "Meaning",
        }

        value_lower = value.lower()
        for key, normalized in normalizations.items():
            if value_lower == key:
                return normalized

        return value

    def _normalize_samples(
        self, chunk: Dict[str, Any], raw_examples: List[Dict[str, Any]]
    ) -> List[SFTSample]:
        """Normalize and validate raw examples into SFTSample objects"""
        normalized = []
        seen_pairs = set()

        for idx, example in enumerate(raw_examples, 1):
            if not isinstance(example, dict):
                continue

            # Extract and clean fields
            question = (example.get("question") or "").strip()
            answer = (example.get("answer") or "").strip()
            thinking = self._clean_thinking(example.get("thinking", ""))
            thinking_arabic = self._clean_thinking(example.get("thinking_arabic", ""))
            thinking_english = (example.get("thinking_english") or "").strip()

            # Skip empty or duplicate samples
            if not question or not answer:
                continue
            if (question, answer) in seen_pairs:
                continue

            seen_pairs.add((question, answer))

            # Extract metadata with normalization
            try:
                question_type = self._normalize_question_type(example.get("question_type", "Meaning"))
                difficulty = example.get("difficulty", "Medium")
                reasoning_mode = example.get("reasoning_mode", "Direct Recall")
                reasoning_depth = example.get("reasoning_depth", 0)
                support_facts = example.get("support_facts", [])

                if not isinstance(support_facts, list):
                    support_facts = []

                # Build sample
                sample_id = f"{chunk['chunk_id']}_sft_{len(normalized) + 1:03d}"

                messages = [
                    Message(role="user", content=question),
                    Message(
                        role="assistant",
                        content=(
                            f"Thinking:\n{thinking}\n\nAnswer:\n{answer}"
                            if thinking
                            else f"Answer:\n{answer}"
                        ),
                    ),
                ]

                sample = SFTSample(
                    sample_id=sample_id,
                    chunk_id=chunk.get("chunk_id", "unknown"),
                    question=question,
                    thinking=thinking,
                    thinking_arabic=thinking_arabic if thinking_arabic else thinking,
                    thinking_english=thinking_english,
                    answer=answer,
                    messages=messages,
                    question_type=question_type,
                    difficulty=difficulty,
                    reasoning_mode=reasoning_mode,
                    reasoning_depth=reasoning_depth,
                    teacher_only=TeacherOnlyKnowledge(
                        support_facts=support_facts,
                        knowledge_units=example.get("knowledge_units", []),
                        required_facts=example.get("required_facts", support_facts),
                        optional_facts=example.get("optional_facts", []),
                        chunk_id=chunk.get("chunk_id"),
                        source_id=chunk.get("doc_id"),
                        source_span=chunk.get("source_pointer"),
                    ),
                    source=SourceMetadata(
                        chunk_id=chunk.get("chunk_id"),
                        doc_id=chunk.get("doc_id"),
                        region=chunk.get("region"),
                        format_type=chunk.get("format_type"),
                        source_pointer=chunk.get("source_pointer"),
                        license=chunk.get("license"),
                    ),
                    generation=GenerationMetadata(
                        model=self.model,
                        prompt_version="sft-v1-bilingual",
                    ),
                    validation_status=ValidationStatus.UNVERIFIED,
                )

                normalized.append(sample)
            except Exception as e:
                # Skip this example but continue processing others
                continue

        return normalized[:12]  # Respect ceiling

    def _clean_thinking(self, thinking: str) -> str:
        """Remove meta-thinking markers"""
        if not isinstance(thinking, str):
            return ""

        thinking = thinking.strip()
        text_lower = thinking.lower()

        # Remove if contains meta-thinking markers
        for marker in META_THINKING_MARKERS:
            if marker.lower() in text_lower:
                return ""

        return thinking

    def _build_fallback_messages(
        self, chunk: Dict[str, Any], max_examples: int
    ) -> List[Dict[str, str]]:
        """Build fallback message (simplified prompt)"""
        fallback_system = """Create grounded Saudi Arabic question-answer examples from this source.
Return JSON with examples list. Each item needs:
- question: Natural Saudi Arabic question
- answer: Complete Saudi Arabic answer
- thinking: Brief reasoning if needed (empty for direct answers)
- question_type: Definition/Meaning/Explanation/Comparison/Cause/etc.
- difficulty: Easy/Medium/Hard
- support_facts: Facts that justify the answer

Produce at least one example. Focus on quality. Use Saudi Arabic throughout."""

        user_content = json.dumps(
            {
                "source_text": chunk.get("chunk_text"),
                "max_examples": max_examples,
            },
            ensure_ascii=False,
        )

        return [
            {"role": "system", "content": fallback_system},
            {"role": "user", "content": user_content},
        ]


def load_chunks(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load chunks from JSONL file"""
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def save_samples(samples: List[SFTSample], output_path: str) -> None:
    """Save samples to JSONL file"""
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
