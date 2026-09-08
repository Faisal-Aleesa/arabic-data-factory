"""
SFT Generation Prompts

Detailed system and user prompts that enforce the specification for:
- Question generation (Saudi Arabic, natural, diverse)
- Thinking generation (semantic reasoning only, no meta-commentary)
- Answer generation (complete, grounded, natural)
"""

import json
from typing import Dict, Any


class SFTPrompts:
    """Prompts for SFT sample generation"""

    SYSTEM_PROMPT = """You are generating SAUDI ARABIC training data for language models.

⚠️ CRITICAL CONSTRAINTS (NO EXCEPTIONS):
1. ✓ EVERY question MUST be in Saudi Arabic dialect (لهجة سعودية) — NOT MSA, NOT Egyptian
2. ✓ Questions MUST use natural Saudi phrasing: "شنو", "إيش", "ليش", "كيف", "متى"
3. ✓ Answers MUST be in Saudi Arabic dialect with natural Saudi phrasing
4. ✓ Never generate meta-commentary (لا تقل: "السؤال يسأل عن", "لازم نوضح", etc.)
5. ✓ Questions must be self-contained and answerable without source text
6. ✓ Answers must sound natural, not dataset-like or formal
7. ✓ Thinking is reasoning only—do NOT restate the question or answer
8. ✓ Preserve exact terminology from source when appropriate
9. ✓ Each sample must represent ONE primary intent

OUTPUT FORMAT:
Return valid JSON with this exact structure:
{
  "examples": [
    {
      "question": "SAUDI ARABIC ONLY",
      "thinking": "SAUDI ARABIC ONLY (semantic reasoning)",
      "thinking_arabic": "SAUDI ARABIC ONLY (copy of thinking for explicit bilingual storage)",
      "thinking_english": "ENGLISH (translation of the semantic reasoning)",
      "answer": "SAUDI ARABIC ONLY",
      "question_type": "...",
      "difficulty": "...",
      "reasoning_mode": "...",
      "reasoning_depth": 0-3,
      "support_facts": [...]
    }
  ]
}

BILINGUAL THINKING REQUIREMENT:
* thinking: Saudi Arabic semantic reasoning (main field)
* thinking_arabic: Explicit copy of Saudi Arabic thinking (same as "thinking")
* thinking_english: English explanation with Arabic terms preserved (keep العربية words, explain in English)
  ✓ CORRECT: "The word 'خوص' is a noun. The word 'أخوصت' is a verb. The difference..."
  ✗ WRONG: "The word 'khoss' is a noun. The word 'akhassatt' is a verb..." (transliteration)
* For Direct Recall (depth=0): All three can be empty
* For Medium/Hard: All three must be populated

QUESTION REQUIREMENTS:
* Language: SAUDI ARABIC DIALECT ONLY (لهجة سعودية فقط)

* APPROVED SAUDI INTERROGATIVES:
  وش، وشو، شنو، إيش، أيش (what)
  وشلون، شلون، كيف، كيفك (how)
  ليه، ليش (why)
  وين (where)
  متى (when)
  مين (who)

* APPROVED SAUDI PARTICLES & EXPRESSIONS:
  عطيني (give me)
  علمني (teach me / tell me)
  ورني (show me)
  فهمني (make me understand)
  أبي / أبغى / ودي (I want / I'd like)
  ما أبي / ما أبغى (I don't want)
  ممكن (is it possible)
  تقدر (can you)
  أكيد (sure / definitely)
  يعني (I mean / that is)
  بس (only / but)
  عاد (you know / emphasis)
  ترا / ترى (you know / keep in mind)
  الحين / دحين (now)
  بعدين (later)
  صح (correct / right)
  و، أو، لكن، حتى (conjunctions)

* Self-contained: Answerable without access to source
* Unambiguous: One clear primary intent
* Avoids revealing the answer
* Varied interrogative forms (شنو، إيش، كيف، متى، أي، هل، etc.)
* Varied sentence length (3–40 words, context-dependent)
* Natural phrasing: "وش معنى ...", "شنو ...", "كيف ...", etc.
* Realistic: What would a Saudi speaker actually ask?
* Use contractions & natural flow: "وش يعني", "الفرق بينهم", "أبي أعرف"

QUESTION TYPES:
- Definition: Request for exact meaning (e.g., "ما معنى كلمة X؟")
- Meaning: Semantic interpretation (e.g., "شنو قصد هذا التعبير؟")
- Explanation: How/why something works (e.g., "كيف يعمل ...؟")
- Clarification: Resolve ambiguity (e.g., "الفرق بين X و Y؟")
- Comparison: Contrast two concepts (e.g., "ما الفرق بين ...؟")
- Distinction: Highlight differences (e.g., "إيش اللي يفرق ...؟")
- Cause/Why: Explain causation (e.g., "ليش حصل ...؟", "شنو السبب ...؟") [use exactly: "Cause/Why"]
- Reasoning: Logical inference (e.g., "إذا كان X و Y، فإن ...؟")
- Inference: Draw conclusions (e.g., "بناءً على ... نستنتج ...؟")
- Application: Apply concept to scenario (e.g., "إذا حصل ... شنو النتيجة؟")
- Example: Request concrete instances (e.g., "اعطيني مثال على ...؟")
- Contextual Meaning: Meaning in specific context (e.g., "في سياق ...، شنو معنى ...؟") [use exactly: "Contextual Meaning"]
- Relationship: How concepts relate (e.g., "العلاقة بين ... و ...؟")
- Multi-step: Requires multiple reasoning steps [use exactly: "Multi-step"]
- Conceptual: Abstract understanding

DIFFICULTY:
- Easy: Direct recall or simple lookup. No reasoning chains required.
- Medium: Requires one inference or connection. Some context interpretation.
- Hard: Requires multi-step reasoning, synthesis, or sophisticated analysis.

REASONING MODES:
- Direct Recall: Retrieve a fact from memory (no reasoning needed)
- Contextual Interpretation: Understand meaning through context clues
- Definition Resolution: Match concept to definition
- Disambiguation: Resolve multiple possible meanings
- Comparison: Analyze similarities and differences
- Cause → Effect: Identify causation
- Evidence → Conclusion: Derive conclusion from evidence
- Concept → Application: Apply abstract concept to specific case
- Multi-step Inference: Chain multiple logical steps
- Relationship Resolution: Clarify how concepts relate

REASONING DEPTH:
- 0 (Direct): No reasoning; pure recall
- 1 (Single-step): One inference or connection
- 2 (Two-step): Two connected inferences
- 3 (Deep): Multiple inference steps

THINKING REQUIREMENTS:
* Language: SAUDI ARABIC DIALECT (لهجة سعودية)
* Capture the cognitive bridge between question and answer
* Structure: Relevant Knowledge → Context → Connection → Inference → Conclusion

* THINKING TEMPLATES BY DIFFICULTY:

  EASY (Direct Recall, depth=0):
    → Keep EMPTY or VERY SHORT (0–10 words)
    ✓ thinking: "" (empty)
    ✓ thinking_arabic: "" (empty)
    ✓ thinking_english: "" (empty)
    ✗ Don't add reasoning where none is needed

  MEDIUM (Single-step, depth=1):
    → 30–100 words (each)
    Structure:
    1. Identify the concept being asked about
    2. Note the relevant context or relationship
    3. Draw the single inference needed
    4. Reach the conclusion

    thinking + thinking_arabic (Saudi Arabic):
    "كلمة X تعني A. في السياق دي، A يرتبط بـ B. يا عني النتيجة إنها تدل على C."

    thinking_english (English with Arabic terms preserved):
    "The word X means A. In this context, A is related to B. So the result is that it indicates C."

  HARD (Multi-step, depth=2-3):
    → 70–180 words (each)
    Structure:
    1. State the core concept(s) involved
    2. Explain relationships between concepts
    3. Note relevant distinctions or special cases
    4. Chain the reasoning steps
    5. Reach the final conclusion

    thinking + thinking_arabic (Saudi Arabic):
    "نبدأ من: X يعني A. والشيء المهم هنا إن A ترتبط بـ B و C. العلاقة بين B و C تدل على D. فالخلاصة إن الإجابة هي E."

    thinking_english (English with Arabic terms preserved):
    "Let's start with: X means A. The important thing here is that A relates to both B and C. The relationship between B and C indicates D. So the conclusion is that the answer is E."

IMPORTANT FOR THINKING_ENGLISH:
* Keep all Arabic terms in original Arabic script (خوص، أخوصت، etc.)
* Use English grammar and explanatory words around them
* ✓ CORRECT: "The word 'خوص' is a noun. The word 'أخوصت' is a verb."
* ✗ WRONG: "The word 'khoss' is a noun. The word 'akhassatt' is a verb." (transliteration)
* ✗ WRONG: Full English translation of Arabic terms
* Purpose: English-speaking readers can understand the reasoning while preserving original terminology

* Be semantic, not meta: explain the reasoning, not the process
* Must NOT:
  - Restate the question
  - Paraphrase the answer repeatedly
  - Mention source, chunks, books, datasets, or AI
  - Meta-reasoning: "السؤال يسأل عن", "لازم نوضح", "المطلوب", "حسب النص", "السؤال هنا يطلب"
  - Meta-reasoning: "I need to find", "The text states", "This asks for"
  - Create artificial reasoning for simple samples
* For simple samples: Keep thinking empty or very short
* Use Saudi phrasing: "يعني", "قصده أن", "يا عني", "الحاصل إن", "والخلاصة إن"

ANSWER REQUIREMENTS:
* Language: SAUDI ARABIC DIALECT ONLY (لهجة سعودية فقط)
  ✓ Use: يعني، قال، تروح، تقول، لازم، شنو، إيش، وش، ليش، ودي، أبي، أبغى
  ✓ Use particles: الحين، دحين، بعدين، عاد، ترا، ترى، بس، بدري، متأخر
  ✗ Avoid: MSA conjugations, Egyptian dialect like "معنى" (use Saudi equivalents)

* ANSWER STRUCTURE TEMPLATES:

  DEFINITION / MEANING (Easy):
    → Direct Definition + Optional Context
    Structure:
    1. Start with direct definition: "X معناها / يعني A"
    2. Add context if helpful: "وهذا يستخدم في حالة B"
    3. Optional example: "مثلاً، في السياق الفلاني"
    Length: 10–60 words

  EXPLANATION (Medium):
    → How/Why it works
    Structure:
    1. Answer the how/why: "السبب إن... / الطريقة إنه..."
    2. Explain the mechanism: "لأن X يؤدي إلى Y"
    3. Note consequences: "والنتيجة إن..."
    4. Optional clarification: "يعني باختصار..."
    Length: 40–120 words

  COMPARISON / DISTINCTION (Medium-Hard):
    → Compare two concepts
    Structure:
    1. State the main difference: "الفرق الأساسي إن X يشير إلى A بينما Y يشير إلى B"
    2. Explain each concept: "X يعني... و Y يعني..."
    3. Highlight implications: "والفرق المهم هنا إن..."
    4. Optional example: "مثلاً، X يستخدم في... و Y يستخدم في..."
    Length: 50–150 words

  REASONING / MULTI-STEP (Hard):
    → Logical inference
    Structure:
    1. State the premise: "إذا كان X صحيح..."
    2. Add intermediate steps: "وبما أن Y يرتبط بـ Z..."
    3. Draw connection: "فإن هذا يدل على..."
    4. Reach conclusion: "والخلاصة إن..."
    5. Optional nuance: "لكن بشروط معينة... يعني..."
    Length: 80–220 words

* Complete: Fully address the question
* Correct: Factually accurate and grounded in source
* Grounded: Every fact traceable to support_facts
* Self-contained: Can stand alone with just the question
* Natural: Sound like a knowledgeable Saudi speaker, not a textbook

* OPENING VARIATIONS (vary these):
  ✓ "معناها...", "يعني...", "القصد إن...", "الحاصل إن...", "يا عني..."
  ✓ "الإجابة إن...", "يقصدون...", "هذا يعني...", "السبب إن...", "الفرق هو..."
  ✗ Never: "ببساطة", "المقصود", "باختصار" (avoid clichés)

* Length: Determined by information need, not arbitrary targets
  - Simple: 10–60 words
  - Medium: 40–120 words
  - Complex: 80–220 words
* Vary openings and structure
* Preserve terminology but explain if specialized
* Example Saudi phrasing: "يعني هذا...", "القصد أن...", "يا عني...", "ما معناه إلا..."

SUPPORT_FACTS:
* Atomic statements that justify the answer
* Each fact must be traceable to source
* Examples: ["كلمة X تعني Y", "المبدأ Z يقول أن ..."]

DIVERSITY REQUIREMENTS:
* Vary question types across samples
* Vary difficulty levels
* Vary reasoning modes
* Vary sentence structure
* Avoid templates and repeated phrasings
* Each sample should represent distinct knowledge

SAUDI ARABIC DIALECT EXAMPLES:

✓ CORRECT Saudi Arabic QUESTIONS (using approved vocabulary):
- "وش معنى كلمة X؟" or "شنو معنى كلمة X؟" (What's the meaning of X?)
- "إيش الفرق بين A و B؟" (What's the difference between A and B?)
- "شلون / وشلون يعمل هذا الشيء؟" (How does this work?)
- "ليش / ليه حصل كذا؟" (Why did this happen?)
- "متى نستخدم كلمة X؟" (When do we use X?)
- "في سياق كذا، وش معنى هذا؟" (In this context, what's this mean?)
- "عطيني مثال على هذا" (Give me an example)
- "اشرح لي الفرق بينهم" (Explain the difference to me)
- "علمني ليش يستخدمون كلمة X؟" (Teach me why they use word X)
- "ورني إيش القصد من هذا الكلام" (Show me what this means)
- "فهمني كيف يعمل هذا" (Make me understand how this works)
- "أبي أعرف الفرق بينهم" (I want to know the difference)

✗ WRONG (MSA or non-Saudi):
- "ما معنى الكلمة؟" (MSA - too formal)
- "كيفما يتم الاستعمال؟" (MSA passive voice)
- "هل يمكن إعطاء مثال؟" (formal, not Saudi conversational)

✓ CORRECT Saudi ANSWERS (structured, using approved particles):
Definition Example:
- "كلمة X معناها A. وهذا يستخدم في الحالات اللي فيها B."

Explanation Example:
- "السبب إن هذا يحصل لأن X يؤدي إلى Y. والنتيجة إنه يصير Z. يعني باختصار، الطريقة إنها كذا."

Comparison Example:
- "الفرق الأساسي إن A يشير إلى معنى معين بينما B يشير إلى معنى مختلف. A يستخدم في الحالة الفلانية و B يستخدم في الحالة الثانية."

✗ WRONG Saudi answers:
- "بناءً على النص المقدم..." (source reference)
- "يجب أن نلاحظ..." (MSA formality)
- "في الواقع..." (overly formal)

✓ CORRECT Saudi THINKING (structured by depth):
Direct Recall (empty or very short):
- "" (empty for easy questions)
- "معناها كذا" (short for easy questions)

Single-step Example:
- "كلمة X تعني المعنى A. وفي السياق دي، A يرتبط بـ B. يا عني المقصود إنها تدل على C."

Multi-step Example:
- "نبدأ من الأساس: X يعني A. والشيء المهم إن A ترتبط بـ B و C بشكل وثيق. العلاقة بين B و C تدل على D. لما نجمع كل هذا، نطلع إن الإجابة هي E."

✗ WRONG thinking:
- "السؤال يسأل عن..." (meta-commentary)
- "لازم نوضح..." (meta-commentary)
- "بناءً على المعلومات المقدمة..." (source reference)

QUALITY CHECKS:
Before returning a sample, verify:
1. ✓ Question is SAUDI ARABIC DIALECT ONLY (not MSA, not Egyptian)
2. ✓ Question uses Saudi interrogatives: شنو، إيش، كيف، ليش، متى، أين
3. ✓ Answer is SAUDI ARABIC DIALECT ONLY
4. ✓ Thinking (if present) is SAUDI ARABIC DIALECT
5. ✓ Question is self-contained and answerable alone
6. ✓ Question does not reveal the answer
7. ✓ Thinking is reasoning, not meta-commentary
8. ✓ Answer is complete and grounded
9. ✓ support_facts justify every claim in answer
10. ✓ No verbosity; every word earns its place
11. ✓ Sample teaches meaningful knowledge
12. ✓ No exact duplicates within batch

RETURN UP TO the provided ceiling. Quality over quantity.
⚠️ FINAL CHECK: Every Q, T, A must be in Saudi Arabic dialect. NO EXCEPTIONS."""

    @staticmethod
    def build_user_prompt(chunk: Dict[str, Any], max_examples: int = 12) -> str:
        """Build user prompt from a chunk"""
        return json.dumps(
            {
                "source_chunk_id": chunk.get("chunk_id"),
                "format_type": chunk.get("format_type"),
                "region": chunk.get("region"),
                "source_text": chunk.get("chunk_text"),
                "instructions": {
                    "max_examples": max_examples,
                    "prefer_quality_over_quantity": True,
                    "enforce_saudi_arabic": True,
                    "enforce_naturalness": True,
                },
            },
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def build_messages(chunk: Dict[str, Any], max_examples: int = 12) -> list:
        """Build complete message list for OpenRouter"""
        return [
            {"role": "system", "content": SFTPrompts.SYSTEM_PROMPT},
            {"role": "user", "content": SFTPrompts.build_user_prompt(chunk, max_examples)},
        ]


class DPOPrompts:
    """Prompts for DPO pair generation"""

    SYSTEM_PROMPT = """You are generating alternative (rejected) answers for Direct Preference Optimization (DPO) training.

⚠️ OBJECTIVE:
Create ONE plausible but meaningfully flawed SAUDI ARABIC answer that differs from the chosen answer in exactly ONE weakness.

⚠️ CRITICAL CONSTRAINTS:
1. ✓ The rejected answer must be SAUDI ARABIC DIALECT ONLY (لهجة سعودية فقط)
2. ✓ Must be plausible—not obviously absurd or nonsensical
3. ✓ Must differ from the chosen answer in exactly ONE meaningful weakness
4. ✓ Do not invent facts not supported by the source domain
5. ✓ The flaw must be subtle enough that it could mislead but not so obvious it's clearly wrong
6. ✓ Maintain Saudi Arabic dialect even when introducing the flaw

ALLOWED REJECTION TYPES:
- partial_factual_errors: Some claims are false; others are correct
- less_faithful_reconstruction: Captures spirit but loses important details or nuance
- unsupported_additions: Adds plausible but unsupported claims
- missing_information: Incomplete; omits essential information
- wrong_register: Technically correct but uses wrong tone/formality/dialect
- weak_organization: Correct information but poorly structured or explained
- poor_instruction_following: Doesn't fully address the specific question asked
- wrong_formatting: Violates formatting requirements if any
- verbosity: Correct but unnecessarily lengthy or repetitive

OUTPUT FORMAT:
{
  "rejected": "The alternative (flawed) answer in SAUDI ARABIC DIALECT ONLY",
  "rejection_type": "One of the allowed types above"
}

GENERATION STRATEGY:
1. Understand the chosen answer's logic and structure
2. Identify one specific weakness you can introduce:
   - Remove or alter one important detail
   - Add plausible but unsupported information
   - Simplify a nuanced explanation
   - Use wrong terminology or wrong dialect (but keep it Saudi)
   - Reorder information in a confusing way
3. Ensure the rejected answer still sounds natural and plausible IN SAUDI ARABIC
4. Keep it approximately the same length as the chosen answer
5. Maintain Saudi Arabic dialect throughout

DO NOT:
- Create answers that are obviously wrong
- Invent completely false information
- Create answers that contradict themselves
- Make the answer incoherent or grammatically broken
- Create answers that are worse in multiple dimensions
- Repeat phrases from thinking or support facts verbatim
- Switch to MSA or other dialects in the rejected answer"""

    @staticmethod
    def build_user_prompt(instruction: str, chosen: str, allowed_types: list) -> str:
        """Build user prompt for DPO generation"""
        return json.dumps(
            {
                "instruction": instruction,
                "chosen": chosen,
                "allowed_rejection_types": allowed_types,
                "constraints": {
                    "one_meaningful_weakness": True,
                    "plausibility_required": True,
                    "saudi_arabic_required": True,
                },
            },
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def build_messages(
        instruction: str, chosen: str, allowed_types: list
    ) -> list:
        """Build complete message list for OpenRouter"""
        return [
            {"role": "system", "content": DPOPrompts.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": DPOPrompts.build_user_prompt(instruction, chosen, allowed_types),
            },
        ]
