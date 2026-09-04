# -*- coding: utf-8 -*-
"""
arabic_dpo_judge.py
====================
الـLLM/NLU Judge لبيانات DPO العربية (نمط الاستخدام: NLU، الإخراج
تصنيف مقيّد وليس نصا حرا مولَّدا).

الاستخدام:
    from arabic_dpo_judge import judge_pair
    result = judge_pair(prompt, chosen, rejected)

المخرجات: dict يحوي:
    verdict            : "PASS" | "FAIL" | "NEEDS_REVIEW"
    rejection_type      : أحد الأنواع القياسية أو None عند PASS
    confidence          : 0..1
    layer_used          : "rules_only" | "rules+model"
    evidence            : تفاصيل قابلة للتدقيق البشري
    reasons             : تفسير نصي قصير (بدون إعادة صياغة/توليد حر — جمل ثابتة الصياغة)
"""

from rule_signals import rule_based_evaluate, normalize
import model_signals as ml

VALID_REJECTION_TYPES = {
    "poor_instruction_following",
    "wrong_formatting",
    "missing_information",
    "unsupported_additions",
    "wrong_register",
    "verbosity",
    "weak_organization",
    "partial_factual_errors",
    "less_faithful_reconstruction",
}

# الحالات هذه ليست "rejection type" بالمعنى القياسي، بل عطب في الزوج نفسه
DATA_QUALITY_FLAGS = {"INVALID_PAIR_IDENTICAL", "SUSPECT_QUALITY_REVERSAL"}


def _apply_model_layer(prompt, chosen, rejected, rule_result):
    """تُستدعى فقط عند توفر النماذج (transformers+torch+اتصال شبكة).
    تُستخدم لترجيح/تصحيح قرار الطبقة القاعدية في الحالات الملتبسة
    (خصوصا poor_instruction_following مقابل less_faithful_reconstruction،
    وتأكيد/نفي wrong_register عبر إشارة مستقلة عن قائمة الكلمات)."""
    label = rule_result["predicted_label"]
    conf = rule_result["confidence"]
    notes = []

    sim = ml.semantic_similarity(chosen, rejected)
    if sim is not None:
        rule_result["evidence"]["semantic_similarity_gate"] = round(sim, 3)
        # فخ التشابه: تشابه لفظي عالٍ (rule) لكن تشابه دلالي منخفض نسبيا
        # يعني أن الفرق الصغير في الحروف غيّر المعنى فعلا (اسم/رقم خاطئ)
        if label == "partial_factual_errors" and sim < 0.97:
            notes.append(f"النموذج الدلالي يؤكد أن الفرق اللفظي الصغير غيّر المعنى (sim={sim:.3f}).")
            conf = min(conf + 0.1, 0.97)
        # العكس: تشابه لفظي منخفض لكن تشابه دلالي عالٍ => إعادة صياغة مقبولة وليست خطأ
        if label == "partial_factual_errors" and sim >= 0.97:
            notes.append(f"تشابه دلالي عالٍ جدا (sim={sim:.3f}) رغم الفرق اللفظي — قد يكون خطأ إملائيا بلا أثر دلالي؛ يُنصح بمراجعة بشرية بدل الحسم.")
            label = "NEEDS_REVIEW"
            conf = 0.5

    if label == "less_faithful_reconstruction" and sim is not None:
        if sim < 0.4:
            notes.append(f"تشابه دلالي منخفض جدا ({sim:.3f}) يؤكد ضعف الإخلاص للمصدر.")
            conf = max(conf, 0.75)
        elif sim > 0.7:
            notes.append(f"تشابه دلالي أعلى من المتوقع ({sim:.3f})؛ إعادة التصنيف كـ poor_instruction_following مرجّح أكثر.")
            label = "poor_instruction_following"
            conf = 0.6

    reg = ml.register_score(rejected)
    if reg is not None:
        rule_result["evidence"]["register_score"] = reg
        if reg["dialect_lean"] > 0 and label not in ("wrong_register",):
            notes.append(f"إشارة نموذج PLL تظهر ميلا للعامية (dialect_lean={reg['dialect_lean']:.3f}) لم تلتقطها القوائم اليدوية.")
        if reg["dialect_lean"] > 0 and label == "wrong_register":
            conf = min(conf + 0.1, 0.95)

    return label, conf, notes


def judge_pair(prompt: str, chosen: str, rejected: str, use_model_layer: bool = True) -> dict:
    if not chosen or not rejected:
        return {
            "verdict": "NEEDS_REVIEW",
            "rejection_type": None,
            "confidence": 0.0,
            "layer_used": "rules_only",
            "evidence": {},
            "reasons": ["نص chosen أو rejected فارغ."],
        }

    rule_result = rule_based_evaluate(prompt, chosen, rejected)
    label = rule_result["predicted_label"]
    confidence = rule_result["confidence"]
    reasons = list(rule_result["reasons"])
    layer_used = "rules_only"

    model_available = use_model_layer and ml.model_layer_available()
    if model_available:
        layer_used = "rules+model"
        label, confidence, model_notes = _apply_model_layer(prompt, chosen, rejected, rule_result)
        reasons.extend(model_notes)
    elif use_model_layer:
        reasons.append(
            "الطبقة الدلالية (GATE-AraBert-v1 / CAMeLBERT) غير متاحة في بيئة التشغيل الحالية؛ "
            "الحكم معتمد على الطبقة القاعدية فقط، وتم خفض الثقة تبعا لذلك."
        )
        if label in ("less_faithful_reconstruction", "UNRESOLVED_NEEDS_MODEL_OR_REVIEW"):
            confidence = min(confidence, 0.5)

    # تحويل التصنيف الداخلي إلى verdict نهائي
    if label in DATA_QUALITY_FLAGS:
        verdict = "FAIL"
        rejection_type = None
        reasons.append("علم جودة بيانات (data-quality flag) وليس rejection_type قياسيا — يحتاج تدقيق الزوج نفسه.")
        rejection_type = label  # نُبقي القيمة كعلم واضح للوحة التحكم
    elif label in ("NEEDS_REVIEW", "UNRESOLVED_NEEDS_MODEL_OR_REVIEW"):
        verdict = "NEEDS_REVIEW"
        rejection_type = None
    elif label in VALID_REJECTION_TYPES:
        verdict = "FAIL"  # rejected مرفوض بحق لسبب محدد => الزوج نفسه PASS كبيانات تدريب
        rejection_type = label
    else:
        verdict = "NEEDS_REVIEW"
        rejection_type = label

    return {
        "verdict": verdict,
        "rejection_type": rejection_type,
        "confidence": round(confidence, 2),
        "layer_used": layer_used,
        "chosen_preferred_over_rejected": (
            True if (verdict == "FAIL" and label in VALID_REJECTION_TYPES)
            else False if label in DATA_QUALITY_FLAGS
            else None
        ),
        "evidence": rule_result["evidence"],
        "reasons": reasons,
    }
