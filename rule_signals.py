# -*- coding: utf-8 -*-
"""
rule_signals.py
================
الطبقة القاعدية (Rule Layer) لحكم DPO العربي.

هذه الطبقة لا تعتمد على أي نموذج، فقط على مكتبة بايثون القياسية
(re, difflib, collections). الهدف منها:
  1) أن تعمل دائما حتى لو لم يتوفر نموذج (fail-safe).
  2) أن تلتقط الحالات "السطحية" بدقة عالية بدل توريطها بنموذج ثقيل
     (verbosity, wrong_formatting, weak_organization, missing/unsupported).
  3) أن تنتج "أدلة" (evidence) قابلة للمراجعة البشرية، وليس رقما فقط.

لا يوجد هنا أي hardcoding لنتيجة حالة بعينها: كل قرار مبني على قياس
فعلي يُحسب من النصين (prompt/chosen/rejected) في وقت التشغيل.
"""

import re
import difflib
from collections import Counter

# --------------------------------------------------------------------------
# تطبيع عربي بسيط (تشكيل، ألف/همزة، مسافات)
# --------------------------------------------------------------------------

_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")
_TATWEEL = re.compile(r"\u0640")


def normalize(text: str) -> str:
    if not text:
        return ""
    t = _DIACRITICS.sub("", text)
    t = _TATWEEL.sub("", t)
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    t = t.replace("ى", "ي").replace("ة", "ه")
    t = re.sub(r"\s+", " ", t).strip()
    return t


AR_WORD = re.compile(r"[\u0621-\u064A]+")


def words(text: str) -> list:
    return AR_WORD.findall(normalize(text))


# --------------------------------------------------------------------------
# قوائم مؤشرات لغوية (يمكن توسيعها لاحقا دون تغيير المنطق)
# --------------------------------------------------------------------------

# كلمات دالة على العامية الشائعة عبر لهجات متعددة (وليست فصيحة)
DIALECT_MARKERS = {
    "ايه", "إيه", "علشان", "عشان", "يعني", "اللي", "بس", "مش",
    "ده", "دي", "كده", "شوي", "شوية", "خلص", "تتاخذ", "ببساطه",
    "لما", "هيك", "احنا", "انتوا", "وايد", "زين", "قد ايش",
}

# روابط/أدوات فصيحة تدل على بنية جملة موصولة (لا قائمة كلمات مفككة)
CONNECTIVES = {"و", "ف", "أي", "اي", "بمعني", "وقال", "وذكر", "التي", "الذي"}

# كلمات حشو/تضخيم أسلوبي تدل على الإطناب دون معلومة جديدة
FILLER_WORDS = {
    "جدا", "كثيرا", "مطولا", "مستفيضا", "قاطبه", "عظيم", "كبير",
    "واسع", "المعروفه", "المشهوره", "المتداوله",
}


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def char_similarity(a: str, b: str) -> float:
    a, b = normalize(a), normalize(b)
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def word_overlap(a_words, b_words) -> dict:
    ca, cb = Counter(a_words), Counter(b_words)
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return {
        "jaccard": inter / union if union else 1.0,
        "recall_of_a_in_b": inter / sum(ca.values()) if sum(ca.values()) else 1.0,
        "extra_in_b": list(((cb - ca)).elements()),
        "missing_from_b": list(((ca - cb)).elements()),
    }


def clause_split(text: str) -> list:
    """يقسّم النص إلى 'شبه-جمل' اعتمادا على الفواصل وواو العطف الرئيسة،
    ليُستخدم في كشف اضطراب الترتيب (weak_organization)."""
    norm = normalize(text)
    parts = re.split(r"[،,.؛]", norm)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def near_duplicate_but_not_identical(chosen: str, rejected: str, sim_floor=0.90):
    """يكشف 'فخ التشابه': نص فصيح جدا قريب حرفيا من النص الصحيح لكنه
    ليس مطابقا له تماما — إشارة قوية على partial_factual_errors إن لم
    يكن الفرق تنسيقيا بحتا (مسافة/علامة ترقيم)."""
    nc, nr = normalize(chosen), normalize(rejected)
    if nc == nr:
        return {"is_identical": True, "similarity": 1.0, "diffs": []}
    sim = char_similarity(nc, nr)
    diffs = []
    sm = difflib.SequenceMatcher(None, nc, nr)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            diffs.append({
                "op": tag,
                "chosen_span": nc[i1:i2],
                "rejected_span": nr[j1:j2],
            })
    return {"is_identical": False, "similarity": sim, "diffs": diffs,
            "near_duplicate": sim >= sim_floor}


def implausible_letter_run(word: str) -> bool:
    """كشف تكرار حرف عربي بشكل غير مألوف صرفيا (مثل 'يي' أو 'يّي')
    يمكن أن يشير إلى خطأ كتابي/تصحيفي داخل النص — يُستخدم لرصد
    الحالات المعطوبة حيث يحوي النص 'الذهبي' (chosen) خطأ فعليا."""
    return bool(re.search(r"(.)\1", word)) and word.endswith("يي")


def suspect_typo_tokens(text: str) -> list:
    return [w for w in words(text) if implausible_letter_run(w)]


def rule_based_evaluate(prompt: str, chosen: str, rejected: str) -> dict:
    """يعيد حزمة إشارات + تصنيف مبدئي + ثقة، بدون أي نموذج."""
    evidence = {}

    dup = near_duplicate_but_not_identical(chosen, rejected)
    evidence["duplicate_check"] = dup

    c_words = words(chosen)
    r_words = words(rejected)
    overlap = word_overlap(c_words, r_words)
    evidence["word_overlap"] = overlap

    len_ratio = (len(r_words) / len(c_words)) if c_words else float("inf")
    evidence["length_ratio_rejected_over_chosen"] = round(len_ratio, 3)

    chosen_typos = suspect_typo_tokens(chosen)
    rejected_typos = suspect_typo_tokens(rejected)
    evidence["suspect_typo_tokens"] = {"chosen": chosen_typos, "rejected": rejected_typos}

    dialect_hits = [w for w in r_words if w in DIALECT_MARKERS]
    evidence["dialect_markers_in_rejected"] = dialect_hits

    filler_hits = [w for w in r_words if w in FILLER_WORDS]
    evidence["filler_words_in_rejected"] = filler_hits

    connective_count_rejected = sum(1 for w in r_words if w in CONNECTIVES)
    has_punct_rejected = bool(re.search(r"[،.]", rejected or ""))
    evidence["formatting"] = {
        "connective_count_rejected": connective_count_rejected,
        "has_sentence_punct_rejected": has_punct_rejected,
    }

    chosen_clauses = clause_split(chosen)
    rejected_clauses = clause_split(rejected)
    # مؤشر أول (على مستوى شبه-الجمل): نفس الجمل الفرعية حرفيا لكن بترتيب مختلف
    same_clauses_diff_order = (
        Counter(chosen_clauses) == Counter(rejected_clauses)
        and chosen_clauses != rejected_clauses
        and len(chosen_clauses) > 1
    )
    # مؤشر ثانٍ (على مستوى الكلمات): نفس "كيس الكلمات" تماما (لا نقص ولا زيادة)
    # لكن الترتيب مختلف — يصمد حتى لو أُعيد تجميع الجمل بحدود مختلفة عن الأصل،
    # وهو ما يحدث فعليا عند خلط ترتيب الجمل الفرعية.
    same_bag_of_words_diff_order = (
        Counter(c_words) == Counter(r_words) and c_words != r_words
    )
    same_set_diff_order = same_clauses_diff_order or same_bag_of_words_diff_order
    evidence["reordering"] = {
        "same_clauses_different_order": same_clauses_diff_order,
        "same_bag_of_words_different_order": same_bag_of_words_diff_order,
        "chosen_clause_order": chosen_clauses,
        "rejected_clause_order": rejected_clauses,
    }

    # كشف "كيانات/استشهادات جديدة" حقيقية (وليس مجرد كلمات وصف عامة) —
    # يميّز unsupported_additions (يضيف مصطلحا معجميا/اسم علم جديدا)
    # عن verbosity (يعيد صياغة نفس الفكرة بكلام أطول دون معلومة جديدة).
    chosen_parens = set(re.findall(r"\(([^)]*)\)", chosen or ""))
    rejected_parens = set(re.findall(r"\(([^)]*)\)", rejected or ""))
    new_parenthetical_terms = list(rejected_parens - chosen_parens)
    chosen_qala_count = len(re.findall(r"\bقال\b", normalize(chosen)))
    rejected_qala_count = len(re.findall(r"\bقال\b", normalize(rejected)))
    evidence["new_entities"] = {
        "new_parenthetical_terms": new_parenthetical_terms,
        "extra_citation_markers": max(rejected_qala_count - chosen_qala_count, 0),
    }

    # هل يشير النص صراحة إلى "هذه المادة/الجذر" (أي يتفاعل مع طلب شرح
    # هذا المدخل تحديدا) أم يتحدث بعمومية عن الموضوع/الحقل؟
    references_this_entry = bool(
        re.search(r"هذه الماده|هذا الجذر|هذه الكلمه|هذه الفقره", normalize(rejected))
    )
    evidence["instruction_engagement"] = {"references_specific_entry": references_this_entry}

    # ------------------ منطق القرار (بلا أي تثبيت مسبق للنتيجة) ------------------
    label = None
    confidence = 0.0
    reasons = []

    # 1) حالات معطوبة في البيانات نفسها (قبل أي حديث عن rejection_type)
    if dup["is_identical"]:
        label = "INVALID_PAIR_IDENTICAL"
        confidence = 0.99
        reasons.append("chosen و rejected متطابقان تماما بعد التطبيع؛ لا يوجد فرق يبرر تفضيل أحدهما.")
    elif chosen_typos and not rejected_typos and dup["similarity"] > 0.9:
        label = "SUSPECT_QUALITY_REVERSAL"
        confidence = 0.75
        reasons.append(
            f"النص chosen يحوي صيغة غير مألوفة صرفيا {chosen_typos} بينما rejected خالٍ منها "
            "مع تشابه حرفي عالٍ جدا؛ هذا مؤشر على احتمال عكس أو دمج خاطئ لأزواج DPO."
        )

    # 2) فخ التشابه: قريب جدا حرفيا لكنه غير مطابق -> partial_factual_errors
    elif dup.get("near_duplicate") and rejected_typos:
        label = "partial_factual_errors"
        confidence = 0.85
        reasons.append(
            f"تشابه حرفي عالٍ جدا ({dup['similarity']:.2f}) مع وجود تحريف صرفي محلي "
            f"{rejected_typos} داخل rejected؛ هذا 'فخ تشابه' كلاسيكي: النص فصيح "
            "وقريب لكنه يحمل خطأ في التفاصيل وليس مطابقا دلاليا."
        )
    elif dup.get("near_duplicate") and dup["diffs"]:
        label = "partial_factual_errors"
        confidence = 0.6
        reasons.append(
            f"تشابه حرفي عالٍ ({dup['similarity']:.2f}) مع فروق موضعية دقيقة "
            f"{dup['diffs'][:2]} قد تمثل تحريفا في اسم/رقم/تفصيل وليست إعادة صياغة كاملة."
        )

    # 3) اضطراب الترتيب (نفس الجمل، ترتيب مختلف)
    elif same_set_diff_order:
        label = "weak_organization"
        confidence = 0.9
        reasons.append("نفس الجمل الفرعية بالضبط لكن بترتيب مختلف عن chosen، دون فقد أو إضافة معلومة.")

    # 4) تنسيق غير سليم: لا روابط ولا علامات ترقيم، مجرد كلمات مبعثرة
    elif connective_count_rejected == 0 and not has_punct_rejected and len(r_words) > 3:
        label = "wrong_formatting"
        confidence = 0.8
        reasons.append("النص مجرد سلسلة كلمات مفصولة بمسافات دون روابط نحوية أو علامات ترقيم تكوّن جملا.")

    # 5) عامية داخل سياق فصيح متوقع
    elif dialect_hits:
        label = "wrong_register"
        confidence = min(0.5 + 0.15 * len(dialect_hits), 0.9)
        reasons.append(f"وجود مؤشرات عامية {dialect_hits} في سياق يفترض أن يكون فصيحا (مادة معجمية كلاسيكية).")

    # 6) نقص معلومة: تغطية جزئية فقط من كلمات/جمل chosen، وrejected أقصر بوضوح
    elif overlap["recall_of_a_in_b"] < 0.75 and len_ratio < 0.75 and not overlap["extra_in_b"]:
        label = "missing_information"
        confidence = 0.75
        reasons.append(
            f"rejected يغطي فقط {overlap['recall_of_a_in_b']:.0%} من كلمات chosen وأقصر بوضوح "
            "دون إضافة معلومات جديدة — حذف جزء من المضمون."
        )

    # 7) إضافات غير مسندة: كيان/مصطلح جديد فعلي (قوس جديد أو استشهاد إضافي)
    #    لم يكن موجودا في chosen ولم يُطلب أصلا — إشارة أقوى وأدق من عدّ الكلمات.
    elif overlap["recall_of_a_in_b"] > 0.9 and len_ratio > 1.15 and (
        evidence["new_entities"]["new_parenthetical_terms"]
        or evidence["new_entities"]["extra_citation_markers"] > 0
    ):
        label = "unsupported_additions"
        confidence = 0.85
        reasons.append(
            "rejected يضيف كيانات/استشهادات جديدة غير مسندة إلى المصدر: "
            f"مصطلحات بين قوسين {evidence['new_entities']['new_parenthetical_terms']}, "
            f"استشهادات إضافية={evidence['new_entities']['extra_citation_markers']}."
        )

    # 8) إطناب أسلوبي: طول أكبر بكثير + نفس المضمون تماما (تغطية كاملة) + لا كيانات جديدة
    elif overlap["recall_of_a_in_b"] > 0.9 and len_ratio > 1.3 and not (
        evidence["new_entities"]["new_parenthetical_terms"]
        or evidence["new_entities"]["extra_citation_markers"] > 0
    ):
        label = "verbosity"
        confidence = min(0.6 + 0.05 * len(filler_hits), 0.9)
        reasons.append(
            f"rejected أطول من chosen بنسبة {len_ratio:.2f}x ويحافظ على كل كلمات المصدر تقريبا "
            f"({overlap['recall_of_a_in_b']:.0%}) دون إضافة أي كيان/معلومة جديدة — إطناب أسلوبي بحت. "
            f"كلمات حشو ملحوظة: {filler_hits}."
        )

    # 9) محتوى ضعيف الصلة: تغطية كلمات منخفضة جدا رغم طول مقارب — يحتاج طبقة دلالية
    if label is None and overlap["recall_of_a_in_b"] < 0.25:
        # المائز الرئيس: هل يتفاعل النص مع "هذه المادة/الجذر" تحديدا (يحاول أن
        # يجيب لكن بمحتوى غير صحيح/غير مطابق) أم يتجاهل التوجيه تماما ويعمم
        # عن الموضوع/الحقل ككل (مدح المعاجم، دور الدارسين، إلخ)؟
        generic_field_talk = bool(re.search(
            r"المعاجم|الدارسين|قديما وحديثا|هذا المجال|هذا الحقل", normalize(rejected)
        ))
        if evidence["instruction_engagement"]["references_specific_entry"]:
            label = "less_faithful_reconstruction"
            confidence = 0.65
            reasons.append(
                "rejected يشير صراحة إلى 'هذه المادة/الجذر' أي يتفاعل مع التوجيه، لكن مضمونه "
                "لا يطابق محتوى chosen الفعلي — إعادة بناء غير أمينة للمصدر."
            )
        elif generic_field_talk:
            label = "poor_instruction_following"
            confidence = 0.6
            reasons.append(
                "rejected يتحدث بعمومية عن حقل المعاجم/الدارسين دون الإشارة إلى هذه المادة "
                "بعينها ولا شرح مضمونها — تجاهل فعلي للتوجيه في الـprompt."
            )
        else:
            label = "less_faithful_reconstruction"
            confidence = 0.4
            reasons.append(
                "rejected فصيح ومترابط لكن تغطيته لكلمات/مضمون chosen ضعيفة جدا دون مؤشر واضح "
                "لنوع الانحراف — يحتاج تأكيدا من الطبقة الدلالية (embeddings)."
            )

    if label is None:
        label = "UNRESOLVED_NEEDS_MODEL_OR_REVIEW"
        confidence = 0.3
        reasons.append("لم تتوفر إشارة قاعدية حاسمة؛ يحتاج القرار إلى الطبقة الدلالية أو مراجعة بشرية.")

    return {
        "predicted_label": label,
        "confidence": round(confidence, 2),
        "reasons": reasons,
        "evidence": evidence,
    }
