# -*- coding: utf-8 -*-
"""
model_signals.py
=================
الطبقة الدلالية (Model Layer) لحكم DPO العربي.

تستخدم نماذج NLU عربية متخصصة (وليست نماذج توليد) لسد الفجوات التي
لا تستطيع القواعد وحدها تغطيتها بثقة: التشابه الدلالي الحقيقي (وليس
اللفظي فقط)، والتماسك بين الجمل، والتقارب الأسلوبي مع الفصحى/العامية.

النماذج المستخدمة (انظر التقرير المرفق لتبرير الاختيار):
  1) Omartificial-Intelligence-Space/GATE-AraBert-v1
     sentence-embeddings مدرّب على NLI+STS -> يعطي تشابها "دلاليا" حقيقيا
     يفرّق بين تشابه شكلي وتطابق معنى، وهذا ما يمنع "فخ التشابه".
  2) CAMeL-Lab/bert-base-arabic-camelbert-ca  و  ...-camelbert-da
     نموذجا Masked-LM مدرَّبان أحدهما على العربية الكلاسيكية والآخر على
     العربية العامية (تغريدات) -> نقارن درجة "ملاءمة" كل نص لكل نموذج
     عبر Pseudo-Log-Likelihood (PLL) كإشارة سجل (register) مستقلة عن
     قائمة الكلمات العامية اليدوية.

ملاحظة تشغيلية مهمة:
  بيئة التنفيذ الحالية (هذا الـsandbox) لا تملك اتصال شبكة، لذلك لا يمكن
  تنزيل هذه النماذج هنا فعليا. الكود أدناه صحيح ومكتمل ويعمل في أي بيئة
  فيها إنترنت (أو النماذج محمّلة محليا مسبقا)، لكنه في هذا التسليم يعمل
  بوضع "تدهور رشيق" (graceful degradation): إن تعذر تحميل النموذج، تُرجع
  الدالة None ويعتمد judge.py على الطبقة القاعدية فقط مع خفض الثقة
  ووضع علامة review_recommended=True، بدل افتراض نتيجة وهمية.
"""

from functools import lru_cache

try:
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
    _HAS_ML = True
except Exception:
    _HAS_ML = False

GATE_MODEL_ID = "Omartificial-Intelligence-Space/GATE-AraBert-v1"
CAMELBERT_CA_ID = "CAMeL-Lab/bert-base-arabic-camelbert-ca"
CAMELBERT_DA_ID = "CAMeL-Lab/bert-base-arabic-camelbert-da"


@lru_cache(maxsize=1)
def _load_gate():
    if not _HAS_ML:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(GATE_MODEL_ID)
    except Exception:
        return None


@lru_cache(maxsize=2)
def _load_mlm(model_id: str):
    if not _HAS_ML:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForMaskedLM.from_pretrained(model_id)
        model.eval()
        return tok, model
    except Exception:
        return None


def semantic_similarity(text_a: str, text_b: str):
    """تشابه دلالي (0..1) عبر GATE-AraBert-v1. يُرجع None إن تعذر تحميل النموذج."""
    m = _load_gate()
    if m is None:
        return None
    import numpy as np
    emb = m.encode([text_a, text_b], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))


def sentence_coherence(text: str):
    """تماسك تسلسل الجمل: متوسط تشابه كل جملة بالتي تليها (0..1).
    نص مُعاد ترتيبه عشوائيا يميل لمتوسط أقل من نص مرتّب منطقيا."""
    m = _load_gate()
    if m is None:
        return None
    import re
    import numpy as np
    clauses = [c.strip() for c in re.split(r"[،.؛]", text) if c.strip()]
    if len(clauses) < 2:
        return None
    emb = m.encode(clauses, normalize_embeddings=True)
    sims = [float(np.dot(emb[i], emb[i + 1])) for i in range(len(emb) - 1)]
    return sum(sims) / len(sims)


def pseudo_log_likelihood(text: str, model_id: str):
    """PLL تقريبي: نقنّع كل كلمة بدورها ونجمع log-prob الكلمة الصحيحة.
    كلما كان الرقم أعلى (أقرب للصفر) كان النص أكثر 'طبيعية' من منظور
    ذلك النموذج تحديدا (الكلاسيكية أو العامية)."""
    loaded = _load_mlm(model_id)
    if loaded is None:
        return None
    tok, model = loaded
    ids = tok(text, return_tensors="pt")["input_ids"][0]
    mask_id = tok.mask_token_id
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(1, len(ids) - 1):  # تجاوز [CLS]/[SEP]
            masked = ids.clone()
            true_tok = masked[i].item()
            masked[i] = mask_id
            out = model(masked.unsqueeze(0)).logits[0, i]
            logp = torch.log_softmax(out, dim=-1)[true_tok].item()
            total += logp
            count += 1
    return total / count if count else None


def register_score(text: str):
    """يقارن ملاءمة النص لنموذج الفصحى الكلاسيكية مقابل نموذج العامية.
    قيمة موجبة كبيرة = أقرب للعامية من المتوقع لسياق كلاسيكي."""
    pll_ca = pseudo_log_likelihood(text, CAMELBERT_CA_ID)
    pll_da = pseudo_log_likelihood(text, CAMELBERT_DA_ID)
    if pll_ca is None or pll_da is None:
        return None
    return {"pll_classical": pll_ca, "pll_dialectal": pll_da, "dialect_lean": pll_da - pll_ca}


def model_layer_available() -> bool:
    return _load_gate() is not None
