# Arabic DPO Judge — ملفات التشغيل النهائية

## المحتوى
- `rule_signals.py` — الطبقة القاعدية (stdlib فقط، تعمل دائمًا).
- `model_signals.py` — الطبقة الدلالية (GATE-AraBert-v1 + CAMeLBERT-CA/DA)، تتفعّل تلقائيًا إن توفرت المكتبات + اتصال إنترنت.
- `arabic_dpo_judge.py` — المنسّق (`judge_pair`) الذي يدمج الطبقتين.
- `run_judge.py` — ينفّذ الحكم على كامل ملف jsonl وينتج `judge_results.jsonl`.
- `test_similarity_trap.py` — اختبارات مخصصة لفخ التشابه (partial_factual_errors).
- `test_weak_organization.py` — اختبار عام ومستقل لـ weak_organization، على نصوص جديدة كليًا (نحو/صرف، فوائد الرياضة، دورة الماء) لا علاقة لها ببيانات `dpo_pairs_classical_lexicon.jsonl`، لتأكيد أن الكشف مبني على قياس هيكلي (نفس الجمل/نفس كيس الكلمات بترتيب مختلف) وليس على مطابقة نص بعينه. يشمل حالتي سلبي حقيقي (true negative) أيضًا.
- `test_wrong_register.py` — اختبار عام ومستقل لـ wrong_register، على نصوص جديدة بلهجات مختلفة (خليجية، مصرية، شامية) في مجالات جديدة (فيزياء، أحياء) لتأكيد أن الكشف يعتمد على قائمة مؤشرات لغوية عامة وليس على نص اختبار محدد. يشمل حالة سلبي حقيقي أيضًا.
- `dpo_pairs_classical_lexicon.jsonl` — بيانات الإدخال الأصلية (11 سجلًا).
- `judge_results.jsonl` — نتائج تشغيل فعلي سابق (طبقة القواعد فقط، بلا اتصال شبكة وقتها).
- `dpo_judge_dashboard.html` — لوحة المراجعة البشرية (تُفتح مباشرة في المتصفح).
- `requirements.txt` — مكتبات الطبقة الدلالية فقط (اختيارية).

## التشغيل في Google Colab

```python
# 1) فك الضغط ثم الدخول للمجلد
!unzip -q LLM_Judge_Deliverables.zip -d judge && %cd judge

# 2) (اختياري) لتفعيل الطبقة الدلالية الفعلية
!pip install -q -r requirements.txt

# 3) تشغيل الحَكَم على البيانات المرفقة
!python run_judge.py dpo_pairs_classical_lexicon.jsonl judge_results.jsonl

# 4) اختبارات فخ التشابه
!python test_similarity_trap.py

# 5) اختبار weak_organization (عام ومستقل، نصوص جديدة كليًا)
!python test_weak_organization.py

# 6) اختبار wrong_register (عام ومستقل، نصوص جديدة كليًا)
!python test_wrong_register.py
```

ثم نزّل `dpo_judge_dashboard.html` وافتحه محليًا في المتصفح، أو استخدمه كـ Colab file preview.

## ملاحظة
إن لم تُثبَّت مكتبات `requirements.txt` (أو تعذّر تحميل النماذج من Hugging Face)،
يعمل النظام تلقائيًا بوضع `rules_only` بثقة مخفّضة بدل التوقف أو اختلاق نتيجة.
