# -*- coding: utf-8 -*-
"""
test_similarity_trap.py
========================
اختبار مخصص لمشكلة "فخ التشابه": نص مرفوض فصيح جدا وقريب حرفيا من
الصحيح، لكنه يقلب معنى صغيرا (اسم علم مختلف، رقم مختلف، نفي مقحم، ...).
الهدف: التأكد أن الحكم لا "يمرّ" الحالة فقط لأنها تبدو شبيهة/فصيحة.

يشمل أيضا حالتي "PASS الحقيقي": إعادة صياغة سليمة تماما لا تستحق أي
رفض، وتشابه لفظي منخفض مع تشابه دلالي عالٍ (لا يجب معاقبته).

تشغيل: python3 test_similarity_trap.py
"""

from arabic_dpo_judge import judge_pair

PROMPT = "اشرح مضمون هذه المادة المعجمية."
CHOSEN = (
    "تعرض هذه المادة معاني الجذر (أتى)، فمنها أتى إليه إحسانا أي فعله، "
    "وأتى عليهم الدهر بمعنى أفناهم، وتذكر الإتاوة وهي الخراج والجباية، "
    "وقال جابر بن حنيّ التغلبي في ذلك بيتا."
)

CASES = [
    {
        "name": "خطأ إملائي صرفي دقيق في اسم العلم (تغلبيي بدل تغلبي)",
        "rejected": CHOSEN.replace("التغلبي", "التغلبيي"),
        "expect_verdict": "FAIL",
        "expect_type": "partial_factual_errors",
    },
    {
        "name": "استبدال اسم القبيلة بالكامل مع إبقاء بقية النص كما هو حرفيا",
        "rejected": CHOSEN.replace("التغلبي", "الطائي"),
        # تغيّر لفظي كبير جدا في اسم علم مع تشابه سياقي عالٍ — لا يزال
        # خطأ واقعيا يجب رفضه (النموذج الدلالي يفصل عند توفره).
        "expect_verdict": "FAIL",
        "expect_type": None,  # قد يُصنَّف partial_factual_errors أو NEEDS_REVIEW بالطبقة القاعدية وحدها
    },
    {
        "name": "إعادة صياغة سليمة كاملة (يجب ألا يُعامل كخطأ رغم اختلاف الألفاظ)",
        "rejected": (
            "تتناول هذه المادة دلالات الجذر (أتى)، فمن ذلك أتى إليه إحسانا "
            "بمعنى فعله، وأتى عليهم الدهر أي أفناهم، وتذكر الإتاوة وهي "
            "الخراج والجباية، وأنشد جابر بن حنيّ التغلبي بيتا في ذلك."
        ),
        # تشابه لفظي أقل من near_duplicate لكن لا خطأ واقعيا — النظام
        # القاعدي وحده سيحكم عليها NEEDS_REVIEW لأنه لا يملك تأكيدا دلاليا؛
        # وهنا بالضبط تُظهر أهمية الطبقة الدلالية لتفادي رفض إعادة صياغة سليمة.
        "expect_verdict": None,  # لا نفرض توقعا صارما بدون النموذج الدلالي
        "expect_type": None,
    },
    {
        "name": "زوج DPO معطوب: rejected مطابق لـ chosen تماما",
        "rejected": CHOSEN,
        "expect_verdict": "FAIL",
        "expect_type": "INVALID_PAIR_IDENTICAL",
    },
]


def run():
    failures = []
    for case in CASES:
        r = judge_pair(PROMPT, CHOSEN, case["rejected"], use_model_layer=True)
        print(f"\n=== {case['name']} ===")
        print("verdict:", r["verdict"], "| rejection_type:", r["rejection_type"],
              "| confidence:", r["confidence"], "| layer:", r["layer_used"])
        for reason in r["reasons"]:
            print("  -", reason)

        if case["expect_verdict"] and r["verdict"] != case["expect_verdict"]:
            failures.append(f"{case['name']}: توقعنا verdict={case['expect_verdict']} وحصلنا {r['verdict']}")
        if case["expect_type"] and r["rejection_type"] != case["expect_type"]:
            failures.append(f"{case['name']}: توقعنا type={case['expect_type']} وحصلنا {r['rejection_type']}")

        # الفحص الجوهري لفخ التشابه: مهما علت درجة التشابه اللفظي، يجب ألا
        # يكون verdict = PASS لحالة نعرف أنها تحوي خطأ فعليا في المعنى.
        if "خطأ" in case["name"] and r["verdict"] == "PASS":
            failures.append(f"{case['name']}: فخ التشابه لم يُكتشف — النظام أعطى PASS رغم وجود خطأ في المعنى!")

    print("\n" + "=" * 60)
    if failures:
        print(f"فشل {len(failures)} من الاختبارات:")
        for f in failures:
            print(" -", f)
    else:
        print("جميع اختبارات فخ التشابه نجحت (ضمن حدود الطبقة المتاحة).")


if __name__ == "__main__":
    run()
