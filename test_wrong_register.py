# -*- coding: utf-8 -*-
"""
test_wrong_register.py
=======================
اختبار عام ومستقل لمنطق wrong_register في rule_signals.py /
arabic_dpo_judge.py.

الهدف: التأكد أن الكشف مبني على مؤشرات لغوية عامة (قائمة
DIALECT_MARKERS في rule_signals.py) تُطبَّق على أي نص عربي عامي يظهر
في rejected ضمن سياق يُفترض أن يكون فصيحا، وليس على مطابقة نص اختبار
بعينه محفوظ مسبقا داخل منطق الحكم.

لذلك، جميع النصوص هنا:
  - من مجالات مختلفة كليا عن dpo_pairs_classical_lexicon.jsonl.
  - تستخدم مؤشرات عامية موجودة أصلا في القائمة العامة (مثل: يعني، بس،
    إيه، زين، وايد، مش) لكن ضمن جمل جديدة كليا لم تُكتب من قبل.

يشمل الاختبار أيضا حالة "سلبي حقيقي" (true negative): نص فصيح تماما
بلا أي مؤشر عامية، يجب ألا يُصنَّف wrong_register رغم اختلافه عن
chosen بطرق أخرى.

تشغيل: python3 test_wrong_register.py
"""

from arabic_dpo_judge import judge_pair

CASES = [
    {
        "name": "لهجة خليجية عامة — تعريف الاستقصاء",
        "prompt": "اشرح معنى كلمة الاستقصاء.",
        "chosen": (
            "الاستقصاء هو البحث الدقيق عن الحقيقة والتقصي التام في الأمر "
            "حتى الوصول إلى غايته."
        ),
        "rejected": (
            "يعني الاستقصاء إنك تدور على الحقيقة وتفتش زين لين توصل للي "
            "تبيه، مو أكثر من كذا."
        ),
        "expect_verdict": "FAIL",
        "expect_type": "wrong_register",
        "expect_chosen_preferred": True,
    },
    {
        "name": "لهجة مصرية عامة — تعريف الانكسار الضوئي",
        "prompt": "عرّف ظاهرة الانكسار الضوئي في الفيزياء.",
        "chosen": (
            "الانكسار الضوئي هو تغيّر اتجاه الضوء عند انتقاله من وسط "
            "شفاف إلى وسط آخر يختلف عنه في الكثافة."
        ),
        "rejected": (
            "يعني كده لما الضوء يعدي من وسط لوسط تاني، بيتغير اتجاهه شوية، "
            "مش حاجة معقدة قوي."
        ),
        "expect_verdict": "FAIL",
        "expect_type": "wrong_register",
        "expect_chosen_preferred": True,
    },
    {
        "name": "لهجة شامية عامة — تعريف التمثيل الضوئي",
        "prompt": "اشرح عملية التمثيل الضوئي عند النباتات.",
        "chosen": (
            "التمثيل الضوئي عملية تحوّل فيها النباتات ضوء الشمس إلى طاقة "
            "كيميائية مخزَّنة في صورة سكريات."
        ),
        "rejected": (
            "يعني هيك النبات بياخد ضوء الشمس وبحوله لطاقة، بس الموضوع "
            "أبسط من هيك."
        ),
        "expect_verdict": "FAIL",
        "expect_type": "wrong_register",
        "expect_chosen_preferred": True,
    },
    {
        "name": "سلبي حقيقي: نص فصيح تماما بلا أي مؤشر عامية (يجب ألا يكون wrong_register)",
        "prompt": "عرّف مفهوم الجاذبية.",
        "chosen": "الجاذبية قوة تجذب الأجسام ذات الكتلة نحو بعضها البعض.",
        "rejected": "الجاذبية هي المسافة الفاصلة بين كوكبين في المجموعة الشمسية.",
        "expect_verdict": None,
        "expect_type": None,
        "expect_chosen_preferred": None,
        "must_not_be_wrong_register": True,
    },
]


def run():
    failures = []
    for case in CASES:
        r = judge_pair(case["prompt"], case["chosen"], case["rejected"], use_model_layer=True)
        print(f"\n=== {case['name']} ===")
        print("verdict:", r["verdict"], "| rejection_type:", r["rejection_type"],
              "| chosen_preferred_over_rejected:", r["chosen_preferred_over_rejected"],
              "| confidence:", r["confidence"], "| layer:", r["layer_used"])
        print("  dialect_markers_in_rejected:", r["evidence"].get("dialect_markers_in_rejected"))
        for reason in r["reasons"]:
            print("  -", reason)

        if case.get("expect_verdict") and r["verdict"] != case["expect_verdict"]:
            failures.append(
                f"{case['name']}: توقعنا verdict={case['expect_verdict']} وحصلنا {r['verdict']}"
            )
        if case.get("expect_type") and r["rejection_type"] != case["expect_type"]:
            failures.append(
                f"{case['name']}: توقعنا type={case['expect_type']} وحصلنا {r['rejection_type']}"
            )
        if case.get("expect_chosen_preferred") is not None and \
                r["chosen_preferred_over_rejected"] != case["expect_chosen_preferred"]:
            failures.append(
                f"{case['name']}: توقعنا chosen_preferred_over_rejected="
                f"{case['expect_chosen_preferred']} وحصلنا {r['chosen_preferred_over_rejected']}"
            )
        if case.get("must_not_be_wrong_register") and r["rejection_type"] == "wrong_register":
            failures.append(
                f"{case['name']}: صُنِّف خطأ كـ wrong_register رغم عدم وجود أي مؤشر عامية فعلي."
            )

    print("\n" + "=" * 60)
    if failures:
        print(f"فشل {len(failures)} من الاختبارات:")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)
    else:
        print("جميع اختبارات wrong_register نجحت.")


if __name__ == "__main__":
    run()
