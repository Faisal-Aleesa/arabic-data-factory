# -*- coding: utf-8 -*-
"""
test_weak_organization.py
==========================
اختبار عام ومستقل لمنطق weak_organization في rule_signals.py /
arabic_dpo_judge.py.

الهدف: التأكد أن الكشف مبني على قياس فعلي (نفس الجمل الفرعية أو نفس
"كيس الكلمات" لكن بترتيب مختلف) يُحسب وقت التشغيل، وليس على مطابقة
نص بعينه محفوظ مسبقا داخل منطق الحكم.

لذلك، جميع النصوص هنا:
  - من مجالات مختلفة كليا عن dpo_pairs_classical_lexicon.jsonl (لا علاقة
    بمعاجم كلاسيكية أو الجذر (أتى))، لتفادي أي شبهة "تحفيظ" أو hardcoding.
  - لا تُقارَن بسلسلة نصية ثابتة داخل rule_signals.py أو arabic_dpo_judge.py؛
    القرار يُشتق فقط من العلاقة الهيكلية بين chosen و rejected في وقت التشغيل.

يشمل الاختبار أيضا حالتي "سلبي حقيقي" (true negative):
  - نص لا يحوي إعادة ترتيب فعلية (يجب ألا يُصنَّف weak_organization).
  - نص معاد صياغته بكلمات مختلفة تماما (تغيير محتوى، وليس مجرد ترتيب)
    يجب ألا يُصنَّف weak_organization أيضا.

تشغيل: python3 test_weak_organization.py
"""

from arabic_dpo_judge import judge_pair

CASES = [
    {
        "name": "إعادة ترتيب جمل فرعية كاملة (فاصلة) — مجال النحو والصرف",
        "prompt": "اشرح الفرق بين النحو والصرف.",
        "chosen": (
            "النحو يدرس أواخر الكلمات وإعرابها، والصرف يدرس بنية الكلمة "
            "وتحولاتها، وكلاهما ضروري لفهم اللغة."
        ),
        "rejected": (
            "وكلاهما ضروري لفهم اللغة، والصرف يدرس بنية الكلمة وتحولاتها، "
            "النحو يدرس أواخر الكلمات وإعرابها"
        ),
        "expect_verdict": "FAIL",
        "expect_type": "weak_organization",
        "expect_chosen_preferred": True,
    },
    {
        "name": "إعادة ترتيب جمل فرعية بفواصل — مجال فوائد الرياضة",
        "prompt": "لخص فوائد الرياضة.",
        "chosen": "تحسن الرياضة صحة القلب, تقوي العضلات, تقلل التوتر النفسي",
        "rejected": "تقلل التوتر النفسي, تقوي العضلات, تحسن الرياضة صحة القلب",
        "expect_verdict": "FAIL",
        "expect_type": "weak_organization",
        "expect_chosen_preferred": True,
    },
    {
        "name": "إعادة ترتيب جمل فرعية — مجال دورة الماء في الطبيعة",
        "prompt": "اشرح دورة الماء في الطبيعة باختصار.",
        "chosen": (
            "يتبخر الماء من سطح البحار بفعل حرارة الشمس، ثم يتكاثف في "
            "الغيوم، ثم يسقط أمطارا على اليابسة."
        ),
        "rejected": (
            "ثم يسقط أمطارا على اليابسة، ثم يتكاثف في الغيوم، يتبخر الماء "
            "من سطح البحار بفعل حرارة الشمس"
        ),
        "expect_verdict": "FAIL",
        "expect_type": "weak_organization",
        "expect_chosen_preferred": True,
    },
    {
        "name": "سلبي حقيقي: لا إعادة ترتيب — نفس النص تماما (يجب ألا يكون weak_organization)",
        "prompt": "عرّف مفهوم الجاذبية.",
        "chosen": "الجاذبية قوة تجذب الأجسام ذات الكتلة نحو بعضها البعض.",
        "rejected": "الجاذبية قوة تجذب الأجسام ذات الكتلة نحو بعضها البعض.",
        "expect_verdict": "FAIL",  # زوج معطوب (تطابق تام) وليس weak_organization
        "expect_type": None,  # سيُصنَّف INVALID_PAIR_IDENTICAL وليس weak_organization
        "expect_chosen_preferred": None,
        "must_not_be_weak_org": True,
    },
    {
        "name": "سلبي حقيقي: تغيير محتوى فعلي لا مجرد ترتيب (يجب ألا يكون weak_organization)",
        "prompt": "عرّف مفهوم الجاذبية.",
        "chosen": "الجاذبية قوة تجذب الأجسام ذات الكتلة نحو بعضها البعض.",
        "rejected": "الجاذبية هي المسافة الفاصلة بين كوكبين في المجموعة الشمسية.",
        "expect_verdict": None,  # لا نفرض توقعا صارما على النوع هنا
        "expect_type": None,
        "expect_chosen_preferred": None,
        "must_not_be_weak_org": True,
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
        if case.get("must_not_be_weak_org") and r["rejection_type"] == "weak_organization":
            failures.append(
                f"{case['name']}: صُنِّف خطأ كـ weak_organization رغم عدم وجود إعادة ترتيب فعلية."
            )

    print("\n" + "=" * 60)
    if failures:
        print(f"فشل {len(failures)} من الاختبارات:")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)
    else:
        print("جميع اختبارات weak_organization نجحت.")


if __name__ == "__main__":
    run()
