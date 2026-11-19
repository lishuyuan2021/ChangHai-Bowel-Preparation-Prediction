"""Shared input and counterfactual constraints (no model fitting)."""
import math

DIET_DAYS_MIN = 0.0
DIET_DAYS_MAX = 2.0
DIET_DAYS_TARGETS = (0.0, 0.5, 1.0, 1.5, 2.0)
NORMAL_DIET = 4
BMI_MIN = 10.0
BMI_MAX = 50.0
TOLERANCE = 1e-9
PEG_VOLUME_LITRES = {1: 2.0, 2: 3.0, 3: 4.0}
PLAN_FIELDS = (
    "DietaryRestrictionDays", "BPEducationModality", "SplitDose_BP",
    "PreColonoscopyPhysicalActivity", "DietaryRestriction", "LaxativeRegimen",
    "BPtoColonoscopyinterval",
)


def finite_number(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def calculate_bmi(height_cm, weight_kg):
    height = finite_number(height_cm, "Height")
    weight = finite_number(weight_kg, "Weight")
    if height <= 0 or weight <= 0:
        raise ValueError("Height and weight must be greater than zero.")
    return weight / (height / 100.0) ** 2


def resolve_bmi(*, use_height_weight, height_cm=None, weight_kg=None, direct_bmi=None):
    if use_height_weight:
        if height_cm is None or weight_kg is None:
            return None
        bmi = calculate_bmi(height_cm, weight_kg)
    else:
        if direct_bmi is None:
            return None
        bmi = finite_number(direct_bmi, "BMI")
    if not BMI_MIN <= bmi <= BMI_MAX:
        raise ValueError(f"BMI must be between {BMI_MIN:g} and {BMI_MAX:g}; please check the entered values.")
    return bmi


def normalize_diet_days(diet_category, days):
    if int(diet_category) == NORMAL_DIET:
        return 0.0
    value = finite_number(days, "Dietary restriction days")
    if value < DIET_DAYS_MIN - TOLERANCE or value > DIET_DAYS_MAX + TOLERANCE:
        raise ValueError("Dietary restriction duration must be between 0 and 2 days.")
    return min(DIET_DAYS_MAX, max(DIET_DAYS_MIN, value))


def changed_plan_fields(original, candidate):
    return [field for field in PLAN_FIELDS
            if abs(float(original[field]) - float(candidate[field])) > TOLERANCE]


def peg_volume_reduced(original, candidate):
    before = PEG_VOLUME_LITRES.get(int(original["LaxativeRegimen"]))
    after = PEG_VOLUME_LITRES.get(int(candidate["LaxativeRegimen"]))
    # Different agents have no common dose scale in the saved categorical data.
    return before is not None and after is not None and after < before - TOLERANCE


def counterfactual_violations(original, candidate, *, exclude_fasting_over_1_day=True):
    """Validate the final plan, rather than the number of search actions.

    A normal-diet switch forces days to zero. That coupled change is not an
    additional supporting measure for a duration reduction by itself.
    """
    reasons = []
    days = finite_number(candidate["DietaryRestrictionDays"], "Dietary restriction days")
    if days < DIET_DAYS_MIN - TOLERANCE or days > DIET_DAYS_MAX + TOLERANCE:
        reasons.append("diet_days_out_of_range")
    if int(candidate["DietaryRestriction"]) == NORMAL_DIET and abs(days) > TOLERANCE:
        reasons.append("normal_diet_requires_zero_days")
    for field, reason in [("SplitDose_BP", "must_preserve_split_dose"),
                          ("PreColonoscopyPhysicalActivity", "must_preserve_activity")]:
        if int(original[field]) == 1 and int(candidate[field]) != 1:
            reasons.append(reason)

    changes = set(changed_plan_fields(original, candidate))
    days_reduced = days < float(original["DietaryRestrictionDays"]) - TOLERANCE
    dose_reduced = peg_volume_reduced(original, candidate)
    if days_reduced and dose_reduced:
        reasons.append("cannot_reduce_duration_and_peg_volume_together")
    if days_reduced:
        supporting = changes - {"DietaryRestrictionDays"}
        if int(candidate["DietaryRestriction"]) == NORMAL_DIET and int(original["DietaryRestriction"]) != NORMAL_DIET:
            supporting.discard("DietaryRestriction")
        if not supporting:
            reasons.append("duration_reduction_requires_another_independent_change")
    if dose_reduced and not (changes - {"LaxativeRegimen"}):
        reasons.append("peg_volume_reduction_requires_another_change")
    if exclude_fasting_over_1_day and changes.intersection({"DietaryRestriction", "DietaryRestrictionDays"}):
        if int(candidate["DietaryRestriction"]) == 1 and days >= 1.0 - TOLERANCE:
            reasons.append("no_fasting_for_one_day_or_more")
    return reasons
