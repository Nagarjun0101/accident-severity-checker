"""Tests for the prediction logic (Step 14). Run with:  python -m pytest"""

from app import predictor

GOOD = dict(weather="Clear", lighting="Day", hour=17, weekday=2, month=6,
            visibility=10, temperature=70, wind_speed=5,
            junction=False, traffic_signal=False, crossing=False)


def predict(**changes):
    return predictor.predict_severity(**{**GOOD, **changes})


def test_model_is_loaded():
    assert len(predictor.COLUMNS) == 16
    assert predictor.CLASSES == [1, 2, 3, 4]


def test_output_has_all_fields():
    result = predict()
    for key in ["predicted_severity", "risk_score", "risk_level", "class_scores", "cutoffs", "note"]:
        assert key in result


def test_class_scores_sum_to_one():
    scores = predict()["class_scores"]
    assert len(scores) == 4
    assert abs(sum(scores.values()) - 1) < 0.01


def test_risk_score_is_between_0_and_1():
    for hour in range(24):
        assert 0 <= predict(hour=hour)["risk_score"] <= 1


def test_risk_level_matches_cutoffs():
    for hour in range(24):
        result = predict(hour=hour, lighting="Night" if hour >= 19 or hour < 6 else "Day")
        score = result["risk_score"]
        if score >= predictor.HIGH_CUTOFF:
            assert result["risk_level"] == "High"
        elif score >= predictor.MEDIUM_CUTOFF:
            assert result["risk_level"] == "Medium"
        else:
            assert result["risk_level"] == "Low"


def test_every_weather_option_works():
    for weather in predictor.WEATHER_OPTIONS:
        assert predict(weather=weather)["risk_level"] in ["Low", "Medium", "High"]


def test_unknown_weather_is_rejected():
    try:
        predict(weather="Tornado")
    except ValueError:
        return
    raise AssertionError("Unknown weather should raise ValueError")


def test_bad_lighting_is_rejected():
    try:
        predict(lighting="Dusk")
    except ValueError:
        return
    raise AssertionError("Bad lighting should raise ValueError")


def test_same_input_gives_same_output():
    assert predict() == predict()


def test_different_conditions_change_the_score():
    calm = predict(hour=11, traffic_signal=True)
    risky = predict(weather="Rain", lighting="Night", hour=22, weekday=5, junction=True)
    assert calm["risk_score"] != risky["risk_score"]


def test_extreme_values_do_not_crash():
    result = predict(visibility=0, temperature=-40, wind_speed=150, hour=0, month=12)
    assert result["risk_level"] in ["Low", "Medium", "High"]
