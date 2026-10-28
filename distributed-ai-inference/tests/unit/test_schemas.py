import pytest
from pydantic import ValidationError

from app.schemas import PredictRequest


def test_valid_request():
    assert PredictRequest(features=[1, 2.5, -3]).features == [1.0, 2.5, -3.0]


@pytest.mark.parametrize(
    "bad",
    [
        {"features": []},
        {"features": ["abc"]},
        {"features": [float("nan")]},
        {"features": [float("inf")]},
        {"features": [1e9]},
        {"features": [0.0] * 1025},
        {"features": [1.0], "extra": 1},
        {},
    ],
)
def test_invalid_requests_rejected(bad):
    with pytest.raises(ValidationError):
        PredictRequest(**bad)
