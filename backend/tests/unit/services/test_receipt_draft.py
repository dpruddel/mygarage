"""How a model's raw receipt fields become the draft the fuel form receives."""

from __future__ import annotations

import pytest

from app.services.receipt_parse_service import _finish_draft


def test_price_is_computed_from_volume_and_total_over_the_models_guess():
    draft = _finish_draft({"liters": 15.42, "cost": 459.21, "price_per_unit": 30})
    assert draft["price_per_unit"] == 29.78


def test_missing_total_is_computed_from_volume_and_price():
    draft = _finish_draft({"liters": "40", "price_per_unit": "23.455"})
    assert draft["cost"] == 938.2


def test_missing_volume_is_computed_from_total_and_price():
    draft = _finish_draft({"cost": 469.10, "price_per_unit": 23.455})
    assert draft["liters"] == 20.0


def test_charge_receipt_settles_kwh_not_liters():
    draft = _finish_draft({"kwh": 40, "cost": 200})
    assert draft["price_per_unit"] == 5.0
    assert draft["liters"] is None


def test_zero_volume_does_not_divide():
    draft = _finish_draft({"liters": 0, "cost": 100, "price_per_unit": 20})
    assert draft["cost"] == 100.0
    assert draft["price_per_unit"] == 20.0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("08:20", "08:20"),
        ("8:05", "08:05"),
        ("23:59:10", "23:59"),
        ("24:00", None),
        ("noon", None),
        (None, None),
    ],
)
def test_time_is_normalised_to_hh_mm(raw, expected):
    assert _finish_draft({"time": raw})["time"] == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Diesel", "diesel"), ("gasoline", "gasoline"), ("Unleaded 95", None), (None, None)],
)
def test_fuel_type_is_kept_only_when_canonical(raw, expected):
    assert _finish_draft({"fuel_type_used": raw})["fuel_type_used"] == expected


@pytest.mark.parametrize("raw", ["29,78", "R 459", True, -5, "NaN"])
def test_unparseable_or_negative_numbers_become_none(raw):
    assert _finish_draft({"cost": raw})["cost"] is None
