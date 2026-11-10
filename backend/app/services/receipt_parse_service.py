"""Opt-in LLM-assisted fuel receipt parsing (draft only, never writes FuelRecord).

A photo goes straight to the vision model when ``llm_vision_model`` is set and
the endpoint is on the household's network; text recognition dropped whole
lines of crumpled thermal slips. Otherwise, or when that call fails, the
receipt is read with OCR and its text sent to ``llm_model``, as before.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.constants.fuel import FuelTypeEnum
from app.models._types import ScaledNumeric
from app.models.fuel import FuelRecord
from app.services import document_images, llm_client
from app.services.settings_service import SettingsService
from app.utils.household_time import household_today
from app.utils.units import UnitConverter

logger = logging.getLogger(__name__)

OCR_FAILED_DETAIL = "Could not read any text from the image. Paste the receipt text instead."

# What we ask the LLM for: numbers and units exactly as printed. Models are bad
# at unit math, so the server does the converting (D4).
_PROMPT_KEYS = (
    "date",
    "time",
    "odometer",
    "odometer_unit",
    "volume",
    "volume_unit",
    "kwh",
    "cost",
    "price_per_unit",
    "price_volume_unit",
    "fuel_type_used",
    "notes",
    "station_name",
)

_TEXT_KEYS = ("date", "fuel_type_used", "notes", "station_name")

_FUEL_TYPES = frozenset(t.value for t in FuelTypeEnum)
_TIME = re.compile(r"([01]?[0-9]|2[0-3]):([0-5][0-9])(?::[0-5][0-9])?\s*([ap])?", re.IGNORECASE)
#: How far back a receipt's date may plausibly be when it is parsed.
_RECENT = timedelta(days=180)

# Spellings we trust, lowercased. Anything else and the value is null, since a
# guessed unit is worse than an empty field.
_UNIT_ALIASES = {
    "l": "L",
    "liter": "L",
    "liters": "L",
    "litre": "L",
    "litres": "L",
    "litre(s)": "L",
    "liter(s)": "L",
    "gal": "gal",
    "gallon": "gal",
    "gallons": "gal",
    "km": "km",
    "kilometer": "km",
    "kilometers": "km",
    "kilometre": "km",
    "kilometres": "km",
    "mi": "mi",
    "mile": "mi",
    "miles": "mi",
}


def _column_type(name: str) -> ScaledNumeric:
    """The FuelRecord column's type, so the draft rounds to what a save would store."""
    col_type = FuelRecord.__table__.c[name].type
    if not isinstance(col_type, ScaledNumeric):
        raise TypeError(f"FuelRecord.{name} is not a ScaledNumeric column")
    return col_type


_ODOMETER_COL = _column_type("odometer_km")
_LITERS_COL = _column_type("liters")
_PRICE_COL = _column_type("price_per_unit")
_COST_COL = _column_type("cost")
_KWH_COL = _column_type("kwh")

# Plain decimals only. Decimal() also takes "1_000", "1e3" and "NaN", and none
# of those is how a receipt prints a number. Fifteen whole digits is more than
# any column holds, and a million-digit one would overflow the mi/gal maths.
_PLAIN_DECIMAL = re.compile(r"-?[0-9]{1,15}(?:\.[0-9]+)?")
# A number as a slip prints it, which is how vision models return it: a
# currency sign or code before or after it ("R2,344.60", "29.37 EUR"), a
# per-unit suffix ("R29.37/L") and thousands separators, sometimes with a
# stray space beside one ("R2, 344.60").
_PRINTED_NUMBER = re.compile(
    r"(?:[A-Za-z]{1,3}\$?|[$\u20ac\u00a3\u00a5\u20b9\u20a6])?\s*"
    r"(?P<number>-?[0-9][0-9 .,]*?)\s*"
    r"(?:(?:/\s*)?[A-Za-z]{1,6}|[$\u20ac\u00a3\u00a5\u20b9\u20a6])?"
)
_GROUPED = {
    ",": re.compile(r"-?[0-9]{1,3}(?:,[0-9]{3})+"),
    ".": re.compile(r"-?[0-9]{1,3}(?:\.[0-9]{3})+"),
}
_SPACE_GROUPED = re.compile(r"-?[0-9]{1,3}(?: [0-9]{3})+(?:[.,][0-9]+)?")


def _printed_decimal(text: str) -> str | None:
    """A printed amount as a plain decimal string, or None when it is ambiguous.

    "2,344.60", "2.344,60" and "2 344.60" are all 2344.60: the last of a dot
    and a comma is the decimal point. A lone comma is a decimal comma ("1,85"),
    except before exactly three digits: "1,859" is 1859 on one slip and 1.859
    per litre on another, so that is None rather than a guess. Several dots or
    commas in groups of three are thousands ("1.234.567").
    """
    match = _PRINTED_NUMBER.fullmatch(text)
    if match is None:
        return None
    number = re.sub(r"\s*([.,])\s*", r"\1", match["number"])
    if " " in number:
        if not _SPACE_GROUPED.fullmatch(number):
            return None
        number = number.replace(" ", "")
    if "." in number and "," in number:
        decimal_mark = "." if number.rfind(".") > number.rfind(",") else ","
        whole, _, fraction = number.rpartition(decimal_mark)
        thousands = "," if decimal_mark == "." else "."
        if not fraction.isdigit() or not _GROUPED[thousands].fullmatch(whole):
            return None
        return whole.replace(thousands, "") + "." + fraction
    for mark in (",", "."):
        if number.count(mark) > 1:
            return number.replace(mark, "") if _GROUPED[mark].fullmatch(number) else None
    if "," in number:
        if _GROUPED[","].fullmatch(number):
            return None
        return number.replace(",", ".")
    return number


def _number(value: Any) -> Decimal | None:
    """An int, float or decimal string as a finite Decimal; anything else is None.

    A string may be a plain decimal or an amount as printed; see
    ``_printed_decimal``.
    """
    # bool is an int subclass, and Decimal(True) is 1. True is not 1 litre.
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        number = Decimal(repr(value))
        return number if number.is_finite() else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not _PLAIN_DECIMAL.fullmatch(text):
        text = _printed_decimal(text) or ""
    if _PLAIN_DECIMAL.fullmatch(text):
        return Decimal(text)
    return None


def _unit(value: Any, allowed: tuple[str, ...]) -> str | None:
    """The canonical unit for a printed spelling, if it's one of `allowed`."""
    if not isinstance(value, str):
        return None
    # A price's unit often comes back as written beside it: "per L", "/L".
    text = re.sub(r"^(?:per\s+|/\s*)", "", value.strip().lower())
    unit = _UNIT_ALIASES.get(text)
    return unit if unit in allowed else None


def _converted(value: Any, unit: Any, allowed: tuple[str, ...]) -> Decimal | None:
    """The value in canonical metric, or None when the number or unit is unusable."""
    number = _number(value)
    canonical_unit = _unit(unit, allowed)
    if number is None or canonical_unit is None:
        return None
    return UnitConverter.to_canonical_decimal(number, canonical_unit)


def _volume_unit(value: Any, household_unit: str | None) -> str | None:
    """The volume's unit: as printed, else the household's own.

    Slips often print the litres bare ("15.42 R 459.21"), and the model then
    reports the currency beside them as the unit. A printed word that is a
    volume unit we can't convert (imperial gallons) still nulls the volume:
    only a missing or non-volume unit falls back to ``household_unit``.
    """
    if (unit := _unit(value, ("L", "gal"))) is not None:
        return unit
    text = value.strip().lower() if isinstance(value, str) else ""
    # A unit of another family (km) is a misread field, not a bare volume.
    if text in _UNIT_ALIASES or any(word in text for word in ("l", "gal", "wh")):
        return None
    return household_unit


def _price_per_liter(value: Any, unit: Any) -> Decimal | None:
    """A per-litre or per-US-gallon price as a per-litre price."""
    number = _number(value)
    canonical_unit = _unit(unit, ("L", "gal"))
    if number is None or canonical_unit is None:
        return None
    if canonical_unit == "gal":
        return number / UnitConverter.US_GALLONS_TO_LITERS
    return number


def _stored(col: ScaledNumeric, value: Decimal | None) -> Decimal | None:
    """The value as the column would store it, or None if it won't fit.

    rounded() hands back a number too long to round untouched, and that one
    fails the width check too.
    """
    if value is None:
        return None
    rounded: Decimal = col.rounded(value)
    if rounded.adjusted() >= col.precision - col.scale:
        return None
    return rounded


def _positive(value: Decimal | None) -> Decimal | None:
    """Null for zero or below. A receipt never prints a zero odometer, volume or price."""
    return value if value is not None and value > 0 else None


def _non_negative(value: Decimal | None) -> Decimal | None:
    """Null below zero. Zero stays, since a free charge is a real receipt."""
    if value is None or value < 0:
        return None
    # "-0.00" passes the check above, and a free charge isn't negative.
    return value.copy_abs() if value.is_zero() else value


def _time(value: Any) -> str | None:
    """A printed time of day as canonical 24-hour HH:MM, or None ("4:03 PM" is 16:03)."""
    if not isinstance(value, str) or not (match := _TIME.match(value.strip())):
        return None
    hour, meridiem = int(match.group(1)), (match.group(3) or "").lower()
    if meridiem:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if meridiem == "p" else 0)
    return f"{hour:02d}:{match.group(2)}"


def _date(value: Any, today: date) -> str | None:
    """The model's YYYY-MM-DD, with a misread DD-MM-YY slip date put right.

    "22-10-26" (22 Oct 2026) comes back as 2022-10-26. A receipt being parsed
    is recent, so a date outside the last ``_RECENT`` days whose day and
    two-digit year, swapped, land inside them is taken swapped. Anything that
    isn't an ISO date passes through untouched, as before.
    """
    if not isinstance(value, str):
        return None
    try:
        parsed = date.fromisoformat(value.strip())
    except ValueError:
        return value
    if today - _RECENT <= parsed <= today:
        return parsed.isoformat()
    try:
        swapped = date(2000 + parsed.day, parsed.month, parsed.year % 100)
    except ValueError:
        return parsed.isoformat()
    return (swapped if today - _RECENT <= swapped <= today else parsed).isoformat()


def _settled(
    liters: Decimal | None,
    cost: Decimal | None,
    price: Decimal | None,
    *,
    price_printed: bool,
) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
    """Litres, cost and price per litre made consistent with each other.

    Slips print the litres and the total most reliably; the unit price is the
    figure models misread or invent. So a price that disagrees with
    total / litres by more than the slip's own rounding is replaced by it, and
    a missing one is worked out; any two give the third. A price that was
    printed in a unit we can't read stays null (``price_printed``), as a
    guessed unit would.
    """
    if liters and cost:
        if price is None and price_printed:
            return liters, cost, None
        # Unit prices are printed to 0.01 at worst, totals to 0.01.
        if price is None or abs(price * liters - cost) > liters * Decimal("0.005") + Decimal(
            "0.01"
        ):
            return liters, cost, cost / liters
        return liters, cost, price
    if liters and price and cost is None:
        return liters, liters * price, price
    if cost and price and liters is None:
        return cost / price, cost, price
    return liters, cost, price


def _build_draft(
    raw: dict[str, Any], household_volume_unit: str | None = None, today: date | None = None
) -> dict[str, Any]:
    """Turn the LLM's as-printed reply into the canonical draft.

    ``household_volume_unit`` (``L`` or ``gal``) is the volume unit assumed when
    the receipt prints none; see ``_volume_unit``. ``today`` (the household's,
    by default) anchors the slip-date repair in ``_date``.
    """
    odometer = _converted(raw.get("odometer"), raw.get("odometer_unit"), ("km", "mi"))
    volume_unit = _volume_unit(raw.get("volume_unit"), household_volume_unit)
    liters_raw = _converted(raw.get("volume"), volume_unit, ("L", "gal"))
    # The price is per the volume's unit when the model gives a non-unit
    # (the "R" of "R29.37/L") or none; settling below still checks it against
    # total / litres. A volume unit we can't convert (imp gal) stays null.
    price_raw = _price_per_liter(
        raw.get("price_per_unit"), _volume_unit(raw.get("price_volume_unit"), volume_unit)
    )
    # Settled only after the column checks, so nothing too big to store
    # yields a "price" from two absurd neighbours.
    liters, cost, price = _settled(
        _positive(_stored(_LITERS_COL, liters_raw)),
        _stored(_COST_COL, _non_negative(_number(raw.get("cost")))),
        _positive(_stored(_PRICE_COL, price_raw)),
        price_printed=raw.get("price_per_unit") is not None,
    )
    # A number where text belongs used to fail the response model with a 500.
    draft: dict[str, Any] = {
        key: value if isinstance(value := raw.get(key), str) else None for key in _TEXT_KEYS
    }
    draft.update(
        date=_date(raw.get("date"), today or household_today()),
        time=_time(raw.get("time")),
        odometer_km=_positive(_stored(_ODOMETER_COL, odometer)),
        liters=_positive(_stored(_LITERS_COL, liters)),
        price_per_unit=_positive(_stored(_PRICE_COL, price)),
        kwh=_stored(_KWH_COL, _non_negative(_number(raw.get("kwh")))),
        cost=_stored(_COST_COL, cost),
    )
    return draft


async def parse_receipt_draft(
    db: AsyncSession,
    *,
    text: str | None = None,
    file_bytes: bytes | None = None,
    filename: str | None = None,
    content_type: str | None = None,
    household_volume_unit: str | None = None,
) -> dict[str, Any]:
    """Return ``{"draft": {...}, "source": "llm"}``. Raises 403 when disabled.

    ``household_volume_unit`` is ``L`` or ``gal`` (US), the volume unit the
    person uses; a receipt that prints none is read in it.
    """
    if not await SettingsService.get_bool(db, "llm_receipt_parse_enabled"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="LLM receipt parsing is disabled",
        )

    receipt_text = (text or "").strip()
    if not receipt_text and file_bytes:
        raw = await _read_with_vision(db, file_bytes, filename, content_type)
        if raw is not None:
            return {"draft": _build_draft(raw, household_volume_unit), "source": "llm"}

        from app.services.document_ocr import DocumentOCRService

        ocr = DocumentOCRService()
        name = (filename or "").lower()
        is_pdf = name.endswith(".pdf") or (content_type or "").endswith("pdf")
        try:
            receipt_text = (
                await ocr.extract_text_from_bytes(file_bytes, is_pdf=is_pdf) or ""
            ).strip()
        except Exception as exc:  # noqa: BLE001 - OCR backends raise broadly
            # Deliberately NOT falling through with a placeholder. That spent a
            # 60s LLM call on text we never had, then returned an all-null draft
            # indistinguishable from a genuinely blank receipt.
            logger.info("Receipt OCR failed (%s)", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=OCR_FAILED_DETAIL,
            ) from exc

    if not receipt_text:
        # document_ocr swallows a missing OCR dependency and returns "", so a
        # user who did upload an image would otherwise be told to provide one.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=OCR_FAILED_DETAIL if file_bytes else "Provide text or an image file",
        )

    content = await llm_client.chat_completion(
        db,
        system=_SYSTEM,
        user=f"Receipt text:\n{receipt_text[:8000]}",
        temperature=0,
    )
    raw = llm_client.extract_json_object(str(content))
    return {"draft": _build_draft(raw, household_volume_unit), "source": "llm"}


_SYSTEM = (
    "You extract vehicle fuel/charge receipt fields. "
    "Reply with ONLY a JSON object using keys: "
    + ", ".join(_PROMPT_KEYS)
    + ". Give every number exactly as printed and never convert units. "
    "odometer_unit is km or mi, volume_unit is L or gal, and price_volume_unit "
    "is the unit price_per_unit is per (L or gal). Gallons are US gallons. "
    "date is YYYY-MM-DD and time is HH:MM (24h), as printed. cost is the total paid. "
    "Give price_per_unit only if the receipt prints it; never estimate it. "
    "fuel_type_used is one of: "
    + ", ".join(sorted(_FUEL_TYPES))
    + " (petrol and unleaded are gasoline). "
    "Do not mistake pump numbers, receipt numbers or card digits for volumes or amounts. "
    "station_name is the fuel retailer (brand and site), never the bank, card or "
    "payment terminal named on the slip. "
    "Use null for anything the receipt doesn't say, including a unit."
)


async def _read_with_vision(
    db: AsyncSession, file_bytes: bytes, filename: str | None, content_type: str | None
) -> dict[str, Any] | None:
    """The vision model's reply for a receipt image, or None to fall back to OCR.

    Opt-in by naming ``llm_vision_model``: blank keeps the OCR path, which is
    what a text-only ``llm_model`` needs. None too when the endpoint is outside
    the network (a receipt photo is not sent out without the consent the
    document reader asks for), the file is not an image or a scan, or the call
    fails: OCR is still worth a try, and its errors are the ones to show.
    """
    if not (await llm_client.setting(db, llm_client.LLM_VISION_MODEL_SETTING, "")).strip():
        return None
    if document_images.is_pdf(filename, content_type) and await asyncio.to_thread(
        document_images.has_text_layer, file_bytes
    ):
        return None
    try:
        base_url = await llm_client.require_vision_consent(db, allow_external=False)
        images = await asyncio.to_thread(
            document_images.to_images, file_bytes, filename, content_type, max_pages=1
        )
        content = await llm_client.vision_completion(
            db,
            system=_SYSTEM,
            user_text="Read this fuel receipt.",
            images=images,
            timeout=120.0,
            base_url=base_url,
        )
        return llm_client.extract_json_object(str(content))
    except (HTTPException, ValueError) as exc:
        logger.info("Receipt vision read failed, falling back to OCR (%s)", exc)
        return None
