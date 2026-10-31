"""Opt-in LLM-assisted fuel receipt parsing (draft only — never writes FuelRecord).

Two ways in. A photo goes straight to ``llm_vision_model`` when one is set:
thermal receipts photographed on a cluttered background defeat tesseract often
enough (it drops whole lines, including the litres/amount line) that the text
model never sees the numbers. Otherwise, and for PDFs or when the vision call
fails, the photo is OCR'd and the text goes to ``llm_model`` as before.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.constants.fuel import FuelTypeEnum
from app.services import llm_client
from app.services.settings_service import SettingsService

logger = logging.getLogger(__name__)

OCR_FAILED_DETAIL = "Could not read any text from the image. Paste the receipt text instead."

_DRAFT_KEYS = (
    "date",
    "time",
    "odometer_km",
    "liters",
    "kwh",
    "cost",
    "price_per_unit",
    "fuel_type_used",
    "notes",
    "station_name",
)

# Long edge sent to the vision model. A phone photo is ~4000px; receipt text
# stays legible at 1600 and the request is a fraction of the size and time.
_VISION_MAX_EDGE = 1600

_TIME = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)")

_SYSTEM = (
    "You extract vehicle fuel/charge receipt fields. "
    "Reply with ONLY a JSON object using keys: "
    + ", ".join(_DRAFT_KEYS)
    + ". Use null for unknown values. Prefer metric: liters, odometer_km, price per liter. "
    "date is YYYY-MM-DD and time is HH:MM (24h), both as printed on the receipt. "
    "cost is the total amount paid. "
    "price_per_unit only if the receipt prints a price per liter or kWh; never estimate it. "
    "fuel_type_used is one of: "
    + ", ".join(t.value for t in FuelTypeEnum)
    + " (petrol and unleaded are gasoline). "
    "Do not mistake pump numbers, receipt numbers or card digits for liters or amounts."
)


async def parse_receipt_draft(
    db: AsyncSession,
    *,
    text: str | None = None,
    file_bytes: bytes | None = None,
    filename: str | None = None,
    content_type: str | None = None,
) -> dict[str, Any]:
    """Return ``{"draft": {...}, "source": "llm"}``. Raises 403 when disabled."""
    if not await SettingsService.get_bool(db, "llm_receipt_parse_enabled"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="LLM receipt parsing is disabled",
        )

    receipt_text = (text or "").strip()
    name = (filename or "").lower()
    is_pdf = name.endswith(".pdf") or (content_type or "").endswith("pdf")

    if not receipt_text and file_bytes and not is_pdf:
        raw = await _parse_photo_with_vision(db, file_bytes)
        if raw is not None:
            return {"draft": _finish_draft(raw), "source": "llm"}

    if not receipt_text and file_bytes:
        from app.services.document_ocr import DocumentOCRService

        ocr = DocumentOCRService()
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
    return {"draft": _finish_draft(raw), "source": "llm"}


async def _parse_photo_with_vision(db: AsyncSession, file_bytes: bytes) -> dict[str, Any] | None:
    """The vision model's fields for a receipt photo, or None to fall back to OCR.

    None when no vision model is configured, the bytes are not an image Pillow
    can open, or the vision call fails: OCR is still worth a try then, and its
    own errors are the ones the user should see.
    """
    model = (await llm_client.setting(db, "llm_vision_model", "")).strip()
    if not model:
        return None
    try:
        jpeg = await asyncio.to_thread(_photo_to_jpeg, file_bytes)
    except Exception as exc:  # noqa: BLE001 - Pillow raises broadly on bad input
        logger.info("Receipt photo could not be opened for vision (%s)", exc)
        return None
    try:
        content = await llm_client.chat_completion(
            db,
            system=_SYSTEM,
            user="Read this fuel receipt.",
            temperature=0,
            timeout=120.0,
            model=model,
            image_jpeg=jpeg,
        )
        return llm_client.extract_json_object(str(content))
    except HTTPException as exc:
        logger.warning("Vision receipt parse failed, falling back to OCR (%s)", exc.detail)
        return None


def _photo_to_jpeg(file_bytes: bytes) -> bytes:
    """Upright, downscaled RGB JPEG of an uploaded photo (HEIC included)."""
    from PIL import Image, ImageOps
    from pillow_heif import register_heif_opener

    register_heif_opener()
    with Image.open(io.BytesIO(file_bytes)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    image.thumbnail((_VISION_MAX_EDGE, _VISION_MAX_EDGE))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=90)
    return out.getvalue()


def _finish_draft(raw: dict[str, Any]) -> dict[str, Any]:
    """Keep the known keys, normalise them, and settle price/volume/cost.

    Models invent a "price per liter" when the receipt prints only litres and a
    total (seen: 30.00 for R459.21 / 15.42 L = 29.78), so whenever two of the
    three are present the third is computed and wins over the model's value.
    """
    draft = {k: raw.get(k) for k in _DRAFT_KEYS}

    time_match = _TIME.match(str(draft["time"] or "").strip())
    draft["time"] = f"{int(time_match.group(1)):02d}:{time_match.group(2)}" if time_match else None

    fuel_type = str(draft["fuel_type_used"] or "").strip().lower()
    draft["fuel_type_used"] = fuel_type if fuel_type in set(FuelTypeEnum) else None

    volume_key = (
        "kwh" if _num(draft["kwh"]) is not None and _num(draft["liters"]) is None else "liters"
    )
    volume = _num(draft[volume_key])
    cost = _num(draft["cost"])
    price = _num(draft["price_per_unit"])
    if volume and cost is not None:
        price = cost / volume
    elif volume and price is not None:
        cost = volume * price
    elif price and cost is not None:
        volume = cost / price

    draft[volume_key] = _round(volume, "0.001")
    draft["cost"] = _round(cost, "0.01")
    draft["price_per_unit"] = _round(price, "0.001")
    return draft


def _num(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        # No separator guessing: "29,78" is 29.78 in one locale and 2978 in
        # another, and a wrong guess is worse than leaving the field empty.
        number = Decimal(str(value).strip())
    except InvalidOperation:
        return None
    return number if number.is_finite() and number >= 0 else None


def _round(value: Decimal | None, places: str) -> float | None:
    if value is None:
        return None
    return float(value.quantize(Decimal(places), rounding=ROUND_HALF_UP))
