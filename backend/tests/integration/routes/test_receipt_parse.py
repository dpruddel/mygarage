"""Integration tests for opt-in LLM fuel receipt parse."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select

from app.models.fuel import FuelRecord
from app.models.settings import Setting


async def set_settings(db_session, settings_dict: dict[str, str]) -> None:
    for key, value in settings_dict.items():
        result = await db_session.execute(select(Setting).where(Setting.key == key))
        existing = result.scalar_one_or_none()
        if existing:
            existing.value = value
        else:
            db_session.add(Setting(key=key, value=value))
    await db_session.commit()


@pytest.mark.integration
@pytest.mark.asyncio
class TestReceiptParse:
    async def test_disabled_returns_403(
        self, client: AsyncClient, auth_headers, test_vehicle, db_session
    ):
        await set_settings(db_session, {"llm_receipt_parse_enabled": "false"})
        response = await client.post(
            f"/api/vehicles/{test_vehicle['vin']}/fuel/parse-receipt",
            data={"text": "Shell 12.4 gal $45.50"},
            headers=auth_headers,
        )
        assert response.status_code == 403
        assert "disabled" in response.json()["detail"].lower()

    async def test_mocked_llm_returns_draft_keys(
        self, client: AsyncClient, auth_headers, test_vehicle, db_session
    ):
        await set_settings(
            db_session,
            {
                "llm_receipt_parse_enabled": "true",
                "llm_base_url": "http://127.0.0.1:11434/v1",
                "llm_model": "llama3.2",
            },
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"date":"2026-08-01","liters":45.4,"cost":52.10,'
                                '"price_per_unit":1.147,"odometer_km":12345.0,'
                                '"station_name":"Shell","fuel_type_used":"Regular",'
                                '"kwh":null,"notes":null}'
                            )
                        }
                    }
                ]
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            response = await client.post(
                f"/api/vehicles/{test_vehicle['vin']}/fuel/parse-receipt",
                data={"text": "Shell station receipt"},
                headers=auth_headers,
            )

        assert response.status_code == 200
        body = response.json()
        assert body["source"] == "llm"
        draft = body["draft"]
        for key in (
            "date",
            "liters",
            "cost",
            "price_per_unit",
            "odometer_km",
            "station_name",
            "fuel_type_used",
            "notes",
            "kwh",
        ):
            assert key in draft
        assert draft["station_name"] == "Shell"
        assert draft["liters"] == 45.4

        count = await db_session.execute(
            select(func.count())
            .select_from(FuelRecord)
            .where(
                FuelRecord.vin == test_vehicle["vin"],
                FuelRecord.notes == "Shell station receipt",
            )
        )
        assert count.scalar_one() == 0

    async def test_oversized_file_returns_413(
        self, client: AsyncClient, auth_headers, test_vehicle, db_session
    ):
        await set_settings(db_session, {"llm_receipt_parse_enabled": "true"})
        with patch("app.routes.fuel.MAX_RECEIPT_UPLOAD_BYTES", 16):
            response = await client.post(
                f"/api/vehicles/{test_vehicle['vin']}/fuel/parse-receipt",
                files={"file": ("receipt.jpg", b"x" * 64, "image/jpeg")},
                headers=auth_headers,
            )
        assert response.status_code == 413
        assert (
            "too large" in response.json()["detail"].lower()
            or "exceeds" in response.json()["detail"].lower()
        )

    async def test_oversize_upload_is_never_read_into_memory(
        self, client: AsyncClient, auth_headers, test_vehicle, db_session
    ):
        await set_settings(db_session, {"llm_receipt_parse_enabled": "true"})
        # Patch starlette's UploadFile, NOT fastapi's. fastapi.datastructures
        # .UploadFile is a distinct subclass, but the multipart parser builds the
        # starlette base class regardless of the annotation, so patching the
        # subclass intercepts nothing (verified: await_count 1 vs 0).
        with (
            patch(
                "starlette.datastructures.UploadFile.read",
                new=AsyncMock(return_value=b""),
            ) as read_mock,
            patch("app.routes.fuel.MAX_RECEIPT_UPLOAD_BYTES", 16),
        ):
            response = await client.post(
                f"/api/vehicles/{test_vehicle['vin']}/fuel/parse-receipt",
                files={"file": ("receipt.jpg", b"x" * 64, "image/jpeg")},
                headers=auth_headers,
            )
        assert response.status_code == 413
        # The point of the fix: an oversize body is rejected on its spooled size,
        # never materialised as a full-size bytes object in memory.
        read_mock.assert_not_awaited()

    async def test_unreadable_image_reports_error_instead_of_calling_llm(
        self, client: AsyncClient, auth_headers, test_vehicle, db_session
    ):
        await set_settings(db_session, {"llm_receipt_parse_enabled": "true"})
        with (
            patch(
                "app.services.document_ocr.DocumentOCRService.extract_text_from_bytes",
                new=AsyncMock(side_effect=RuntimeError("tesseract exploded")),
            ),
            # Patch the constructor, not .post: the test client is itself an
            # httpx.AsyncClient, and patching the method would break its own
            # request. The service builds a fresh client inside the call, so a
            # constructor that is never invoked proves no LLM request happened.
            patch("httpx.AsyncClient") as llm_client,
        ):
            response = await client.post(
                f"/api/vehicles/{test_vehicle['vin']}/fuel/parse-receipt",
                files={"file": ("blurry.jpg", b"not-an-image", "image/jpeg")},
                headers=auth_headers,
            )
        assert response.status_code == 400
        assert "could not read" in response.json()["detail"].lower()
        # A receipt we could not read must not spend a 60s LLM call that can
        # only come back all-null, then report success to the user.
        llm_client.assert_not_called()

    async def test_ocr_yielding_nothing_says_so(
        self, client: AsyncClient, auth_headers, test_vehicle, db_session
    ):
        await set_settings(db_session, {"llm_receipt_parse_enabled": "true"})
        # document_ocr swallows a missing OCR dependency and returns "", so this
        # is the shape a user hits when tesseract is absent, not a synthetic case.
        with patch(
            "app.services.document_ocr.DocumentOCRService.extract_text_from_bytes",
            new=AsyncMock(return_value=""),
        ):
            response = await client.post(
                f"/api/vehicles/{test_vehicle['vin']}/fuel/parse-receipt",
                files={"file": ("blank.jpg", b"blank", "image/jpeg")},
                headers=auth_headers,
            )
        assert response.status_code == 400
        detail = response.json()["detail"].lower()
        # Telling someone who just uploaded a photo to "provide an image" is wrong.
        assert "could not read" in detail
        assert "provide text or an image file" not in detail

    async def test_oversized_text_returns_413(
        self, client: AsyncClient, auth_headers, test_vehicle, db_session
    ):
        await set_settings(db_session, {"llm_receipt_parse_enabled": "true"})
        with patch("app.routes.fuel.MAX_RECEIPT_TEXT_CHARS", 8):
            response = await client.post(
                f"/api/vehicles/{test_vehicle['vin']}/fuel/parse-receipt",
                data={"text": "this is longer than eight"},
                headers=auth_headers,
            )
        assert response.status_code == 413


def _png_bytes() -> bytes:
    import io

    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (40, 80), "white").save(out, format="PNG")
    return out.getvalue()


def _llm_response(content: str) -> MagicMock:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value={"choices": [{"message": {"content": content}}]})
    return response


# What gemma3:4b returned for a real Engen slip that prints litres and a total
# but no unit price: the date, time and amounts right, the price invented.
_VISION_REPLY = (
    '{"date":"2026-10-24","time":"08:20","station_name":"Engen Zaaymans Garage",'
    '"fuel_type_used":"Unleaded 95","liters":15.42,"cost":459.21,"price_per_unit":30.00}'
)


@pytest.mark.integration
@pytest.mark.asyncio
class TestReceiptParseVision:
    async def test_photo_goes_to_vision_model_without_ocr(
        self, client: AsyncClient, auth_headers, test_vehicle, db_session
    ):
        await set_settings(
            db_session,
            {
                "llm_receipt_parse_enabled": "true",
                "llm_model": "qwen2.5:7b",
                "llm_vision_model": "gemma3:4b",
            },
        )
        with (
            patch("httpx.AsyncClient") as mock_client,
            patch(
                "app.services.document_ocr.DocumentOCRService.extract_text_from_bytes",
                new=AsyncMock(return_value="should not be used"),
            ) as ocr,
        ):
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=_llm_response(_VISION_REPLY))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            response = await client.post(
                f"/api/vehicles/{test_vehicle['vin']}/fuel/parse-receipt",
                files={"file": ("slip.png", _png_bytes(), "image/png")},
                headers=auth_headers,
            )

        assert response.status_code == 200
        ocr.assert_not_awaited()
        payload = mock_instance.post.await_args.kwargs["json"]
        assert payload["model"] == "gemma3:4b"
        parts = payload["messages"][1]["content"]
        assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

        draft = response.json()["draft"]
        assert draft["time"] == "08:20"
        assert draft["liters"] == 15.42
        assert draft["cost"] == 459.21
        # Computed from the slip's own numbers, not the model's 30.00.
        assert draft["price_per_unit"] == 29.78
        # "Unleaded 95" is not in the vocabulary; better empty than wrong.
        assert draft["fuel_type_used"] is None

    async def test_failed_vision_call_falls_back_to_ocr_and_text_model(
        self, client: AsyncClient, auth_headers, test_vehicle, db_session
    ):
        import httpx

        await set_settings(
            db_session,
            {
                "llm_receipt_parse_enabled": "true",
                "llm_model": "qwen2.5:7b",
                "llm_vision_model": "missing-vision-model",
            },
        )
        text_reply = _llm_response(
            '{"date":"2026-10-24","liters":15.42,"cost":459.21,"fuel_type_used":"gasoline"}'
        )
        with (
            patch("httpx.AsyncClient") as mock_client,
            patch(
                "app.services.document_ocr.DocumentOCRService.extract_text_from_bytes",
                new=AsyncMock(return_value="Unleaded 95 P 7 15.42 R 459.21"),
            ) as ocr,
        ):
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(
                side_effect=[httpx.ConnectError("model not found"), text_reply]
            )
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            response = await client.post(
                f"/api/vehicles/{test_vehicle['vin']}/fuel/parse-receipt",
                files={"file": ("slip.png", _png_bytes(), "image/png")},
                headers=auth_headers,
            )

        assert response.status_code == 200
        ocr.assert_awaited_once()
        second = mock_instance.post.await_args_list[1].kwargs["json"]
        assert second["model"] == "qwen2.5:7b"
        assert isinstance(second["messages"][1]["content"], str)
        draft = response.json()["draft"]
        assert draft["fuel_type_used"] == "gasoline"
        assert draft["price_per_unit"] == 29.78
