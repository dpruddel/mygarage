"""Fuel CSV import reads back ``Filled At`` and ``Station``.

The exporter has written both columns since migration 054, but the importer
dropped them, so a fuel CSV round trip lost every fill-up time and station.
``Station`` comes back as free text: ``Station ID`` points into one
instance's address book and means nothing in another, so it is ignored.
"""

from io import BytesIO

import pytest
from httpx import AsyncClient

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

HEADER = "Date,Filled At,Odometer (km),Volume (L),Price Per Unit (L),Total Cost,Full Tank,Station\n"


async def _import(client: AsyncClient, headers: dict, vin: str, csv_content: str) -> dict:
    r = await client.post(
        f"/api/import/vehicles/{vin}/fuel/csv",
        headers=headers,
        files={"file": ("fuel.csv", BytesIO(csv_content.encode()), "text/csv")},
    )
    assert r.status_code == 200, r.text
    return r.json()


async def _records_on(client: AsyncClient, headers: dict, vin: str, day: str) -> list[dict]:
    r = await client.get(f"/api/vehicles/{vin}/fuel", headers=headers)
    return [x for x in r.json()["records"] if x["date"] == day]


class TestFuelCsvFilledAtStation:
    async def test_filled_at_and_station_are_imported(self, client, auth_headers, test_vehicle):
        vin = test_vehicle["vin"]
        body = await _import(
            client,
            auth_headers,
            vin,
            HEADER + "2026-10-22,2026-10-22T14:35:00,301000,11.702,29.69,347.43,True,"
            "Astron Energy Villiersdorp\n",
        )
        assert body["success_count"] == 1, body

        [record] = await _records_on(client, auth_headers, vin, "2026-10-22")
        assert record["filled_at"].startswith("2026-10-22T14:35:00")
        assert record["station_name_freetext"] == "Astron Energy Villiersdorp"

    async def test_export_import_round_trip_keeps_time_and_station(
        self, client, auth_headers, test_vehicle
    ):
        vin = test_vehicle["vin"]
        r = await client.post(
            f"/api/vehicles/{vin}/fuel",
            headers=auth_headers,
            json={
                "vin": vin,
                "date": "2026-10-24",
                "filled_at": "2026-10-24T08:20:00",
                "odometer_km": 302000,
                "liters": 15.42,
                "cost": 459.21,
                "is_full_tank": True,
                "station_name_freetext": "Engen Zaaymans Garage",
            },
        )
        assert r.status_code == 201, r.text
        created = r.json()

        r = await client.get(
            f"/api/export/vehicles/{vin}/fuel/csv?units=metric", headers=auth_headers
        )
        assert r.status_code == 200, r.text
        exported = r.text

        r = await client.delete(f"/api/vehicles/{vin}/fuel/{created['id']}", headers=auth_headers)
        assert r.status_code in (200, 204), r.text

        body = await _import(client, auth_headers, vin, exported)
        assert body["error_count"] == 0, body

        [record] = await _records_on(client, auth_headers, vin, "2026-10-24")
        assert record["filled_at"].startswith("2026-10-24T08:20:00")
        assert record["station_name_freetext"] == "Engen Zaaymans Garage"

    async def test_blank_cells_and_old_files_leave_both_null(
        self, client, auth_headers, test_vehicle
    ):
        vin = test_vehicle["vin"]
        body = await _import(
            client,
            auth_headers,
            vin,
            HEADER + "2026-10-25,,303000,7.12,29.58,210.60,True,\n",
        )
        assert body["success_count"] == 1, body
        body = await _import(
            client,
            auth_headers,
            vin,
            "Date,Odometer (km),Liters,Price Per Liter,Total Cost,Full Tank\n"
            "2026-10-26,303500,10.49,29.78,312.40,True\n",
        )
        assert body["success_count"] == 1, body

        for day in ("2026-10-25", "2026-10-26"):
            [record] = await _records_on(client, auth_headers, vin, day)
            assert record["filled_at"] is None
            assert record["station_name_freetext"] is None

    async def test_date_comes_from_filled_at_when_date_is_blank(
        self, client, auth_headers, test_vehicle
    ):
        vin = test_vehicle["vin"]
        body = await _import(
            client,
            auth_headers,
            vin,
            HEADER + ",2026-10-27 08:53,304000,16.35,29.69,485.43,True,\n",
        )
        assert body["success_count"] == 1, body
        [record] = await _records_on(client, auth_headers, vin, "2026-10-27")
        assert record["filled_at"].startswith("2026-10-27T08:53:00")

    async def test_bad_or_mismatched_filled_at_fails_that_row_alone(
        self, client, auth_headers, test_vehicle
    ):
        vin = test_vehicle["vin"]
        body = await _import(
            client,
            auth_headers,
            vin,
            HEADER
            + "2026-10-28,yesterday,305000,11.82,29.38,347.25,True,\n"
            + "2026-10-29,2026-10-30T10:00:00,305100,11.82,29.38,347.25,True,\n"
            + "2026-10-31,2026-10-31T10:50:00,305200,11.82,29.38,347.25,True,Clives\n",
        )
        assert body["error_count"] == 2, body
        assert body["success_count"] == 1, body
        assert not await _records_on(client, auth_headers, vin, "2026-10-28")
        assert not await _records_on(client, auth_headers, vin, "2026-10-29")
        [record] = await _records_on(client, auth_headers, vin, "2026-10-31")
        assert record["station_name_freetext"] == "Clives"
