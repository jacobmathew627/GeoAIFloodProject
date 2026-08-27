"""
Tests for the FastAPI backend.

This module had 0% coverage: every route was written and none was ever
exercised by a test. Manual curling confirmed all nine routes return 200
against the real artefacts, but nothing pinned that, so any refactor of
config paths, visualization helpers or the raster loader could have broken
the API silently while the 400-test suite stayed green.

These use FastAPI's TestClient, which drives the ASGI app in-process -- no
server, no port, no network. Routes that need large rasters are marked
`requires_model` and skipped when the artefacts are absent, matching the
convention in pytest.ini, so the suite still passes on a fresh checkout.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import MODELS_DIR, OUTPUT_DIR, RAINFALL  # noqa: E402


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    import backend

    return TestClient(backend.app)


def _hazard_exists(mm) -> bool:
    return (OUTPUT_DIR / f"flood_hazard_{int(mm)}mm.tif").exists()


class TestRoutesAlwaysAvailable:
    """Routes that must work with no model artefacts present at all."""

    def test_health_reports_status(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        # "degraded" is a legitimate answer on a fresh checkout; what must not
        # happen is a 500 or a missing key.
        assert body["status"] in ("ok", "degraded")
        for key in (
            "hazard_scenarios_available",
            "susceptibility_model_present",
            "susceptibility_surface_present",
        ):
            assert key in body

    def test_scenarios_lists_every_configured_depth(self, client):
        r = client.get("/api/scenarios")
        assert r.status_code == 200
        body = r.json()
        assert [s["rainfall_mm"] for s in body] == list(RAINFALL.scenarios)
        assert all(isinstance(s["available"], bool) for s in body)

    def test_places_returns_map_defaults(self, client):
        r = client.get("/api/places")
        assert r.status_code == 200
        body = r.json()
        assert len(body["map_center"]) == 2
        assert body["places"]

    def test_index_serves_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_openapi_schema_builds(self, client):
        """
        A malformed type annotation on any route makes schema generation throw,
        which breaks /docs without breaking the routes themselves.
        """
        r = client.get("/openapi.json")
        assert r.status_code == 200
        assert r.json()["info"]["title"] == "GeoAI Flood Risk API"


class TestRunoff:
    """Pure computation -- no rasters needed, so this is always exercised."""

    def test_returns_physical_runoff(self, client):
        r = client.get("/api/runoff", params={"rainfall_mm": 150.0})
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["runoff_depth_mm"] <= 150.0
        assert 0.0 <= body["runoff_coefficient"] <= 1.0

    def test_more_rain_never_yields_less_runoff(self, client):
        low = client.get("/api/runoff", params={"rainfall_mm": 50.0}).json()
        high = client.get("/api/runoff", params={"rainfall_mm": 300.0}).json()
        assert high["runoff_depth_mm"] >= low["runoff_depth_mm"]

    def test_zero_rain_gives_zero_runoff_and_no_division_error(self, client):
        r = client.get("/api/runoff", params={"rainfall_mm": 0.0})
        assert r.status_code == 200
        body = r.json()
        assert body["runoff_depth_mm"] == pytest.approx(0.0)
        # Guarded in the route: q / rainfall_mm would be ZeroDivisionError.
        assert body["runoff_coefficient"] == 0.0

    def test_impervious_sheds_more_than_pervious(self, client):
        pervious = client.get(
            "/api/runoff", params={"rainfall_mm": 150.0, "curve_number": 55.0}
        ).json()
        impervious = client.get(
            "/api/runoff", params={"rainfall_mm": 150.0, "curve_number": 95.0}
        ).json()
        assert impervious["runoff_depth_mm"] > pervious["runoff_depth_mm"]

    @pytest.mark.parametrize(
        "params",
        [
            {"rainfall_mm": -1.0},
            {"rainfall_mm": 99999.0},
            {"curve_number": 5.0},
            {"curve_number": 150.0},
        ],
    )
    def test_out_of_range_is_422_not_500(self, client, params):
        """
        Query(ge=..., le=...) must reject these at validation time. A 500 here
        would mean the bounds were dropped and nonsense reached the model.
        """
        assert client.get("/api/runoff", params=params).status_code == 422


class TestErrorHandling:
    def test_absent_scenario_is_404_with_guidance(self, client):
        r = client.get("/api/map/9999")
        assert r.status_code == 404
        # The message must say how to produce the missing artefact.
        assert "hazard" in r.json()["detail"].lower()

    def test_absent_scenario_stats_is_404(self, client):
        assert client.get("/api/risk_stats/9999").status_code == 404

    def test_non_integer_scenario_is_422(self, client):
        assert client.get("/api/map/abc").status_code == 422


@pytest.mark.requires_model
class TestRoutesNeedingArtefacts:
    def test_model_card_reports_metrics(self, client):
        if not (MODELS_DIR / "susceptibility_metrics.json").exists():
            pytest.skip("no susceptibility_metrics.json")
        r = client.get("/api/model")
        assert r.status_code == 200
        body = r.json()
        assert body["rainfall_response"]["reference_event_mm"] == RAINFALL.reference_event_mm

    def test_conformal_summary_available(self, client):
        path = MODELS_DIR / "susceptibility_metrics.json"
        if not path.exists():
            pytest.skip("no susceptibility_metrics.json")
        if not json.loads(path.read_text(encoding="utf-8")).get("conformal"):
            pytest.skip("model trained without conformal calibration")
        r = client.get("/api/conformal")
        assert r.status_code == 200
        assert "set_codes" in r.json()

    def test_risk_stats_percentages_sum_to_100(self, client):
        mm = next((m for m in RAINFALL.scenarios if _hazard_exists(m)), None)
        if mm is None:
            pytest.skip("no hazard rasters built")
        r = client.get(f"/api/risk_stats/{int(mm)}")
        assert r.status_code == 200
        body = r.json()
        total = sum(v for k, v in body.items() if k.endswith("_pct"))
        assert total == pytest.approx(100.0, abs=0.5)

    def test_map_returns_png_and_wgs84_bounds(self, client):
        mm = next((m for m in RAINFALL.scenarios if _hazard_exists(m)), None)
        if mm is None:
            pytest.skip("no hazard rasters built")
        r = client.get(f"/api/map/{int(mm)}")
        assert r.status_code == 200
        body = r.json()
        assert body["image_b64"]
        (lat_min, lon_min), (lat_max, lon_max) = body["bounds"]
        # Ernakulam sits near 10 N, 76.3 E. Anything else means the reprojection
        # from EPSG:32643 to WGS84 has broken.
        assert 9.0 < lat_min < lat_max < 11.5
        assert 75.5 < lon_min < lon_max < 77.5
