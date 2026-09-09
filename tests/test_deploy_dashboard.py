import json
from pathlib import Path

import pytest

streamlit = pytest.importorskip(
    "streamlit.testing.v1",
    reason="Streamlit no instalado"
)
AppTest = streamlit.AppTest

ROOT = Path(__file__).resolve().parent.parent
APP_PATH = ROOT / "deploy" / "app.py"
ASSETS = APP_PATH.parent / "assets"
PRED_DIR = ASSETS / "predictions"


def _load_stores() -> list[int]:
    with open(ASSETS / "stores.json") as f:
        return json.load(f)


def _load_families() -> dict[int, str]:
    with open(ASSETS / "families.json") as f:
        return {int(k): v for k, v in json.load(f).items()}


SKIP_ASSETS = not (
    PRED_DIR.exists()
    and any(PRED_DIR.glob("store_*_h7.parquet"))
    and ASSETS.joinpath("stores.json").exists()
    and ASSETS.joinpath("families.json").exists()
)


@pytest.mark.skipif(SKIP_ASSETS, reason="Assets no materializados")
class TestDeployDashboard:
    def test_sidebar_has_three_widgets(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        assert len(app.sidebar.selectbox) == 3

    def test_title_renders(self):
        app = AppTest.from_file(str(APP_PATH)).run()
        assert app.title[0].value == (
            "\U0001f6d2 Demand Forecast - Minimercados Ecuador"
        )

    def test_run_no_errors(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        assert not app.exception

    def test_selecting_family_no_crash(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        families = _load_families()
        if not families:
            pytest.skip("Sin familias en families.json")
        family_name = list(families.values())[0]
        app.sidebar.selectbox[2].set_value(family_name).run()
        assert not app.exception

    def test_toggle_horizon_no_crash(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        app.sidebar.selectbox[1].set_value(30).run()
        assert not app.exception

    def test_family_view_renders_dataframe(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        families = _load_families()
        if not families:
            pytest.skip("Sin familias en families.json")
        app.sidebar.selectbox[2].set_value("BEVERAGES").run()
        assert not app.exception
        assert len(app.dataframe) > 0
