"""
Tests unitarios de la integración con el MLflow Model Registry.

Objetivo MLOps: validar la promoción con alias 'production' y la
recuperación de artefactos sin depender de DagsHub ni credenciales
(mlflow.* se mockea por completo). La validación real contra el
registry remoto es manual (ver docs/VALIDATION_GUIDE.md).
"""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from mlflow.exceptions import MlflowException

from src.models import registry
from src.models.registry import (
    PRODUCTION_ALIAS,
    ensure_local_artifacts,
    get_model_name,
    promote_local_artifacts,
)


def make_fake_artifacts(models_dir: Path, horizon: int = 7) -> list[Path]:
    """Crea los 3 artefactos vacíos que espera el registry."""
    names = [
        f"lgbm_h{horizon}.pkl",
        f"features_h{horizon}.pkl",
        f"feature_pipeline_h{horizon}.pkl",
    ]
    models_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        path = models_dir / name
        path.write_bytes(b"fake")
        paths.append(path)
    return paths


class FakeClient:
    """Sustituto de MlflowClient que registra las llamadas recibidas."""

    def __init__(self):
        self.aliases: list[tuple] = []
        self.tags: list[tuple] = []
        self.model_versions: list[dict] = []
        self.version_to_return = None

    def get_run(self, run_id):
        return SimpleNamespace(
            info=SimpleNamespace(
                run_id=run_id,
                artifact_uri="s3://fake-bucket/exp/run-123/artifacts"
            )
        )

    def create_model_version(self, name, source, run_id):
        self.model_versions.append(
            {"name": name, "source": source, "run_id": run_id}
        )
        return SimpleNamespace(version="3")

    def set_registered_model_alias(self, name, alias, version):
        self.aliases.append((name, alias, str(version)))

    def set_model_version_tag(self, name, version, key, value):
        self.tags.append((name, version, key, value))

    def get_model_version_by_alias(self, name, alias):
        if self.version_to_return is None:
            raise MlflowException(f"sin alias {alias}")
        return self.version_to_return


@pytest.fixture
def fake_mlflow(monkeypatch, tmp_path):
    """
    Reemplaza los puntos de contacto con mlflow dentro del módulo
    registry y expone el FakeClient para asserts.
    """
    client = FakeClient()

    @contextmanager
    def fake_start_run(run_name=None, nested=False):
        yield SimpleNamespace(
            info=SimpleNamespace(run_id="run-123")
        )

    monkeypatch.setattr(registry.mlflow, "start_run", fake_start_run)
    monkeypatch.setattr(registry.mlflow, "log_artifact", lambda *a, **k: None)
    monkeypatch.setattr(registry.mlflow, "log_metrics", lambda *a, **k: None)
    monkeypatch.setattr(registry.mlflow, "active_run", lambda: SimpleNamespace(
        info=SimpleNamespace(run_id="run-123")
    ))
    monkeypatch.setattr(registry, "MlflowClient", lambda: client)

    class FakeVersion(SimpleNamespace):
        pass

    registry._test_client = client
    yield client


class TestPromotion:
    def test_promote_registers_and_sets_production_alias(self, fake_mlflow, tmp_path):
        make_fake_artifacts(tmp_path)
        result = promote_local_artifacts(
            7, metrics={"mae": 0.1234}, models_dir=str(tmp_path)
        )
        assert result == "demand-forecast-daily@production (v3)"
        assert fake_mlflow.model_versions == [{
            "name": "demand-forecast-daily",
            "source": (
                "s3://fake-bucket/exp/run-123/artifacts"
                "/model/lgbm_h7.pkl"
            ),
            "run_id": "run-123",
        }]
        assert (registry.get_model_name(7), PRODUCTION_ALIAS, "3") \
            in fake_mlflow.aliases

        tag_keys = {(n, k) for n, v, k, _ in fake_mlflow.tags}
        assert ("demand-forecast-daily", "horizon") in tag_keys
        assert ("demand-forecast-daily", "test_mae") in tag_keys

    def test_promote_without_artifacts_returns_none(self, fake_mlflow, tmp_path):
        result = promote_local_artifacts(7, models_dir=str(tmp_path))
        assert result is None

    def test_promote_survives_registry_failure(self, monkeypatch, tmp_path):
        make_fake_artifacts(tmp_path)

        @contextmanager
        def failing_start_run(run_name=None, nested=False):
            raise MlflowException("DagsHub caído")
            yield

        monkeypatch.setattr(registry.mlflow, "start_run", failing_start_run)

        result = promote_local_artifacts(7, models_dir=str(tmp_path))
        assert result is None


class TestEnsureLocalArtifacts:
    def test_noop_when_all_present(self, fake_mlflow, tmp_path):
        make_fake_artifacts(tmp_path)
        assert ensure_local_artifacts(7, models_dir=str(tmp_path)) is True

    def test_fails_when_no_production_version(self, fake_mlflow, tmp_path):
        assert ensure_local_artifacts(7, models_dir=str(tmp_path)) is False

    def test_downloads_missing_from_registry(self, fake_mlflow, monkeypatch, tmp_path):
        staging = tmp_path / "downloaded" / "model"
        staging.mkdir(parents=True)
        make_fake_artifacts(staging)

        fake_mlflow.version_to_return = SimpleNamespace(
            version="2", run_id="run-old"
        )
        monkeypatch.setattr(
            registry.mlflow.artifacts,
            "download_artifacts",
            lambda run_id, artifact_path: str(staging),
        )

        target = tmp_path / "models"
        assert ensure_local_artifacts(7, models_dir=str(target)) is True
        assert (target / "lgbm_h7.pkl").exists()
        assert (target / "feature_pipeline_h7.pkl").exists()


class TestHelpers:
    def test_invalid_horizon_rejected(self):
        with pytest.raises(ValueError):
            get_model_name(15)

    def test_model_names_per_horizon(self):
        assert get_model_name(7).endswith("daily")
        assert get_model_name(30).endswith("monthly")


class TestModelRegistryFallback:
    def test_load_raises_when_nothing_available(self, monkeypatch, tmp_path):
        """Sin artefactos locales ni registry → error accionable."""
        from src.models.predict import ModelRegistry

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(ModelRegistry, "_models", {})
        monkeypatch.setattr(ModelRegistry, "_features", {})
        monkeypatch.setattr(ModelRegistry, "_pipelines", {})
        monkeypatch.setattr(
            "src.models.registry.ensure_local_artifacts",
            lambda horizon, models_dir="models": False,
        )

        with pytest.raises(FileNotFoundError, match="registry"):
            ModelRegistry.load(7)
