from pathlib import Path

from ots_compression.core.experiments import ExperimentSpec, ExperimentStore
from ots_compression.core.settings import Settings


def _spec() -> ExperimentSpec:
    return ExperimentSpec(
        domain="vision",
        task="classification",
        dataset={"name": "cifar10", "split": "train"},
        model={"architecture": "resnet18", "parameters": 10_000_000},
        optimizer={"name": "adamw", "lr": 3e-4, "weight_decay": 0.01},
        training={"steps": 1_000, "batch_size": 128, "precision": "fp32"},
        seed=7,
    )


def test_store_allocates_numbered_experiments(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path)
    first = store.create(_spec())
    second = store.create(_spec())

    assert first.experiment_id == "experiment_001"
    assert second.experiment_id == "experiment_002"
    assert first.gradients_path.name == "gradients.otsg"
    assert first.manifest_path.is_file()


def test_settings_environment_precedes_dotenv(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("OTS_EXPERIMENTS_DIR=/dotenv/path\n", encoding="utf-8")

    settings = Settings.load(
        environ={"OTS_EXPERIMENTS_DIR": "/environment/path"},
        dotenv_path=dotenv,
    )

    assert settings.experiments_dir == Path("/environment/path")
