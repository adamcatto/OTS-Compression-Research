"""Load and validate an experiment specification from a JSON recipe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match
from referencing import Registry, Resource

from ..core.experiments import ExperimentSpec


META_CONFIG_NAME = "experiment.schema.json"


def _read_json_object(path: Path, context: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {context} JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context} must contain a JSON object: {path}")
    return value


def _discover_meta_config(recipe_path: Path) -> Optional[Path]:
    for directory in (recipe_path.parent, *recipe_path.parents):
        candidate = directory / META_CONFIG_NAME
        if candidate.is_file():
            return candidate
    return None


def validate_experiment_config(
    value: Mapping[str, Any], meta_config_path: Path
) -> None:
    """Validate one concrete recipe against a split JSON meta-config."""

    meta_config_path = Path(meta_config_path).resolve()
    schema = dict(_read_json_object(meta_config_path, "meta-config"))
    schema["$id"] = meta_config_path.as_uri()

    registry = Registry()
    schema_paths = [meta_config_path]
    schema_directory = meta_config_path.parent / "schema"
    if schema_directory.is_dir():
        schema_paths.extend(sorted(schema_directory.rglob("*.schema.json")))
    for schema_path in schema_paths:
        contents = _read_json_object(schema_path, "schema")
        registry = registry.with_resource(
            schema_path.resolve().as_uri(), Resource.from_contents(contents)
        )

    Draft202012Validator.check_schema(schema)
    error = best_match(
        Draft202012Validator(schema, registry=registry).iter_errors(value)
    )
    if error is None:
        return
    location = ".".join(str(part) for part in error.absolute_path)
    prefix = f" at {location}" if location else ""
    raise ValueError(f"experiment config{prefix}: {error.message}")


def load_experiment_spec(
    path: Path, *, meta_config_path: Optional[Path] = None
) -> ExperimentSpec:
    path = Path(path)
    value = _read_json_object(path, "experiment config")
    selected_meta_config = (
        Path(meta_config_path)
        if meta_config_path is not None
        else _discover_meta_config(path)
    )
    if selected_meta_config is not None:
        validate_experiment_config(value, selected_meta_config)
    try:
        return ExperimentSpec(**value)
    except TypeError as exc:
        raise ValueError(f"invalid experiment fields in {path}: {exc}") from exc
