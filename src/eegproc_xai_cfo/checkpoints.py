"""Load Keras checkpoints across package-name changes."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import zipfile


def _registered_objects(path: Path) -> dict[str, object]:
    """Resolve project objects from the module suffix saved in a Keras archive."""
    with zipfile.ZipFile(path) as archive:
        config = json.loads(archive.read("config.json"))

    custom_objects: dict[str, object] = {}

    def visit(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            module = value.get("module")
            class_name = value.get("class_name")
            registered_name = value.get("registered_name")
            marker = ".deep_learning."
            if (
                isinstance(module, str)
                and marker in module
                and isinstance(class_name, str)
                and class_name.isidentifier()
                and isinstance(registered_name, str)
            ):
                suffix = module.split(marker, 1)[1]
                if suffix and all(part.isidentifier() for part in suffix.split(".")):
                    try:
                        current_module = importlib.import_module(
                            "eegproc_xai_cfo.deep_learning." + suffix
                        )
                        custom_objects[registered_name] = getattr(
                            current_module, class_name
                        )
                    except (ImportError, AttributeError):
                        pass
            for item in value.values():
                visit(item)

    visit(config)
    return custom_objects


def load_model(path: str | Path, **kwargs):
    """Load a model while resolving saved project classes under the current name."""
    import tensorflow as tf

    path = Path(path)
    if path.suffix == ".keras" and path.is_file():
        try:
            objects = _registered_objects(path)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            objects = {}
        objects.update(kwargs.pop("custom_objects", None) or {})
        kwargs["custom_objects"] = objects
    return tf.keras.models.load_model(path, **kwargs)
