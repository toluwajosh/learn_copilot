import json
from dataclasses import asdict, dataclass
from typing import Dict, Optional


@dataclass
class AppParams:
    subjects: dict
    persist: bool
    rerun_indexing: bool
    library_paths: list
    added_paths: list
    models: Dict
    enable_search: bool = False
    config_path: str = "config/params.json"

    @classmethod
    def from_file(cls, path: Optional[str] = None) -> "AppParams":
        if path is None:
            path = cls.config_path
        with open(path) as f:
            params = json.load(f)
        return cls(**params)

    def update(self) -> None:
        with open(self.config_path, "w") as f:
            json.dump(asdict(self), f, indent=4)


PARAMS = AppParams.from_file()
