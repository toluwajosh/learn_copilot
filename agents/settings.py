import json
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class AppParams:
    subjects: dict
    persist: bool
    rerun_indexing: bool
    library_paths: list
    added_paths: list
    enable_search: bool = False
    model: str = "gpt-3.5-turbo"
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
