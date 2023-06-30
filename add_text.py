"""
Script for adding text to the collection
"""
from pathlib import Path
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

from settings import PARAMS


COLLECTION_PATH = PARAMS["collection_path"]
LIBRARY_PATHS = PARAMS["library_paths"]

doc_loaders = {
    "pdf": PyPDFLoader,
    "txt": TextLoader,
    "docx": Docx2txtLoader,
}


for path in LIBRARY_PATHS:
    if path in PARAMS["added_paths"]:
        print(f"{path} already in library. Skipping...")
        continue
    loader = doc_loaders[Path(path).suffix.split(".")[-1]](path)
    pages = loader.load_and_split()
    with open(COLLECTION_PATH, "a") as f:
        print(f"writing pages.... {path}")
        for page in pages:
            f.write(page.page_content)
        PARAMS["added_paths"].append(path)
