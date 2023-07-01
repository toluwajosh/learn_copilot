"""
Script for adding text to the collection
"""
import argparse
from pathlib import Path
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

from agents.settings import PARAMS


def parse_arguments():
    parser = argparse.ArgumentParser(description="Add text to a collection.")
    parser.add_argument(
        "--collection-path",
        "-p",
        type=str,
        required=True,
        help="Path to the collection to add.",
    )

    parser.add_argument(
        "--library-paths",
        "-l",
        nargs="+",
        default=[],
        help="List of libraries to add",
    )

    return parser.parse_args()


cli_args = parse_arguments()

COLLECTION_PATH = cli_args.collection_path
LIBRARY_PATHS = (
    PARAMS["library_paths"]
    if not cli_args.library_paths
    else cli_args.library_paths
)

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

