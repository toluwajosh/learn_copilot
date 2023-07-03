from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
import argparse
from dotenv import load_dotenv

load_dotenv()


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
        "--persist-path",
        "-l",
        type=str,
        required=True,
        help="List of libraries to add",
    )

    return parser.parse_args()


cli_args = parse_arguments()

loader = TextLoader(cli_args.collection_path)
# text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
# texts = text_splitter.split_documents(loader)
print("Creating index...\n")
index = VectorstoreIndexCreator(
    vectorstore_kwargs={"persist_directory": cli_args.persist_path}
).from_loaders([loader])
# ).from_documents(texts)
