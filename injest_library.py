import argparse
from agents.data import injest_data


def parse_arguments():
    parser = argparse.ArgumentParser(description="Add text to a collection.")
    parser.add_argument(
        "--library-directory",
        "-l",
        type=str,
        required=True,
        help="List of libraries to add",
    )
    parser.add_argument(
        "--persist-directory",
        "-p",
        type=str,
        required=True,
        help="Path to the collection to add.",
    )

    return parser.parse_args()


cli_args = parse_arguments()

injest_data(
    cli_args.library_directory,
    cli_args.persist_directory,
)
