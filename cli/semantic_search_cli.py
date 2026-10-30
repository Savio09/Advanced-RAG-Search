import argparse

from lib.semantic_search import verify_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify", help="Verify the model loads appropriately")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
