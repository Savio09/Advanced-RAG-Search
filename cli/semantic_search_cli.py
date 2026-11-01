import argparse

from lib.semantic_search import verify_model, embed_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify", help="Verify the model loads appropriately")

    embed_text_parser = subparsers.add_parser("embed_text", help="Create a text embedding")
    embed_text_parser.add_argument("text", type=str, help="Embedding text")
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
