import argparse
import string
import sys
from parse_json import load_movies
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def normalize(text: str) -> str:
    """Lowercase and strip punctuation for matching."""
    return text.lower().translate(str.maketrans("", "", string.punctuation))

def load_stopwords(file_path: str) -> list[str]:
    """Read stop words from file, preprocessed the same way as queries/titles."""
    with open(file_path, "r", encoding="utf-8") as f:
        words = f.read().splitlines()
    return [normalize(word) for word in words if normalize(word)]

STOPWORDS = set(load_stopwords("data/stopwords.txt"))

def tokenize_text(text: str) -> list[str]:
    """Normalize, split, drop stop words, and stem into a list of tokens."""
    tokens = normalize(text).split()
    return [stemmer.stem(token) for token in tokens if token not in STOPWORDS]

def tokenize_term(term: str) -> str:
    """Tokenize a single term, raising if it doesn't reduce to exactly one token."""
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError(f"Expected a single token, got {len(tokens)}: {tokens}")
    return tokens[0]

def build_command() -> None:
    """Build the inverted index and save it to disk."""
    from inverted_index import InvertedIndex

    index = InvertedIndex()
    index.build()
    index.save()
    print("Index built and saved to cache/")

def tf_command(doc_id: int, term: str) -> None:
    """Print how many times a term appears in the document with the given ID."""
    from inverted_index import InvertedIndex

    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    token = tokenize_term(term)
    print(index.get_tf(doc_id, token))

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index and save it to disk")

    tf_parser = subparsers.add_parser("tf", help="Show term frequency for a term in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to count")

    idf_parser = subparsers.add_parser("idf", help="Show inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to count")

    args = parser.parse_args()

    match args.command:
        case "search":
            from inverted_index import InvertedIndex

            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError as e:
                print(e)
                sys.exit(1)

            query = tokenize_text(args.query)
            results = {}  # doc_id -> movie, dedupes across tokens
            for token in query:
                for doc_id in index.get_documents(token):
                    results.setdefault(doc_id, index.docmap[doc_id])
                    if len(results) >= 5:
                        break
                if len(results) >= 5:
                    break
            print(f"Searching for: {args.query}")
            for i, (doc_id, movie) in enumerate(results.items(), start=1):
                print(f"{i}. {movie['title']} (id: {doc_id})")
        case "build":
            build_command()
        case "tf":
            tf_command(args.doc_id, args.term)
        case "idf":
            idf_command(args.term)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()