import argparse
import math
import string
import sys
from parse_json import load_movies
from constants import BM25_K1, BM25_B
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

def compute_idf(index, token: str) -> float:
    """Inverse document frequency of a preprocessed token across the corpus."""
    n = len(index.docmap)                       # total number of documents
    df = len(index.get_documents(token))        # documents containing the token
    return math.log((n + 1) / (df + 1))

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

def idf_command(term: str) -> None:
    """Print the inverse document frequency of a term across the corpus."""
    from inverted_index import InvertedIndex

    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    token = tokenize_term(term)
    idf = compute_idf(index, token)
    print(f"Inverse document frequency of {token}: {idf:.2f}")

def tf_idf_command(doc_id: int, term: str) -> None:
    """Print the tf-idf score for a term in a document."""
    from inverted_index import InvertedIndex

    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    token = tokenize_term(term)
    tf = index.get_tf(doc_id, token)
    idf = compute_idf(index, token)
    tf_idf = tf * idf
    print(f"TF-IDF score of {token} in document {doc_id}: {tf_idf:.2f}")

def bm25_idf_command(term: str) -> float:
    """Return the BM25 IDF score for a term."""
    from inverted_index import InvertedIndex

    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    token = tokenize_term(term)
    return index.get_bm25_idf(token)

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    """Return the BM25 TF (length-normalized, saturated term frequency) score."""
    from inverted_index import InvertedIndex

    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    token = tokenize_term(term)
    return index.get_bm25_tf(doc_id, token, k1, b)

def bm25_search_command(query: str, limit: int = 5) -> None:
    """Run a full BM25 search and print the ranked results."""
    from inverted_index import InvertedIndex

    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    results = index.bm25_search(query, limit)
    for i, (doc_id, score) in enumerate(results, start=1):
        title = index.docmap[doc_id]["title"]
        print(f"{i}. ({doc_id}) {title} - Score: {score:.2f}")

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

    tf_idf_parser = subparsers.add_parser("tfidf", help="Show tf-idf score for a term in a document")
    tf_idf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_idf_parser.add_argument("term", type=str, help="Term to count")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )

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
        case "tfidf":
            tf_idf_command(args.doc_id, args.term)
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            bm25_search_command(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()