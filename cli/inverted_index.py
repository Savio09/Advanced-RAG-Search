import os
import pickle
from collections import Counter, defaultdict

from parse_json import load_movies
from keyword_search_cli import tokenize_text


class InvertedIndex:
    def __init__(self) -> None:
        # token -> set of document IDs that contain it
        self.index: dict[str, set[int]] = {}
        # document ID -> full document object
        self.docmap: dict[int, dict] = {}
        # document ID -> Counter of token -> count within that document
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)

    def __add_document(self, doc_id: int, text: str) -> None:
        """Tokenize text and record doc_id under each token in the index."""
        for token in tokenize_text(text):
            self.index.setdefault(token, set()).add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term: str) -> list[int]:
        """Return the document IDs for a preprocessed token, sorted ascending."""
        return sorted(self.index.get(term, set()))

    def get_tf(self, doc_id: int, term: str) -> int:
        """Return how many times a preprocessed token appears in the given document."""
        return self.term_frequencies.get(doc_id, Counter())[term]

    def build(self) -> None:
        """Load movies and index each one by its id."""
        movies = load_movies("data/movies.json")
        for m in movies:
            doc_id = m["id"]
            self.docmap[doc_id] = m
            self.__add_document(doc_id, f"{m['title']} {m['description']}")

    def save(self) -> None:
        """Pickle the index and docmap into the cache/ directory."""
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        """Load the index and docmap from cache/, raising if they don't exist."""
        if not (
            os.path.exists("cache/index.pkl")
            and os.path.exists("cache/docmap.pkl")
            and os.path.exists("cache/term_frequencies.pkl")
        ):
            raise FileNotFoundError("Index not found. Run the 'build' command first.")
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
        with open("cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)
