import math
import os
import pickle
from collections import Counter, defaultdict

from parse_json import load_movies
from keyword_search_cli import tokenize_text
from constants import BM25_K1, BM25_B, CACHE_DIR


class InvertedIndex:
    def __init__(self) -> None:
        # token -> set of document IDs that contain it
        self.index: dict[str, set[int]] = {}
        # document ID -> full document object
        self.docmap: dict[int, dict] = {}
        # document ID -> Counter of token -> count within that document
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)
        # document ID -> total number of tokens in that document
        self.doc_lengths: dict[int, int] = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        """Tokenize text and record doc_id under each token in the index."""
        tokens = tokenize_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term: str) -> list[int]:
        """Return the document IDs for a preprocessed token, sorted ascending."""
        return sorted(self.index.get(term, set()))

    def get_tf(self, doc_id: int, term: str) -> int:
        """Return how many times a preprocessed token appears in the given document."""
        return self.term_frequencies.get(doc_id, Counter())[term]

    def get_bm25_idf(self, term: str) -> float:
        """BM25 inverse document frequency of a preprocessed token. Assumes a single token."""
        n = len(self.docmap)                     # total number of documents
        df = len(self.get_documents(term))       # documents containing the token
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def __get_avg_doc_length(self) -> float:
        """Average number of tokens per document (0.0 when there are no documents)."""
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        """BM25 saturated term frequency, normalized by document length."""
        tf = self.get_tf(doc_id, term)
        doc_len = self.doc_lengths.get(doc_id, 0)
        avg_len = self.__get_avg_doc_length()
        # a longer-than-average document is penalized; shorter is rewarded
        length_norm = 1 - b + b * (doc_len / avg_len) if avg_len else 1 - b
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        """Full BM25 score of a single term in one document: bm25_tf * bm25_idf."""
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int = 5) -> list[tuple[int, float]]:
        """Rank documents by their summed BM25 score over the query's tokens."""
        tokens = tokenize_text(query)
        scores: dict[int, float] = {}
        for token in tokens:
            for doc_id in self.get_documents(token):
                scores[doc_id] = scores.get(doc_id, 0.0) + self.bm25(doc_id, token)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:limit]

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
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        """Load the index and docmap from cache/, raising if they don't exist."""
        if not (
            os.path.exists("cache/index.pkl")
            and os.path.exists("cache/docmap.pkl")
            and os.path.exists("cache/term_frequencies.pkl")
            and os.path.exists(self.doc_lengths_path)
        ):
            raise FileNotFoundError("Index not found. Run the 'build' command first.")
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
        with open("cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)
