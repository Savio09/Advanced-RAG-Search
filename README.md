# Learning RAG and the Lifecycle of a RAG Pipeline again while building a simple RAG search engine and an app around it.

## Structure

* `cli/`: contains the keyword search cli tool
* `docs/`: contains the documentation

# Lifecycle

1. Preprocessing: In this step, we start with the raw data. Relative to my application, I started witb doing some simple search on the movie database. I simply take the query from the user and then do a walk through all the movies in the database. If any movie contains the search term, I add it to the result. This might be the use case for applications where identifying exact matches are important, but sometimes, we want to simulate user intent. This is where the idea of tokenization comes in. 

 - Tokenization: In this step, we break down the text into smaller units called tokens. 

 - Normalization: In this step, we normalize the text by converting it to lowercase and removing punctuation. 

 - Stop words removal: In this step, we remove stop words from the text. 

 - Stemming: In this step, we stem the text by reducing it to its root form. 

All of this taken together form the core of the preprocessing stage of a RAG pipeline. 

2. Indexing: Indexing is a way of mapping words to documents. In a naive search, we might go through all the movies and check if the search term is in the title. This is inefficient, so we create an index that maps words to the documents that contain them. For example, if we have a movie database with the following movies:

```
1. The Matrix
2. The Godfather
3. The Dark Knight
```

An index would look like this:

```
"The": [1, 2, 3],
"Matrix": [1],
"Godfather": [2],
"Dark": [3],
"Knight": [3]
```

Now, if a user searches for "Matrix", we can quickly find the document that contains the word "Matrix" by looking it up in the index. This is much more efficient than going through all the movies and checking if the search term is in the title. This is the whole idea underpining an inverted index. Simply put, an inverted index is a data structure that maps words to the documents that contain them. There's many types of indices and depending on the application, one might be better than another. A more sophisticated way to do this would be to use TF-IDF (Term Frequency-Inverse Document Frequency), which is a numerical statistic that is used to reflect how important a word is to a document in a collection or corpus. With TF-IDF, we're doing the same thing, but not only do we care if a word is in a document, we are also concerned with the frequency of the word in the document and the rarity of the word across the corpus (where corpus is the whole collection of documents. Here's an analogy to make things more concrete for my potential readers, let's assume we have a novel that's about 50 pages long. Per page, we want to create an index of words that appear on that page. After creating that index, we would then obtain the frequency of the words on the page. We'd do this for all 50 pages, and then we would see how for each word, it happens to be distributed across the 50 pages. If a word is very common in a page, but rare across the entire book, then it's likely to be an important word in that page. Conversely, if a word is very common across all 50 pages, then it's likely to be a stop word and thus not very important for the purposes of retrieval.) A better way would be to use the BM25 algorithm, which is a ranking function used by search engines to estimate the relevance of documents to a given search query. (It's an improvement over TF-IDF that takes into account document length and term frequency normalization.) In simpler terms, it is simply: $${\text{score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\bar{d}}\right)}}$$

The piece I've built so far is the $\text{IDF}(q_i)$ term inside that sum: BM25's **inverse document frequency** for a single query term $q_i$. Its formula is:

$$\text{IDF}(q_i) = \log\!\left(\frac{N - df(q_i) + 0.5}{df(q_i) + 0.5} + 1\right)$$

where:

- **$N$** — the total number of documents in the corpus (in my code, `len(self.docmap)`).
- **$df(q_i)$** — the *document frequency* of $q_i$: the number of documents that contain the term (in my code, `len(self.get_documents(q_i))`). This counts *documents*, not how many times the word occurs.
- **$0.5$** — smoothing constants in the numerator and denominator, so we never divide by zero and the ratio stays well-behaved for both rare and common terms.
- **$+\,1$ inside the log** — keeps the IDF from ever going negative, even for a term that appears in more than half the documents (plain BM25 IDF without it can go negative).

The intuition: a term in *few* documents gets a *high* IDF (it's discriminating), and a term in *most* documents gets a *low* IDF (it's close to a stop word).

### Walking a sample through the pipeline

Let me make this concrete with a tiny corpus of three movies, each a title plus a one-line description (exactly what my `build` step concatenates):

```
1. "The Lion King"   — "A young lion runs from danger."
2. "King Kong"       — "A giant ape runs through the city."
3. "The Dark Knight" — "A masked hero fights crime."
```

Say the user searches for: **`the lion`**.

**Step 1 — Preprocessing (the same path for documents and the query).** For each document I concatenate title + description, then normalize, tokenize, drop stop words, and stem. For document 1, `"The Lion King A young lion runs from danger."` becomes:

```
lowercase + strip punctuation -> "the lion king a young lion runs from danger"
split                         -> [the, lion, king, a, young, lion, runs, from, danger]
drop stop words (the, a, from)-> [lion, king, young, lion, runs, danger]
stem (runs -> run)            -> [lion, king, young, lion, run, danger]
```

Doing the same for all three gives these token lists (note `city` stems to `citi` — a quirk of the Porter stemmer, which trims to a root that isn't always a real word):

```
doc 1 -> [lion, king, young, lion, run, danger]
doc 2 -> [king, kong, giant, ape, run, citi]
doc 3 -> [dark, knight, mask, hero, fight, crime]
```

**Step 2 — Indexing.** From those tokens I build the three structures my `InvertedIndex` holds.

`index` (token -> set of document IDs that contain it):

```
lion -> {1}       king -> {1, 2}    young -> {1}
run  -> {1, 2}    danger -> {1}     kong -> {2}
giant-> {2}       ape -> {2}        citi -> {2}
dark -> {3}       knight -> {3}     mask -> {3}
hero -> {3}       fight -> {3}      crime -> {3}
```

`docmap` (document ID -> the full movie object), so an ID like `1` can be turned back into "The Lion King":

```
1 -> {"id": 1, "title": "The Lion King", ...}
2 -> {"id": 2, "title": "King Kong", ...}
3 -> {"id": 3, "title": "The Dark Knight", ...}
```

`term_frequencies` (document ID -> count of each token *within that document*) — note `lion` appears twice in doc 1:

```
1 -> {lion: 2, king: 1, young: 1, run: 1, danger: 1}
2 -> {king: 1, kong: 1, giant: 1, ape: 1, run: 1, citi: 1}
3 -> {dark: 1, knight: 1, mask: 1, hero: 1, fight: 1, crime: 1}
```

**Step 3 — The query runs through the same preprocessing.** `the lion` -> `[the, lion]` -> drop `the` -> `[lion]` -> stem -> `lion`. So the query reduces to the single token `lion`.

**Step 4 — BM25 IDF of the query term.** Now I plug into the formula. There are $N = 3$ documents, and `lion` appears in only one, so $df = 1$:

$$\text{IDF}(\text{lion}) = \log\!\left(\frac{3 - 1 + 0.5}{1 + 0.5} + 1\right) = \log\!\left(\frac{2.5}{1.5} + 1\right) = \log(2.667) \approx 0.98$$

Compare that with a common term like `king`, which is in two of the three documents ($df = 2$):

$$\text{IDF}(\text{king}) = \log\!\left(\frac{3 - 2 + 0.5}{2 + 0.5} + 1\right) = \log\!\left(\frac{1.5}{2.5} + 1\right) = \log(1.6) \approx 0.47$$

So the rarer term `lion` (0.98) outscores the common term `king` (0.47) — exactly the behaviour we want from IDF.

That gives me the IDF half of each term's score. The other half is the **term-frequency component**, and this is where BM25 does something cleverer than plain TF-IDF: it *saturates* the count and *normalizes* it by document length. I've implemented both, so let me keep walking the same example through them.

**Step 5 — Term frequency saturation.** With plain TF-IDF, a word that appears 10 times counts 10× as much as a word that appears once. That's rarely what I actually want: once a movie's description is clearly *about* lions, the 8th mention of "lion" tells me very little more than the 2nd did. BM25 models this with a *saturating* term frequency:

$$\text{tf}_{\text{BM25}} = \frac{f \cdot (k_1 + 1)}{f + k_1}$$

where $f$ is the raw term frequency (my `get_tf`) and $k_1$ is a tuning knob (I default it to `BM25_K1 = 1.5`). The shape is the whole point — it climbs fast for the first few occurrences, then flattens toward a ceiling of $k_1 + 1$:

| raw $f$ | saturated tf ($k_1 = 1.5$) |
|---------|-----------------------------|
| 1       | 1.00 |
| 2       | 1.43 |
| 5       | 1.92 |
| 10      | 2.17 |
| ∞       | 2.50 (the ceiling, $k_1 + 1$) |

In my toy corpus, `lion` occurs twice in document 1 ($f = 2$), which saturates to:

$$\frac{2 \cdot (1.5 + 1)}{2 + 1.5} = \frac{5}{3.5} \approx 1.43$$

$k_1$ controls *how fast* diminishing returns kick in: a smaller $k_1$ saturates sooner (frequency barely matters past the first hit), a larger $k_1$ keeps rewarding repeats for longer.

**Step 6 — Document length normalization.** There's still a bias to fix. A long document has more room to mention "lion" a few times just by being long, not because it's more *about* lions. BM25 corrects for this by comparing a document's length to the corpus average, through a second knob $b$ (I default it to `BM25_B = 0.75`), folded into the denominator:

$$\text{tf}_{\text{BM25}} = \frac{f \cdot (k_1 + 1)}{f + k_1 \cdot \left(1 - b + b \cdot \dfrac{|D|}{\bar{d}}\right)}$$

where $|D|$ is this document's length in tokens (my `doc_lengths[doc_id]`) and $\bar{d}$ is the average document length across the corpus (my private `__get_avg_doc_length`). The bracketed factor is the normalizer:

- If the document is exactly average length, $|D| / \bar{d} = 1$ and the factor is $1$ — no change.
- A *longer*-than-average document makes the factor $> 1$, inflating the denominator and *dragging the score down*.
- A *shorter*-than-average document makes it $< 1$, *lifting the score up*.
- $b$ sets how strongly this applies: $b = 0$ turns normalization off entirely, $b = 1$ applies it fully.

My toy corpus is a poor place to show this off, because all three documents happen to be 6 tokens long — so $|D| / \bar{d} = 1$ everywhere and the factor does nothing. The effect shows up on the real 5000-movie index. Take the movie *Brave* (id `4651`): it's a short entry at 142 tokens against a corpus average of ~486, so it sits well below average. Watching its `bm25tf` score for the token `merida` (raw $f = 6$) as I turn $b$ up:

| $b$            | length factor | `bm25tf` |
|----------------|---------------|----------|
| 0 (off)        | 1.000         | 2.00 |
| 0.75 (default) | 0.469         | 2.24 |
| 1 (full)       | 0.292         | 2.33 |

The more I let normalization act, the more this *short* document is rewarded for mentioning `merida` six times — which is right, because six hits in 142 tokens is a far stronger signal than six hits in a 1000-token synopsis would be.

**Putting it together.** The full score for a query is then simple: for each query token, multiply its saturated-and-normalized term frequency by its BM25 IDF, and sum those products over the query:

$$\text{score}(D, Q) = \sum_{q_i \in Q} \text{tf}_{\text{BM25}}(q_i, D) \cdot \text{IDF}(q_i)$$

That sum is what my `bm25_search` method computes for every candidate document before sorting by score and returning the top matches — exposed as the `bm25search` CLI command. With that, the keyword-search half of the engine is done; the next stage is semantic search with embeddings.