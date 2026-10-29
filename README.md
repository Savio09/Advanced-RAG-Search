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

2. Indexing: 