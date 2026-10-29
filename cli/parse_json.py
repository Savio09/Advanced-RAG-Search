import json

def load_movies(file_path: str):
    """
    load movies from the json file
    """
    movies = []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        movies = data["movies"]
    return movies

