import os
import requests
import zipfile
import io
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
DATA_URL = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

def download_dataset():
    """Download and extract MovieLens small dataset."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    # Check if files already exist
    movies_path = os.path.join(DATA_DIR, 'movies.csv')
    ratings_path = os.path.join(DATA_DIR, 'ratings.csv')
    
    if os.path.exists(movies_path) and os.path.exists(ratings_path):
        print("Dataset already exists in data/ folder.")
        return
        
    print(f"Downloading dataset from {DATA_URL}...")
    response = requests.get(DATA_URL)
    if response.status_code == 200:
        z = zipfile.ZipFile(io.BytesIO(response.content))
        # ml-latest-small.zip contains a folder 'ml-latest-small/'
        # We want to extract its contents directly into DATA_DIR
        for member in z.namelist():
            filename = os.path.basename(member)
            if not filename:
                continue
                
            source = z.open(member)
            target_path = os.path.join(DATA_DIR, filename)
            with open(target_path, "wb") as f:
                f.write(source.read())
        print("Dataset downloaded and extracted successfully.")
    else:
        raise Exception(f"Failed to download dataset. Status code: {response.status_code}")

def load_data():
    """Load movies and ratings dataframes."""
    download_dataset()
    movies = pd.read_csv(os.path.join(DATA_DIR, 'movies.csv'))
    ratings = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'))
    
    # Simple cleaning: remove duplicates if any
    movies.drop_duplicates(subset='movieId', inplace=True)
    movies.reset_index(drop=True, inplace=True)
    ratings.drop_duplicates(inplace=True)
    ratings.reset_index(drop=True, inplace=True)
    
    return movies, ratings

if __name__ == "__main__":
    m, r = load_data()
    print(f"Loaded {len(m)} movies and {len(r)} ratings.")
