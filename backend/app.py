import os
import sys
from backend.data_loader import load_data
from backend.recommender import HybridRecommender

class MovieApp:
    def __init__(self):
        print("Initializing Recommendation System...")
        self.movies, self.ratings = load_data()
        self.recommender = HybridRecommender(self.movies, self.ratings)
        
        if not self.recommender.load_model():
            print("Model not found. Training new SVD model (this may take a few seconds)...")
            self.recommender.train_svd(full_train=True)
            print("Model trained and saved.")
        else:
            print("Pre-trained model loaded.")

    def get_recommendations(self, user_id, movie_title, top_n=10, cf_weight=0.6, content_weight=0.4):
        # Find movie ID from title
        movie_row = self.movies[self.movies['title'] == movie_title]
        
        if movie_row.empty:
            return None, "Movie not found in database."
        
        target_movie_id = movie_row.iloc[0]['movieId']
        actual_title = movie_row.iloc[0]['title']
        
        recs = self.recommender.get_hybrid_recommendations(
            user_id=user_id,
            target_movie_id=target_movie_id,
            top_n=top_n,
            cf_weight=cf_weight,
            content_weight=content_weight
        )
        
        return recs, actual_title

    def get_all_titles(self):
        """Return all movie titles for dropdown selections."""
        return self.movies['title'].sort_values().tolist()

    def get_random_viewer(self):
        """Get a sample user ID from ratings."""
        return int(self.ratings['userId'].sample(1).iloc[0])

if __name__ == "__main__":
    app = MovieApp()
    recs, title = app.get_recommendations(user_id=1, movie_title="Toy Story")
    if recs is not None:
        print(f"\nTop recommendations for User 1 who liked '{title}':")
        print(recs[['title', 'genres', 'hybrid_score']])
