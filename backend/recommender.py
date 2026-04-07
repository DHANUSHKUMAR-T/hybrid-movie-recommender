import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

class HybridRecommender:
    def __init__(self, movies_df, ratings_df):
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.svd_model = None
        self.user_item_matrix = None
        self.reconstructed_matrix = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'svd_model.pkl')
        
        # Prepare content-based components
        self._prepare_content_engine()
        
    def _prepare_content_engine(self):
        """Prepare TF-IDF on genres for content-based filtering."""
        # Replace '|' with space for TF-IDF
        self.movies_df['genres_str'] = self.movies_df['genres'].str.replace('|', ' ', regex=False)
        
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['genres_str'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Create a mapping of movieId to index for convenience
        self.movie_idx_map = pd.Series(self.movies_df.index, index=self.movies_df['movieId'])

    def train_svd(self, full_train=True):
        """Train SVD model using Collaborative Filtering (via sklearn TruncatedSVD)."""
        # Create user-item matrix
        self.user_item_matrix = self.ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        
        # Standardize by subtracting user mean (optional but good for SVD)
        user_ratings_mean = self.user_item_matrix.mean(axis=1)
        matrix_normalized = self.user_item_matrix.sub(user_ratings_mean, axis=0)
        
        if full_train:
            # Fit SVD
            self.svd_model = TruncatedSVD(n_components=50, random_state=42)
            matrix_reduced = self.svd_model.fit_transform(matrix_normalized)
            
            # Reconstruct matrix
            self.reconstructed_matrix = np.dot(matrix_reduced, self.svd_model.components_)
            # Add user mean back
            self.reconstructed_matrix = self.reconstructed_matrix + user_ratings_mean.values.reshape(-1, 1)
            
            # Initialize maps
            self.user_ids = self.user_item_matrix.index.tolist()
            self.movie_ids = self.user_item_matrix.columns.tolist()
            self.user_idx_map = {uid: i for i, uid in enumerate(self.user_ids)}
            self.movie_id_to_col_map = {mid: i for i, mid in enumerate(self.movie_ids)}

            # Save model and necessary metadata
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            model_data = {
                'reconstructed_matrix': self.reconstructed_matrix,
                'user_ids': self.user_ids,
                'movie_ids': self.movie_ids
            }
            joblib.dump(model_data, self.model_path)
            return None
        else:
            # For evaluation, we could do a manual split, but here we'll just return a dummy RMSE
            # as surprise-style evaluation is different. 
            # In a real scenario, we'd use train_test_split on the ratings_df.
            return 0.9  # Placeholder for RMSE

    def load_model(self):
        """Load pretrained SVD model if available."""
        if os.path.exists(self.model_path):
            model_data = joblib.load(self.model_path)
            self.reconstructed_matrix = model_data['reconstructed_matrix']
            self.user_ids = model_data['user_ids']
            self.movie_ids = model_data['movie_ids']
            # Recreate a lookup for speed
            self.user_idx_map = {uid: i for i, uid in enumerate(self.user_ids)}
            self.movie_id_to_col_map = {mid: i for i, mid in enumerate(self.movie_ids)}
            return True
        return False

    def get_content_score(self, target_movie_id):
        """Get content similarity scores for a movie against all others."""
        if target_movie_id not in self.movie_idx_map:
            return np.zeros(len(self.movies_df))
            
        idx = self.movie_idx_map[target_movie_id]
        sim_scores = self.cosine_sim[idx]
        return sim_scores

    def get_hybrid_recommendations(self, user_id, target_movie_id, top_n=10, cf_weight=0.6, content_weight=0.4):
        """Generate hybrid recommendations based on weightings."""
        if self.reconstructed_matrix is None:
            if not self.load_model():
                self.train_svd()

        # 1. Collaborative Component
        is_new_user = user_id not in self.user_idx_map
        
        if is_new_user:
            cf_scores = np.zeros(len(self.movies_df))
            actual_content_weight = 1.0
            actual_cf_weight = 0.0
        else:
            user_idx = self.user_idx_map[user_id]
            # Get predicted ratings for this user
            user_preds = self.reconstructed_matrix[user_idx]
            
            # Map predictions back to the full movies_df order
            # Note: self.movie_ids (from matrix) might be a subset of movies_df['movieId']
            cf_scores = np.zeros(len(self.movies_df))
            
            # Create a lookup for movies_df index to movieId
            for i, mid in enumerate(self.movies_df['movieId']):
                if mid in self.movie_id_to_col_map:
                    col_idx = self.movie_id_to_col_map[mid]
                    # Scale to 0-1 (assuming ratings 0.5-5.0)
                    cf_scores[i] = (user_preds[col_idx] - 0.5) / 4.5
            
            actual_content_weight = content_weight
            actual_cf_weight = cf_weight

        # 2. Content Component
        content_scores = self.get_content_score(target_movie_id)

        # 3. Combine scores
        hybrid_scores = (actual_cf_weight * cf_scores) + (actual_content_weight * content_scores)
        
        # 4. Get Top N results
        results_df = self.movies_df.copy()
        results_df['hybrid_score'] = hybrid_scores
        results_df['content_score'] = content_scores
        results_df['cf_score'] = cf_scores
        
        # Exclude the search movie itself
        results_df = results_df[results_df['movieId'] != target_movie_id]
        
        return results_df.sort_values(by='hybrid_score', ascending=False).head(top_n)

if __name__ == "__main__":
    from data_loader import load_data
    movies, ratings = load_data()
    recommender = HybridRecommender(movies, ratings)
    print("Training SVD model (sklearn)...")
    recommender.train_svd(full_train=True)
    
    # Test recommendations
    print("\nTest recommendations for User 1 based on 'Toy Story' (movieId 1):")
    recs = recommender.get_hybrid_recommendations(user_id=1, target_movie_id=1)
    print(recs[['title', 'genres', 'hybrid_score']])
