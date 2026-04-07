import streamlit as st
import pandas as pd
import sys
import os

# Add the root directory to sys.path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app import MovieApp

# --- Page Config ---
st.set_page_config(
    page_title="Hybrid Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# --- App Initialization ---
@st.cache_resource
def init_app():
    return MovieApp()

app = init_app()

# --- Sidebar ---
st.sidebar.title("Configuration")
st.sidebar.info("Adjust the hybrid logic weights here.")

cf_weight = st.sidebar.slider("Collaborative Weight (SVD)", 0.0, 1.0, 0.6)
content_weight = 1.0 - cf_weight
st.sidebar.write(f"Content-Based Weight: **{content_weight:.1f}**")

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Stats")
st.sidebar.write(f"Movies: {len(app.movies)}")
st.sidebar.write(f"Ratings: {len(app.ratings)}")

# --- Main Page ---
st.title("🎬 Hybrid Movie Recommendation System")
st.markdown("""
This system combines **Collaborative Filtering (SVD)** and **Content-Based Filtering (TF-IDF)** 
to provide personalized movie recommendations. 
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Identify User")
    user_id = st.number_input("Enter User ID (Example: 1 for existing, 9999 for new)", min_value=1, value=1)
    
    # Check if user is known
    is_known = user_id in app.ratings['userId'].values
    if is_known:
        st.success(f"User {user_id} found in history! System will use Collaborative Filtering.")
    else:
        st.warning(f"New User {user_id}. System will fallback to Content-Based Filtering only.")

with col2:
    st.subheader("2. Select Movie")
    movie_titles = app.get_all_titles()
    selected_movie = st.selectbox("Pick a movie you liked:", movie_titles, index=0)

st.markdown("---")

# --- Recommendation Logic ---
if st.button("Generate Recommendations", type="primary"):
    with st.spinner("Analyzing patterns and calculating scores..."):
        recs, actual_title = app.get_recommendations(
            user_id=user_id,
            movie_title=selected_movie,
            top_n=10,
            cf_weight=cf_weight,
            content_weight=content_weight
        )
        
    if recs is not None:
        st.subheader(f"Recommendations for user profile based on '{actual_title}'")
        
        # Format the dataframe for display
        display_df = recs[['title', 'genres', 'hybrid_score', 'content_score', 'cf_score']].copy()
        
        # Color coding the scores
        def color_scores(val):
            color = 'blue' if val > 0.7 else 'black'
            return f'color: {color}'

        st.dataframe(
            display_df.style.highlight_max(axis=0, subset=['hybrid_score']),
            use_container_width=True
        )
        
        # Displaying posters/cards (Simplified)
        st.info("💡 **Logic:** Collaborative (SVD) predicts what you might like based on other users, while Content-Based finds movies with similar genres.")
    else:
        st.error("Could not generate recommendations. Please check inputs.")

# --- Footer ---
st.markdown("---")
st.markdown("Created with ❤️ by Antigravity AI | Dataset: MovieLens ml-latest-small")
