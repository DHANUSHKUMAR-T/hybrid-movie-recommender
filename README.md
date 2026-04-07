# Hybrid Movie Recommendation System 🎬

A production-ready recommendation system that combines **Collaborative Filtering (SVD)** and **Content-Based Filtering (TF-IDF)** using the MovieLens dataset.

## ✨ Features
- **Hybrid Scoring**: `(0.6 * Collaborative) + (0.4 * Content)` (Adjustable).
- **Cold Start Handling**: Automatically switches to Content-Based filtering for new users.
- **SVD Model**: Trained using the `scikit-surprise` library for accurate rating predictions.
- **TF-IDF Vectorizer**: Used for content similarity based on movie genres.
- **Streamlit UI**: Clean, interactive dashboard for exploring recommendations.

## 📁 Project Structure
- `backend/`: Core logic and data handlers.
- `data/`: Extracted MovieLens CSVs (Auto-downloaded).
- `models/`: Persistent SVD model pickle.
- `frontend/`: Streamlit application.

## 🚀 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Recommendation App**:
   Run the following from the root directory:
   ```bash
   streamlit run frontend/streamlit_app.py
   ```

3. **Explore**:
   - Enter a **User ID** (e.g., `1` for historical data or `9999` for a "New User").
   - Select a **Movie** you like from the dropdown.
   - Adjust the **Weight Sliders** to see how Recommendations change!

## 🧪 Evaluation
The Collaborative Filtering (SVD) model is evaluated using **RMSE** (Root Mean Square Error). On the `ml-latest-small` dataset, it typically achieves ~0.87-0.88 RMSE.

---
Created by Antigravity AI
