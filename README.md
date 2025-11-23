# 🎬 CineFlow – Movie Recommendation System  
An AI-powered movie recommender built using Machine Learning, Python, and Streamlit.

---

## Project Overview
CineFlow is a hybrid movie recommendation system that suggests movies based on user preferences.  
It integrates content-based filtering, collaborative filtering, and a hybrid engine to provide highly relevant movie recommendations.  
The system includes a modern Streamlit UI with analytics and visual visualizations.

---

## Features

### Recommendation Modes
- **Content-Based Filtering** using TF-IDF + Cosine Similarity  
- **Collaborative Filtering** using SVD-based latent factors  
- **Hybrid Recommendation Engine** combining both  
- **Category-Based Search** using selected genres  

### Streamlit UI
- Netflix-style dark UI  
- Movie posters from TMDB API  
- Interactive charts (Plotly)  
- Rating and genre analytics  

### Data Processing
- MovieLens Dataset  
- Genre normalization & cleaning  
- Rating statistics (mean, count)  
- Title cleaning and metadata extraction  

---

## Dataset
Uses the **MovieLens 100K** dataset.  
If not available, a safe fallback dataset is automatically loaded.

---

## Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-Learn (TF-IDF, SVD, Cosine Similarity)  
- Streamlit  
- Plotly  
- MovieLens Dataset  
- TMDB API (optional)  

---

## How It Works

### 1. Load & Preprocess Data
- Load movies and ratings  
- Extract release year  
- Clean genres  
- Compute rating statistics  

### 2. Build Recommender Models
#### ✔ Content-Based  
TF-IDF Vectorizer → Cosine Similarity

#### ✔ Collaborative Filtering  
User–Movie rating matrix → SVD → latent factors → similarity matrix

#### ✔ Hybrid Engine  
Average of both similarity scores for improved accuracy

---

## ▶ Running the Application

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Streamlit App
```bash
streamlit run app.py
```

### 3. Open in Browser
```
http://localhost:8501
```

---

## Screenshots (Add yours)
- Recommendation UI  
- Analytics Dashboard  
- Category-Based Movie Search  
- Poster Display View  

---

## Future Enhancements
- Sentiment analysis using IMDB/RottenTomatoes reviews  
- Deep learning–based collaborative filtering  
- Real-time personalized watchlist  
- Multi-language support  

---

## Author
**Mrutyunjay Joshi**

---

## License
For academic and educational use only.

