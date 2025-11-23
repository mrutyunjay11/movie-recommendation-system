import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import plotly.express as px
import plotly.graph_objects as go
import requests
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------------------------
st.set_page_config(
    page_title="CineFlow – Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------------------------
st.markdown("""
<style>
    .stApp {background:linear-gradient(135deg,#0a0a0a 0%,#141414 100%);color:#fff;}
    .main-header {
        font-size:3.5rem;
        background:linear-gradient(90deg,#E50914 0%,#ff6b6b 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        font-weight:900;text-align:center;margin-bottom:0.5rem;
    }
    .sub-header {font-size:1.2rem;color:#b3b3b3;text-align:center;margin-bottom:2rem;letter-spacing:2px;}
    .movie-card,.metric-card {
        background:linear-gradient(135deg,rgba(42,42,42,.9)0%,rgba(26,26,26,.9)100%);
        border-radius:12px;padding:1.5rem;margin:1rem 0;border-left:4px solid #E50914;
        box-shadow:0 8px 25px rgba(0,0,0,.5);
    }
    .movie-card:hover{transform:translateY(-5px);box-shadow:0 15px 35px rgba(229,9,20,.4);}
    .stButton button {
        background:linear-gradient(90deg,#E50914 0%,#B81D24 100%);
        color:white;border:none;padding:.8rem 2rem;border-radius:8px;font-weight:600;
    }
    .stButton button:hover{transform:scale(1.05);box-shadow:0 8px 25px rgba(229,9,20,.5);}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------------
@st.cache_data
def load_movielens_data():
    """Load MovieLens dataset or safe fallback."""
    if os.path.exists("ml-latest-small/movies.csv") and os.path.exists("ml-latest-small/ratings.csv"):
        movies_df = pd.read_csv("ml-latest-small/movies.csv")
        ratings_df = pd.read_csv("ml-latest-small/ratings.csv")
        movies_df["genres"] = movies_df["genres"].fillna("Unknown").replace("", "Unknown")
    else:
        st.warning("MovieLens dataset not found — using sample data.")
        movies_df = pd.DataFrame({
            "movieId": [1,2,3],
            "title": ["Toy Story (1995)", "Jumanji (1995)", "Heat (1995)"],
            "genres": [
                "Adventure|Animation|Children|Comedy|Fantasy",
                "Adventure|Children|Fantasy",
                "Action|Crime|Thriller"
            ]
        })
        ratings_df = pd.DataFrame({
            "userId":[1,2,3],
            "movieId":[1,2,3],
            "rating":[4.0,3.5,5.0]
        })
    movies_df["genres"] = movies_df["genres"].astype(str)
    return movies_df, ratings_df

@st.cache_data
def preprocess_data(movies_df, ratings_df):
    """Feature engineering & stats."""
    movies_df["year"] = movies_df["title"].str.extract(r"\((\d{4})\)")
    movies_df["clean_title"] = movies_df["title"].str.replace(r"\s*\(\d{4}\)","",regex=True)
    stats = ratings_df.groupby("movieId")["rating"].agg(["mean","count"]).reset_index()
    stats.columns = ["movieId","avg_rating","rating_count"]
    merged = movies_df.merge(stats,on="movieId",how="left").fillna({"avg_rating":0,"rating_count":0})
    return merged, ratings_df

# --------------------------------------------------------------------
# RECOMMENDATION MODELS
# --------------------------------------------------------------------
@st.cache_resource
def build_content_based_engine(movies_df):
    """Build TF-IDF model safely (immune to empty genres)."""

    # ------------------------------------------------------------------
    # Guarantee the column exists
    # ------------------------------------------------------------------
    if "genres" not in movies_df.columns:
        st.warning("Genres column missing — creating placeholder data.")
        movies_df["genres"] = "Unknown"

    # ------------------------------------------------------------------
    # Clean invalid entries
    # ------------------------------------------------------------------
    movies_df["genres"] = (
        movies_df["genres"]
        .astype(str)
        .fillna("Unknown")
        .replace("", "Unknown")
        .replace("nan", "Unknown")
    )

    # ------------------------------------------------------------------
    # Verify content (in case all rows are blank)
    # ------------------------------------------------------------------
    valid_docs = movies_df["genres"][movies_df["genres"].str.strip() != ""]
    if valid_docs.empty:
        st.warning("No valid genre text found — using fallback sample genres.")
        movies_df["genres"] = ["Action|Drama|Adventure", "Comedy|Romance", "Thriller|Crime", "Animation|Family"] * (
            len(movies_df) // 4 + 1
        )
        movies_df["genres"] = movies_df["genres"].head(len(movies_df))

    # ------------------------------------------------------------------
    # Build the TF-IDF model safely
    # ------------------------------------------------------------------
    tfidf = TfidfVectorizer(token_pattern=r"\b\w+\b", min_df=1)

    try:
        tfidf_matrix = tfidf.fit_transform(movies_df["genres"])
    except ValueError:
        # if this still fails, reset Streamlit cache and use fallback data
        st.cache_data.clear()
        st.cache_resource.clear()
        st.error("TF-IDF failed due to cached empty data. Cache cleared — please rerun.")
        st.stop()

    # ------------------------------------------------------------------
    # Compute cosine similarity
    # ------------------------------------------------------------------
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim, tfidf
@st.cache_resource
def build_collaborative_engine(ratings_df, movies_df, n_factors=20):
    """Collaborative filtering with SVD."""
    user_movie = ratings_df.pivot_table(index="userId",columns="movieId",values="rating").fillna(0)
    if user_movie.shape[0]<2 or user_movie.shape[1]<2:
        return np.zeros((len(movies_df),len(movies_df))), user_movie
    svd = TruncatedSVD(n_components=min(n_factors,min(user_movie.shape)-1),random_state=42)
    reduced = svd.fit_transform(csr_matrix(user_movie.values))
    similarity = cosine_similarity(reduced.T)
    return similarity, user_movie

def get_content_recommendations(movie_title, movies_df, cos, n=5):
    try:
        idx = movies_df.index[movies_df["title"]==movie_title][0]
        sims = sorted(list(enumerate(cos[idx])),key=lambda x:x[1],reverse=True)[1:n+1]
        recs = movies_df.iloc[[i[0] for i in sims]].copy()
        recs["similarity_score"] = [s[1] for s in sims]
        recs["method"]="Content-Based"
        return recs
    except Exception:
        return pd.DataFrame()

def get_collab_recommendations(movie_title,movies_df,sim,matrix,n=5):
    try:
        mid = movies_df.loc[movies_df["title"]==movie_title,"movieId"].values[0]
        if mid not in matrix.columns: return pd.DataFrame()
        idx = list(matrix.columns).index(mid)
        sims = sorted(list(enumerate(sim[idx])),key=lambda x:x[1],reverse=True)[1:n+1]
        mids = [matrix.columns[i[0]] for i in sims]
        recs = movies_df[movies_df["movieId"].isin(mids)].copy()
        recs["similarity_score"]=[s[1] for s in sims]
        recs["method"]="Collaborative"
        return recs
    except Exception:
        return pd.DataFrame()

def get_hybrid_recommendations(title,movies_df,cos,sim,matrix,n=5,genre_filter=None):
    """Hybrid model + optional genre filter."""
    c = get_content_recommendations(title,movies_df,cos,n*2)
    f = get_collab_recommendations(title,movies_df,sim,matrix,n*2)
    if c.empty and f.empty: return pd.DataFrame()
    comb = pd.concat([c,f])
    hybrid = comb.groupby("movieId").agg({
        "similarity_score":"mean","title":"first","genres":"first",
        "avg_rating":"first","rating_count":"first"
    }).reset_index().sort_values("similarity_score",ascending=False)

    if genre_filter:
        hybrid = hybrid[hybrid["genres"].apply(lambda x: any(g in x for g in genre_filter))]

    hybrid = hybrid.head(n)
    hybrid["method"]="Hybrid"
    return hybrid

# --------------------------------------------------------------------
# VISUALIZATION HELPERS
# --------------------------------------------------------------------
def rec_chart(df):
    fig=go.Figure([go.Bar(
        x=df["similarity_score"]*100,y=df["title"],orientation="h",
        marker=dict(color=df["similarity_score"]*100,colorscale=[[0,"#B81D24"],[1,"#E50914"]]),
        text=[f"{v*100:.1f}%" for v in df["similarity_score"]],textposition="auto")])
    fig.update_layout(title="Recommendation Scores",xaxis_title="Similarity (%)",
        plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",font=dict(color="white"))
    return fig

def genre_distribution(df):
    allg=[g for sub in df["genres"] for g in sub.split("|")]
    cnt=pd.Series(allg).value_counts().head(10)
    fig=px.bar(x=cnt.values,y=cnt.index,orientation="h",title="Top Genres",
        labels={"x":"Count","y":"Genre"},
        color=cnt.values,color_continuous_scale=[[0,"#B81D24"],[1,"#E50914"]])
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",font_color="white")
    return fig

def rating_distribution(df):
    fig=px.histogram(df,x="avg_rating",nbins=20,title="Average Rating Distribution",
        color_discrete_sequence=["#E50914"])
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",font_color="white")
    return fig

# --------------------------------------------------------------------
# TMDB POSTER
# --------------------------------------------------------------------
def get_poster(title):
    api_key = st.secrets.get("TMDB_API_KEY") or os.getenv("TMDB_API_KEY")
    if not api_key: return None
    try:
        q=title.split("(")[0]
        r=requests.get("https://api.themoviedb.org/3/search/movie",
                       params={"api_key":api_key,"query":q})
        js=r.json()
        if js.get("results"):
            path=js["results"][0].get("poster_path")
            if path: return f"https://image.tmdb.org/t/p/w500{path}"
    except Exception: return None
    return None

# --------------------------------------------------------------------
# MAIN APP
# --------------------------------------------------------------------
def main():
    st.markdown('<h1 class="main-header">CINEFLOW</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Movie Recommendation System</p>', unsafe_allow_html=True)

    with st.sidebar:
        search_mode = st.radio("Search Mode", ["Movie-based", "Category-based"])
        method = st.selectbox("Recommendation Method", ["Hybrid", "Content-Based", "Collaborative"])
        n = st.slider("Number of recommendations", 3, 10, 5)
        st.info("Dataset: MovieLens 100K (or sample).")

    movies,ratings=load_movielens_data()
    movies_enriched,ratings_clean=preprocess_data(movies,ratings)
    cos,tfidf=build_content_based_engine(movies_enriched)
    sim,user_matrix=build_collaborative_engine(ratings_clean,movies_enriched)

    tab1,tab2=st.tabs(["Recommendations","Analytics"])

    with tab1:
        if search_mode == "Movie-based":
            movie=st.selectbox("Select a movie",movies_enriched["title"].values)
            if st.button("Get Recommendations",use_container_width=True):
                if method=="Content-Based":
                    recs=get_content_recommendations(movie,movies_enriched,cos,n)
                elif method=="Collaborative":
                    recs=get_collab_recommendations(movie,movies_enriched,sim,user_matrix,n)
                else:
                    recs=get_hybrid_recommendations(movie,movies_enriched,cos,sim,user_matrix,n)
                if recs.empty:
                    st.warning("No recommendations found.")
                else:
                    st.success(f"Found {len(recs)} recommendations!")
                    for i,(_,row) in enumerate(recs.iterrows(),1):
                        poster=get_poster(row["title"])
                        col1,col2=st.columns([1,4])
                        with col1:
                            if poster: st.image(poster,width=150)
                        with col2:
                            st.markdown(f"""
                            <div class="movie-card">
                            <h4>#{i} – {row['title']}</h4>
                            <p>{row['genres'].replace('|',', ')}</p>
                            <p>⭐ {row['avg_rating']:.1f}/5.0 ({int(row['rating_count'])} ratings)</p>
                            </div>
                            """,unsafe_allow_html=True)
                    st.plotly_chart(rec_chart(recs),use_container_width=True)
        else:
            all_genres = sorted(
                list(set(g for sub in movies_enriched["genres"] for g in sub.split("|")))
            )
            selected_genres = st.multiselect("Select categories", all_genres)
            if st.button("Find Movies",use_container_width=True):
                if not selected_genres:
                    st.warning("Please select at least one category.")
                else:
                    filtered = movies_enriched[
                        movies_enriched["genres"].apply(
                            lambda x: any(g in x for g in selected_genres)
                        )
                    ]
                    if filtered.empty:
                        st.warning("No movies found in those categories.")
                    else:
                        st.success(f"Top {n} recommendations within {', '.join(selected_genres)}:")
                        # Pick one top-rated movie to anchor recommendations
                        base_movie = filtered.sort_values("avg_rating",ascending=False).iloc[0]["title"]
                        recs = get_hybrid_recommendations(base_movie,movies_enriched,cos,sim,user_matrix,n,genre_filter=selected_genres)
                        if recs.empty: recs = filtered.sort_values("avg_rating",ascending=False).head(n)
                        for i,(_,row) in enumerate(recs.iterrows(),1):
                            poster=get_poster(row["title"])
                            col1,col2=st.columns([1,4])
                            with col1:
                                if poster: st.image(poster,width=150)
                            with col2:
                                st.markdown(f"""
                                <div class="movie-card">
                                <h4>#{i} – {row['title']}</h4>
                                <p>{row['genres'].replace('|',', ')}</p>
                                <p>⭐ {row['avg_rating']:.1f}/5.0 ({int(row['rating_count'])} ratings)</p>
                                </div>
                                """,unsafe_allow_html=True)

    with tab2:
        st.markdown("### Dataset Analytics")
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Movies",len(movies_enriched))
        c2.metric("Ratings",len(ratings_clean))
        c3.metric("Users",ratings_clean["userId"].nunique())
        c4.metric("Avg Rating",f"{ratings_clean['rating'].mean():.2f}")
        st.plotly_chart(genre_distribution(movies_enriched),use_container_width=True)
        st.plotly_chart(rating_distribution(movies_enriched),use_container_width=True)

if __name__=="__main__":
    main()