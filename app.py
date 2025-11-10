import streamlit as st
import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(layout="wide")
st.title("🎬 Movie Recommendation and Filtering Engine")

# --- DATA FILE PATHS ---
# Make sure these files are in the same folder as this app.py script
MOVIES_FILE = 'tmdb_5000_movies.csv'
CREDITS_FILE = 'tmdb_5000_credits.csv'

# Helper 1: Extracts names from JSON-like strings
@st.cache_data
def get_list_names(data_string):
    if not isinstance(data_string, str) or not data_string.strip():
        return []
    try:
        data = literal_eval(data_string)
        return [item.get('name', '') for item in data]
    except:
        return []

# Helper 2: Extracts top 3 actors
@st.cache_data
def get_top_actors(cast_string, limit=3):
    if not isinstance(cast_string, str) or not cast_string.strip():
        return []
    try:
        cast = literal_eval(cast_string)
        # Sort by 'order' (billing) and take the top 'limit'
        names = [item.get('name', '') for item in sorted(cast, key=lambda x: x.get('order', np.inf))[:limit]]
        return names
    except:
        return []

# Helper 3: Extracts the Director
@st.cache_data
def get_director(crew_string):
    if not isinstance(crew_string, str) or not crew_string.strip():
        return ''
    try:
        crew = literal_eval(crew_string)
        for member in crew:
            if member.get('job', '') == 'Director':
                return member.get('name', '')
        return ''
    except:
        return ''

# Helper 4: Cleans names (lowercase, no spaces)
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    elif isinstance(x, str):
        return str.lower(x.replace(" ", ""))
    else:
        return ''
    
@st.cache_resource
def build_model():
    # 1. Load Data and Merge
    df_movies = pd.read_csv(MOVIES_FILE)
    df_credits = pd.read_csv(CREDITS_FILE)
    
    df_credits.columns = ['movie_id', 'title', 'cast', 'crew']
    df_movies.rename(columns={'id': 'movie_id'}, inplace=True)
    df = df_movies.merge(df_credits, on=['movie_id', 'title'])

    # 2. Process Features
    df['overview'] = df['overview'].fillna('')
    df['genres'] = df['genres'].apply(get_list_names)
    df['keywords'] = df['keywords'].apply(get_list_names)
    df['cast'] = df['cast'].apply(get_top_actors)
    df['director'] = df['crew'].apply(get_director)

    # 3. Clean Features (Lowercase, No Spaces)
    for feature in ['director', 'cast', 'genres', 'keywords']:
        df[feature] = df[feature].apply(clean_data)

    # 4. Create Content Soup
    df['soup'] = df['keywords'].apply(lambda x: ' '.join(x)) + ' ' + \
                 df['cast'].apply(lambda x: ' '.join(x)) + ' ' + \
                 df['genres'].apply(lambda x: ' '.join(x)) + ' ' + \
                 df['director'] + ' ' + \
                 df['overview'].str.lower()

    # 5. TF-IDF and Cosine Similarity
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['soup'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 6. Create Index Mapping
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    return df, cosine_sim, indices

# Run the model build once
df, cosine_sim, indices = build_model()

def get_recommendations(title, cosine_sim=cosine_sim, df=df, indices=indices, top_n=10):
    try:
        idx = indices[title]
    except KeyError:
        return "Movie not found in the dataset. Please select from the list."

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1] # Skip the movie itself
    
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the results as a DataFrame for easy display
    return df.iloc[movie_indices][['title', 'vote_average', 'vote_count', 'release_date']]

st.header("1️⃣ Content-Based Recommender (Find Similar Movies)")
st.markdown("Select a movie, and the engine will recommend 10 others based on **director, actors, genres, and plot summary.**")

# Select box for movie title
movie_titles = sorted(df['title'].tolist())
selected_movie = st.selectbox(
    "Choose a Movie:",
    movie_titles
)

if st.button("Get Content-Based Recommendations"):
    with st.spinner('Calculating similarities...'):
        recommendations_df = get_recommendations(selected_movie)
        
        st.subheader(f"Top 10 Movies Similar to '{selected_movie}'")
        st.dataframe(recommendations_df.style.format({
            'vote_average': "{:.1f}", 
            'vote_count': "{:,}"
        }), use_container_width=True)
        
st.header("2️⃣ Search Movies by Rating and Genre (Filtering)")
st.markdown("Filter movies by setting a minimum average user rating and selecting desired genres.")

col1, col2 = st.columns(2)

with col1:
    # Slider for minimum rating
    min_rating = st.slider("Minimum Average Rating (0.0 to 10.0):", 
                           min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    
with col2:
    # Multiselect for genres
    # Flatten the list of lists in 'genres' to get all unique genres
    all_genres = sorted(list(set(g for genres in df['genres'] for g in genres if g)))
    selected_genres = st.multiselect("Select Genres (Optional):", all_genres)

if st.button("Filter Movies by Criteria"):
    # Start with the rating filter
    filtered_df = df[df['vote_average'] >= min_rating]
    
    # Apply genre filter if genres are selected
    if selected_genres:
        # Filter for movies that contain AT LEAST ONE of the selected genres
        filtered_df = filtered_df[
            filtered_df['genres'].apply(lambda x: any(g in x for g in selected_genres))
        ]
    
    if filtered_df.empty:
        st.warning("No movies found matching your criteria. Try lowering the minimum rating or adjusting genres.")
    else:
        st.success(f"Found {len(filtered_df)} Movies Matching Criteria.")
        
        # Display the top results, sorted by vote average
        display_cols = ['title', 'vote_average', 'vote_count', 'release_date', 'genres']
        st.dataframe(filtered_df[display_cols]
                     .sort_values(by='vote_average', ascending=False)
                     .head(50) # Show top 50 matches
                     .style.format({'vote_average': "{:.1f}", 'vote_count': "{:,}"}), 
                     use_container_width=True)
        
        
