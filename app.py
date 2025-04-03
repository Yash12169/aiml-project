from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# flask app
app = Flask(__name__)

# Load your dataset
df = pd.read_csv('clustered_df.csv')

numerical_features = [
    "valence", "danceability", "energy", "tempo",
    "acousticness", "liveness", "speechiness", "instrumentalness"
]

def normalize_name(name):
    """Normalize song name for better matching (lowercase, remove punctuation)"""
    return re.sub(r'[^\w\s]', '', name.lower()).strip()

# Add normalized column to dataset for better matching
df['normalized_name'] = df['name'].apply(normalize_name)

def recommend_songs(song_name, df, num_recommendations=5):
    """Recommend songs based on input song name with fuzzy matching"""
    # Normalize the input song name
    normalized_input = normalize_name(song_name)
    
    # Try exact match first on normalized names
    matching_songs = df[df["normalized_name"] == normalized_input]
    
    # If no exact match, try partial match
    if matching_songs.empty:
        matching_songs = df[df["normalized_name"].str.contains(normalized_input)]
    
    # If still no match, return empty
    if matching_songs.empty:
        return pd.DataFrame()
    
    # Use the first match
    match_song = matching_songs.iloc[0]
    song_cluster = match_song["Cluster"]
    
    # Get songs from the same cluster
    same_cluster_songs = df[df["Cluster"] == song_cluster]
    song_index = same_cluster_songs[same_cluster_songs["name"] == match_song["name"]].index[0]
    
    # Calculate similarity
    cluster_features = same_cluster_songs[numerical_features]
    similarity = cosine_similarity(cluster_features, cluster_features)
    
    # Get similar songs indices
    similar_songs = np.argsort(similarity[song_index])[-(num_recommendations + 1):-1][::-1]
    
    # Get recommendations
    recommendations = same_cluster_songs.iloc[similar_songs][["name", "year", "artists"]]
    return recommendations

# route
@app.route("/")
def index():
    # Pass some sample song names to the template for demonstration
    sample_songs = df['name'].sample(10).tolist()
    return render_template('index.html', sample_songs=sample_songs)

@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    recommendations = []
    song_name = ""
    error_message = ""
    
    if request.method == "POST":
        song_name = request.form.get("song_name")
        try:
            result_df = recommend_songs(song_name, df)
            if result_df.empty:
                error_message = f"No song found matching '{song_name}'. Please try another song name."
            else:
                recommendations = result_df.to_dict(orient="records")
        except Exception as e:
            error_message = f"Error: {str(e)}"
    
    return render_template("index.html", 
                          recommendations=recommendations, 
                          song_name=song_name,
                          error_message=error_message,
                          sample_songs=df['name'].sample(10).tolist())

if __name__ == "__main__":
    app.run(debug=True)