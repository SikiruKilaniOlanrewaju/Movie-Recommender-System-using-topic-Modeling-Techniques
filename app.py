"""
app.py
Streamlit web app for the Movie Recommender System using Topic Modeling
"""
import os
os.environ["STREAMLIT_WATCHER_IGNORE_PACKAGES"] = "true"

iimport streamlit as st
iimport pandas as pd
import requests
import json
from movie_recommender import preprocess, build_topic_model, recommend_movies
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# --- Custom CSS for a beautiful look ---
st.markdown('''
    <style>
    .main {
        background-color: #f7f9fa;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        border-radius: 12px;
        background: #fff;
        box-shadow: 0 2px 16px rgba(0,0,0,0.07);
    }
    .stButton>button {
        background: linear-gradient(90deg, #4f8cff 0%, #235390 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        margin: 0.2rem 0;
    }
    .stDownloadButton>button {
        background: #f7b731;
        color: #222;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        margin: 0.2rem 0;
    }
    .stTextInput>div>input {
        border-radius: 8px;
        border: 1px solid #dbeafe;
        padding: 0.5rem;
    }
    .stExpanderHeader {
        font-size: 1.1rem;
        font-weight: 700;
        color: #235390;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #235390;
    }
    .stSidebar {
        background: #e3eafc;
    }
    .stDataFrame {
        background: #f7f9fa;
    }
    </style>
''', unsafe_allow_html=True)

# --- Custom header with logo, subtitle, and project info ---
st.markdown("""
<div style='display: flex; align-items: center; gap: 1rem;'>
    <img src='https://img.icons8.com/color/96/000000/movie-projector.png' width='60' style='margin-bottom:0;'>
    <div>
        <h1 style='margin-bottom:0;'>A MOVIE RECOMMENDER SYSTEM USING TOPIC MODELING TECHNIQUES</h1>
        <p style='margin-top:0; color:#235390; font-size:1.2rem;'>BY <b>LAWAL MUIZ IDOWU</b><br>Matric Number: <b>PT/20/0168</b></p>
        <p style='margin-top:0; color:#888; font-size:1rem;'>Powered by Topic Modeling (NMF)</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Track which dataset is in use
if 'using_uploaded' not in st.session_state:
    st.session_state['using_uploaded'] = False

# Upload movie dataset
uploaded_file = st.file_uploader("Upload your movie dataset (CSV)", type=["csv"])

# Reset to default dataset button
if st.session_state.get('using_uploaded', False):
    if st.button("Reset to Default Dataset"):
        uploaded_file = None
        st.session_state['using_uploaded'] = False
        st.rerun()

# Download sample dataset button
with open("movies.csv", "rb") as f:
    st.download_button("Download Sample Dataset (CSV)", f, file_name="movies.csv")

# Load data (uploaded or default)
def get_movie_data(uploaded_file):
    try:
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state['using_uploaded'] = True
        else:
            df = pd.read_csv("movies.csv")
            st.session_state['using_uploaded'] = False
        if 'description' not in df.columns or 'title' not in df.columns:
            st.error("CSV must have 'title' and 'description' columns.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = get_movie_data(uploaded_file)

TMDB_API_KEY = "64e9c7b2e6e95088d9d7c7a5d0f4f39d"

def fetch_tmdb_data(title, api_key):
    """Fetch poster and overview from TMDb API."""
    if not api_key:
        return None, None
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={title}"
    try:
        resp = requests.get(url)
        data = resp.json()
        if data.get("results"):
            movie = data["results"][0]
            poster_path = movie.get("poster_path")
            overview = movie.get("overview")
            poster_url = f"https://image.tmdb.org/t/p/w200{poster_path}" if poster_path else None
            return poster_url, overview
    except Exception:
        pass
    return None, None

NOTES_FILE = "user_notes.json"

def load_notes():
    if os.path.exists(NOTES_FILE):
        with open(NOTES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_notes(notes):
    with open(NOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f)

notes = load_notes()

if df is not None:
    descriptions = preprocess(df['description'])
    vectorizer, nmf, W, H = build_topic_model(descriptions, n_topics=6)
    # Show topic keywords
    st.subheader("Topic Keywords (NMF)")
    n_top_words = 8
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(nmf.components_):
        st.write(f"**Topic {topic_idx+1}:** ", ', '.join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    # Genre filter
    genres = ['All'] + sorted(df['genre'].dropna().unique().tolist()) if 'genre' in df.columns else []
    selected_genre = st.selectbox("Filter by genre:", genres) if genres else 'All'
    if selected_genre != 'All':
        genre_mask = df['genre'] == selected_genre
        filtered_df = df[genre_mask]
    else:
        filtered_df = df
    movie_titles = filtered_df['title']
    # Search/filter box for movie titles
    search = st.text_input("Search for a movie:")
    filtered_titles = movie_titles[movie_titles.str.contains(search, case=False, na=False)] if search else movie_titles
    # --- Advanced Feature: Customizable Recommendation Count ---
    st.sidebar.header("Advanced Options")
    rec_count = st.sidebar.slider("Number of recommendations", min_value=1, max_value=10, value=5)
    # --- Advanced Feature: Genre-based Recommendations Toggle ---
    genre_restrict = False
    if 'genre' in df.columns:
        genre_restrict = st.sidebar.checkbox("Recommend only from same genre as selected movie", value=False)
    # --- Advanced Feature: Recent Activity Panel ---
    if 'recent_activity' not in st.session_state:
        st.session_state['recent_activity'] = []
    st.sidebar.header("Recent Activity")
    for entry in st.session_state['recent_activity'][-5:][::-1]:
        st.sidebar.write(f"{entry['movie']} → {', '.join(entry['recs'])}")
    # --- Advanced Feature: Favorites System ---
    if 'favorites' not in st.session_state:
        st.session_state['favorites'] = set()

    def toggle_favorite(movie_title):
        if movie_title in st.session_state['favorites']:
            st.session_state['favorites'].remove(movie_title)
        else:
            st.session_state['favorites'].add(movie_title)

    # --- Advanced Feature: Random Movie Picker ---
    if st.sidebar.button("Suggest a Random Movie"):
        import random
        rand_title = random.choice(df['title'].tolist())
        st.sidebar.success(f"Try: {rand_title}")

    st.sidebar.header("Your Favorites")
    if st.session_state['favorites']:
        for fav in st.session_state['favorites']:
            st.sidebar.write(fav)
    else:
        st.sidebar.caption("No favorites yet.")
    # --- Advanced Feature: Import/Export User Notes ---
    st.sidebar.header("Notes Backup")
    if st.sidebar.button("Export Notes as JSON"):
        with open(NOTES_FILE, "r", encoding="utf-8") as f:
            st.sidebar.download_button("Download Notes JSON", f, file_name="user_notes.json")
    imported_notes = st.sidebar.file_uploader("Import Notes JSON", type=["json"])
    if imported_notes:
        import json
        notes = json.load(imported_notes)
        save_notes(notes)
        st.sidebar.success("Notes imported! Reload the app to see changes.")
    # --- Advanced Feature: Session Analytics Panel ---
    st.sidebar.header("Session Analytics")
    if 'analytics' not in st.session_state:
        st.session_state['analytics'] = {'recommendations': 0, 'genre_counts': {}}
    st.sidebar.write(f"Total recommendations: {st.session_state['analytics']['recommendations']}")
    if st.session_state['analytics']['genre_counts']:
        st.sidebar.write("Most recommended genres:")
        for g, c in sorted(st.session_state['analytics']['genre_counts'].items(), key=lambda x: -x[1]):
            st.sidebar.write(f"{g}: {c}")
    # --- Advanced Feature: Movie Search by Description ---
    st.sidebar.header("Search by Description")
    desc_query = st.sidebar.text_input("Enter keyword(s) in description:")
    if desc_query:
        desc_mask = filtered_df['description'].str.contains(desc_query, case=False, na=False)
        filtered_df = filtered_df[desc_mask]
        movie_titles = filtered_df['title']
        filtered_titles = movie_titles[movie_titles.str.contains(search, case=False, na=False)] if search else movie_titles
    # --- Advanced Feature: Clear All Favorites/Notes ---
    if st.sidebar.button("Clear All Favorites"):
        st.session_state['favorites'] = set()
        st.sidebar.success("All favorites cleared.")
    if st.sidebar.button("Clear All Notes"):
        notes.clear()
        save_notes(notes)
        st.sidebar.success("All notes cleared.")
    # --- Advanced Feature: Show Most Similar Topic for Each Recommendation ---
    def get_top_topic(W_row):
        return int(W_row.argmax()) + 1

    # --- Add tabs for advanced layout ---
    tabs = st.tabs(["Get Recommendations", "Recommended Movies", "Analytics & Visualizations", "How it Works"])

    with tabs[0]:
        st.markdown('---')
        st.subheader('Get Recommendations')
        if len(filtered_titles) == 0:
            st.warning("No movies found matching your search.")
        else:
            selected_movie = st.selectbox("Select a movie to get recommendations:", filtered_titles)
            if st.button("Recommend"):
                idx = df[df['title'] == selected_movie].index[0]
                # Genre-based recommendations
                if genre_restrict and 'genre' in df.columns:
                    selected_genre_val = df.loc[idx, 'genre']
                    genre_indices = df[df['genre'] == selected_genre_val].index.tolist()
                    W_subset = W[genre_indices]
                    idx_in_subset = genre_indices.index(idx)
                    rec_indices_subset = recommend_movies(idx_in_subset, W_subset, top_n=rec_count)
                    rec_indices = [genre_indices[i] for i in rec_indices_subset]
                else:
                    rec_indices = recommend_movies(idx, W, top_n=rec_count)
                st.session_state['rec_data'] = []
                st.session_state['rec_titles'] = []
                st.session_state['selected_movie'] = selected_movie
                st.session_state['idx'] = idx
                for i in rec_indices:
                    rec_title = df['title'].iloc[i]
                    st.session_state['rec_titles'].append(rec_title)
                    with st.expander(f"{rec_title}"):
                        st.write(f"**Genre:** {df['genre'].iloc[i]}")
                        st.write(f"**Description:** {df['description'].iloc[i]}")
                        # TMDb API: Try to fetch poster and overview (if online)
                        poster_url, overview = fetch_tmdb_data(rec_title, TMDB_API_KEY)
                        if poster_url:
                            st.image(poster_url, width=150)
                        else:
                            st.caption("Poster not available (offline or not found).")
                        if overview:
                            st.write(overview)
                        else:
                            st.caption("Overview not available (offline or not found).")
                        # Persistent user notes
                        note_key = rec_title
                        note = notes.get(note_key, "")
                        new_note = st.text_area(f"Your notes for {rec_title}", value=note, key=f"note_{rec_title}")
                        if st.button(f"Save note for {rec_title}"):
                            notes[note_key] = new_note
                            save_notes(notes)
                            st.success("Note saved!")
                        # Show similarity score
                        sim_score = float(cosine_similarity([W[idx]], [W[i]])[0][0])
                        # Show most similar topic
                        top_topic = get_top_topic(W[i])
                        # Favorite button
                        fav_label = "Remove from Favorites" if rec_title in st.session_state['favorites'] else "Add to Favorites"
                        if st.button(fav_label, key=f"fav_{rec_title}"):
                            toggle_favorite(rec_title)
                        st.session_state['rec_data'].append({"title": rec_title, "similarity": sim_score, "note": new_note, "topic": top_topic})
                # Log recent activity
                st.session_state['recent_activity'].append({
                    'movie': selected_movie,
                    'recs': st.session_state['rec_titles']
                })
                # --- Analytics update ---
                st.session_state['analytics']['recommendations'] += 1
                genre_val = df.loc[idx, 'genre'] if 'genre' in df.columns else None
                if genre_val:
                    st.session_state['analytics']['genre_counts'][genre_val] = st.session_state['analytics']['genre_counts'].get(genre_val, 0) + 1

    with tabs[1]:
        st.markdown('---')
        st.subheader('Recommended Movies')
        # --- Show recommendations and advanced features if available in session_state ---
        rec_data = st.session_state.get('rec_data', [])
        selected_movie = st.session_state.get('selected_movie', None)
        idx = st.session_state.get('idx', None)
        show_all_overviews = st.checkbox("Show all overviews for recommended movies", value=False)
        if rec_data:
            st.write("Recommended Movies:")
            for r in rec_data:
                with st.expander(f"{r['title']}"):
                    st.write(f"**Similarity score:** {r['similarity']:.2f}")
                    st.write(f"**Most similar topic:** Topic {r['topic']}")
                    st.write(f"**Your note:** {r['note']}")
                    # --- Enhancement: Show poster and overview in the expander ---
                    poster_url, overview = fetch_tmdb_data(r['title'], TMDB_API_KEY)
                    if poster_url:
                        st.image(poster_url, width=150)
                        # Enhancement: Download poster button
                        try:
                            poster_bytes = requests.get(poster_url).content
                            st.download_button("Download Poster", poster_bytes, file_name=f"{r['title']}_poster.jpg")
                        except Exception:
                            st.caption("Poster download failed (offline or error).")
                    else:
                        st.caption("Poster not available (offline or not found).")
                    if overview:
                        if show_all_overviews:
                            st.write(overview)
                        else:
                            st.caption("(Enable 'Show all overviews' to display here)")
                        # Enhancement: Copy overview button
                        st.code(overview, language=None)
                        st.caption("Copy the above overview text if needed.")
                    else:
                        st.caption("Overview not available (offline or not found).")
                    # Enhancement: Show More Like This button
                    if st.button(f"Show More Like This: {r['title']}", key=f"more_like_{r['title']}"):
                        st.session_state['selected_movie'] = r['title']
                        st.rerun()
                    # Enhancement: Show release year if available
                    if 'year' in df.columns:
                        idx_year = df[df['title'] == r['title']].index[0]
                        st.write(f"**Year:** {df['year'].iloc[idx_year]}")
        # --- Advanced Feature: Download Recommendations with Topics ---
            if st.button("Download Recommendations as CSV"):
                rec_df = pd.DataFrame(rec_data)
                rec_df.to_csv("recommendations.csv", index=False)
                with open("recommendations.csv", "rb") as f:
                    st.download_button("Download CSV", f, file_name="recommendations.csv")
            # --- Advanced Feature: Show Recommendations as a Chart ---
            show_chart = st.checkbox("Show Recommendations Chart", key="show_chart_checkbox")
            if show_chart:
                fig, ax = plt.subplots()
                titles = [r['title'] for r in rec_data]
                scores = [r['similarity'] for r in rec_data]
                ax.barh(titles, scores, color='skyblue')
                ax.set_xlabel('Similarity Score')
                ax.set_title('Recommended Movies (Similarity)')
                st.pyplot(fig, clear_figure=True)
            # --- Advanced Feature: Show Topic Distribution for Selected Movie ---
            if W is not None and idx is not None and len(W[idx]) > 0:
                show_topic_dist = st.checkbox("Show Topic Distribution for Selected Movie", key=f"show_topic_dist_checkbox_{selected_movie}")
                if show_topic_dist:
                    fig2, ax2 = plt.subplots()
                    topic_weights = W[idx]
                    ax2.bar(range(1, len(topic_weights)+1), topic_weights, color='orange')
                    ax2.set_xlabel('Topic')
                    ax2.set_ylabel('Weight')
                    ax2.set_title(f'Topic Distribution for {selected_movie}')
                    st.pyplot(fig2, clear_figure=True)
            # --- Advanced Feature: Download Topic Distribution as CSV ---
            if st.button("Download Topic Distribution as CSV"):
                topic_dict = {f'Topic {i+1}': w for i, w in enumerate(W[idx])}
                topic_df = pd.DataFrame([topic_dict])
                topic_df.to_csv("topic_distribution.csv", index=False)
                with open("topic_distribution.csv", "rb") as f:
                    st.download_button("Download Topic Distribution CSV", f, file_name="topic_distribution.csv")
            # --- Advanced Feature: Movie Comparison Tool ---
            st.markdown('---')
            st.subheader('Compare Two Movies')
            compare_titles = st.multiselect('Select two movies to compare:', df['title'].tolist(), max_selections=2)
            if len(compare_titles) == 2:
                idx1 = df[df['title'] == compare_titles[0]].index[0]
                idx2 = df[df['title'] == compare_titles[1]].index[0]
                st.write(f"**{compare_titles[0]}** vs **{compare_titles[1]}**")
                # Show topic distributions
                fig3, ax3 = plt.subplots()
                ax3.bar(range(1, len(W[idx1])+1), W[idx1], alpha=0.6, label=compare_titles[0], color='blue')
                ax3.bar(range(1, len(W[idx2])+1), W[idx2], alpha=0.6, label=compare_titles[1], color='red')
                ax3.set_xlabel('Topic')
                ax3.set_ylabel('Weight')
                ax3.set_title('Topic Distribution Comparison')
                ax3.legend()
                st.pyplot(fig3)
                # Show similarity
                sim = float(cosine_similarity([W[idx1]], [W[idx2]])[0][0])
                st.info(f"Cosine similarity between movies: {sim:.2f}")
            # --- Advanced Feature: Session Summary Download ---
            if st.button('Download Session Summary (TXT)'):
                summary = []
                summary.append(f"Session Date: {pd.Timestamp.now()}")
                summary.append(f"Selected Movie: {selected_movie}")
                summary.append(f"Recent Activity: {st.session_state['recent_activity'][-5:]}")
                summary.append(f"Favorites: {list(st.session_state['favorites'])}")
                summary.append(f"Total Recommendations: {st.session_state['analytics']['recommendations']}")
                summary.append(f"Most Recommended Genres: {st.session_state['analytics']['genre_counts']}")
                notes_str = json.dumps(notes, indent=2)
                summary.append(f"Notes: {notes_str}")
                summary_txt = '\n\n'.join(summary)
                st.download_button('Download Session Summary', summary_txt, file_name='session_summary.txt')
            # --- Advanced Feature: Personalized Recommendation Profile ---
            st.markdown('---')
            st.subheader('Your Recommendation Profile')
            # Genre distribution
            if st.session_state['analytics']['genre_counts']:
                genres = list(st.session_state['analytics']['genre_counts'].keys())
                counts = list(st.session_state['analytics']['genre_counts'].values())
                fig4, ax4 = plt.subplots()
                ax4.pie(counts, labels=genres, autopct='%1.1f%%', startangle=90)
                ax4.set_title('Your Most Recommended Genres')
                st.pyplot(fig4)
            # Topic distribution (across all recommended movies)
            if st.session_state['analytics']['recommendations'] > 0:
                topic_hist = [0] * W.shape[1]
                for entry in st.session_state['recent_activity']:
                    for rec in entry['recs']:
                        rec_idx = df[df['title'] == rec].index[0]
                        top_topic = int(W[rec_idx].argmax())
                        topic_hist[top_topic] += 1
                if sum(topic_hist) > 0:
                    fig5, ax5 = plt.subplots()
                    ax5.bar(range(1, len(topic_hist)+1), topic_hist, color='purple')
                    ax5.set_xlabel('Topic')
                    ax5.set_ylabel('Count')
                    ax5.set_title('Most Frequent Topics in Your Recommendations')
                    st.pyplot(fig5)
            # Most frequent words in your recommended movies
            from collections import Counter
            if st.session_state['analytics']['recommendations'] > 0:
                all_desc = []
                for entry in st.session_state['recent_activity']:
                    for rec in entry['recs']:
                        rec_idx = df[df['title'] == rec].index[0]
                        all_desc.append(df['description'].iloc[rec_idx])
                words = [w.lower() for desc in all_desc for w in desc.split() if w.isalpha()]
                word_counts = Counter(words)
                if word_counts:
                    most_common = word_counts.most_common(10)
                    st.write('**Most Frequent Words in Your Recommendations:**')
                    st.write(', '.join([f'{w} ({c})' for w, c in most_common]))
    with tabs[2]:
        st.markdown('---')
        st.subheader('Analytics & Visualizations')
        rec_data = st.session_state.get('rec_data', [])
        selected_movie = st.session_state.get('selected_movie', None)
        idx = st.session_state.get('idx', None)
        # --- Advanced Feature: Download Recommendations with Topics ---
        if rec_data:
            if st.button("Download Recommendations as CSV", key="dl_recs_csv"):
                rec_df = pd.DataFrame(rec_data)
                rec_df.to_csv("recommendations.csv", index=False)
                with open("recommendations.csv", "rb") as f:
                    st.download_button("Download CSV", f, file_name="recommendations.csv")
            # --- Advanced Feature: Show Recommendations as a Chart ---
            show_chart = st.checkbox("Show Recommendations Chart", key="show_chart_checkbox_analytics")
            if show_chart:
                fig, ax = plt.subplots()
                titles = [r['title'] for r in rec_data]
                scores = [r['similarity'] for r in rec_data]
                ax.barh(titles, scores, color='skyblue')
                ax.set_xlabel('Similarity Score')
                ax.set_title('Recommended Movies (Similarity)')
                st.pyplot(fig, clear_figure=True)
            # --- Advanced Feature: Show Topic Distribution for Selected Movie ---
            if W is not None and idx is not None and len(W[idx]) > 0:
                show_topic_dist = st.checkbox("Show Topic Distribution for Selected Movie", key=f"show_topic_dist_checkbox_analytics_{selected_movie}")
                if show_topic_dist:
                    fig2, ax2 = plt.subplots()
                    topic_weights = W[idx]
                    ax2.bar(range(1, len(topic_weights)+1), topic_weights, color='orange')
                    ax2.set_xlabel('Topic')
                    ax2.set_ylabel('Weight')
                    ax2.set_title(f'Topic Distribution for {selected_movie}')
                    st.pyplot(fig2, clear_figure=True)
            # --- Advanced Feature: Download Topic Distribution as CSV ---
            if st.button("Download Topic Distribution as CSV", key="dl_topic_dist_csv"):
                topic_dict = {f'Topic {i+1}': w for i, w in enumerate(W[idx])}
                topic_df = pd.DataFrame([topic_dict])
                topic_df.to_csv("topic_distribution.csv", index=False)
                with open("topic_distribution.csv", "rb") as f:
                    st.download_button("Download Topic Distribution CSV", f, file_name="topic_distribution.csv")
            # --- Advanced Feature: Movie Comparison Tool ---
            st.markdown('---')
            st.subheader('Compare Two Movies')
            compare_titles = st.multiselect('Select two movies to compare:', df['title'].tolist(), max_selections=2, key="compare_titles_analytics")
            if len(compare_titles) == 2:
                idx1 = df[df['title'] == compare_titles[0]].index[0]
                idx2 = df[df['title'] == compare_titles[1]].index[0]
                st.write(f"**{compare_titles[0]}** vs **{compare_titles[1]}**")
                # Show topic distributions
                fig3, ax3 = plt.subplots()
                ax3.bar(range(1, len(W[idx1])+1), W[idx1], alpha=0.6, label=compare_titles[0], color='blue')
                ax3.bar(range(1, len(W[idx2])+1), W[idx2], alpha=0.6, label=compare_titles[1], color='red')
                ax3.set_xlabel('Topic')
                ax3.set_ylabel('Weight')
                ax3.set_title('Topic Distribution Comparison')
                ax3.legend()
                st.pyplot(fig3)
                # Show similarity
                sim = float(cosine_similarity([W[idx1]], [W[idx2]])[0][0])
                st.info(f"Cosine similarity between movies: {sim:.2f}")
            # --- Advanced Feature: Personalized Recommendation Profile ---
            st.markdown('---')
            st.subheader('Your Recommendation Profile')
            # Genre distribution
            if st.session_state['analytics']['genre_counts']:
                genres = list(st.session_state['analytics']['genre_counts'].keys())
                counts = list(st.session_state['analytics']['genre_counts'].values())
                fig4, ax4 = plt.subplots()
                ax4.pie(counts, labels=genres, autopct='%1.1f%%', startangle=90)
                ax4.set_title('Your Most Recommended Genres')
                st.pyplot(fig4)
            # Topic distribution (across all recommended movies)
            if st.session_state['analytics']['recommendations'] > 0:
                topic_hist = [0] * W.shape[1]
                for entry in st.session_state['recent_activity']:
                    for rec in entry['recs']:
                        rec_idx = df[df['title'] == rec].index[0]
                        top_topic = int(W[rec_idx].argmax())
                        topic_hist[top_topic] += 1
                if sum(topic_hist) > 0:
                    fig5, ax5 = plt.subplots()
                    ax5.bar(range(1, len(topic_hist)+1), topic_hist, color='purple')
                    ax5.set_xlabel('Topic')
                    ax5.set_ylabel('Count')
                    ax5.set_title('Most Frequent Topics in Your Recommendations')
                    st.pyplot(fig5)
            # Most frequent words in your recommended movies
            from collections import Counter
            if st.session_state['analytics']['recommendations'] > 0:
                all_desc = []
                for entry in st.session_state['recent_activity']:
                    for rec in entry['recs']:
                        rec_idx = df[df['title'] == rec].index[0]
                        all_desc.append(df['description'].iloc[rec_idx])
                words = [w.lower() for desc in all_desc for w in desc.split() if w.isalpha()]
                word_counts = Counter(words)
                if word_counts:
                    most_common = word_counts.most_common(10)
                    st.write('**Most Frequent Words in Your Recommendations:**')
                    st.write(', '.join([f'{w} ({c})' for w, c in most_common]))
    with tabs[3]:
        st.markdown('---')
        st.subheader('How it works')
        st.markdown("""
        **Movie Recommender System**
        - By default, the app uses a built-in sample movie dataset.
        - You can upload your own CSV to use custom data (must have 'title', 'description', and 'genre' columns).
        - Use the 'Reset to Default Dataset' button to switch back to the sample data.
        - Download the sample dataset as a template if needed.
        - Search for a movie and select it to get recommendations.
        - All features work offline with the built-in dataset.
        """)
# --- Optional: Add card-style layout for recommended movies ---
# (Inside the Recommended Movies tab, wrap each expander in a styled div for card effect)
# Example:
# with st.expander(f"{r['title']}"):
#     st.markdown("""<div style='background:#f7f9fa; border-radius:10px; box-shadow:0 2px 8px #dbeafe; padding:1rem; margin-bottom:1rem;'>""", unsafe_allow_html=True)
#     ...movie info...
#     st.markdown("</div>", unsafe_allow_html=True)

# ---
# Extensibility: Placeholder for future features
# - Integrate user ratings for hybrid recommendations
# - Connect to external APIs (TMDb, IMDb) for richer data
# - Add user authentication and logging
# ---

# --- Add a footer ---
st.markdown("""
<div style='text-align:center; color:#888; margin-top:2rem;'>
    <small>Movie Recommender System &copy; 2025. Built with Streamlit & NMF Topic Modeling.</small>
</div>
""", unsafe_allow_html=True)
