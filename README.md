# Movie Recommender System Using Topic Modeling Techniques

A robust, offline-compatible movie recommender system built with Python and Streamlit, leveraging topic modeling (NMF) for content-based recommendations. The app features dataset uploads, genre filtering, user notes, advanced analytics/visualizations, and a beautiful, professional UI. Works fully offline, with optional online enhancements (TMDb posters/overviews).

## Features
- Content-based movie recommendations using NMF topic modeling
- Upload your own movie dataset (CSV)
- Genre filtering and search
- User notes per movie (persistent)
- Favorites system and recent activity
- Advanced analytics & visualizations (charts, topic distributions, comparison tools, session summary, personalized profile)
- Random movie picker
- Import/export user notes
- Download recommendations and analytics as CSV/TXT
- TMDb API integration for posters/overviews (with offline fallback)
- Modern, professional UI with custom CSS, logo, and organized layout

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone or download this repository.
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. (Optional) Set your TMDb API key in `app.py` for poster/overview fetching.

### Running the App
```sh
streamlit run app.py
```

### Usage
- By default, the app uses the included `movies.csv` sample dataset.
- Upload your own CSV (must have `title`, `description`, and `genre` columns) to use custom data.
- Use the sidebar and tabs to explore recommendations, analytics, and more.

## File Structure
- `app.py` - Streamlit web app
- `movie_recommender.py` - Core recommendation logic
- `movies.csv` - Sample movie dataset
- `requirements.txt` - Python dependencies

## Screenshots
*(Add screenshots of the UI here for best results)*

## License
See [LICENSE](LICENSE).

## Author
**Lawal Muiz Idowu**  
Matric Number: PT/20/0168

---
*Built with Streamlit & NMF Topic Modeling. 2025.*
