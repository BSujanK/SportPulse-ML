# ⚽ Football Analytics ML

An interactive Streamlit web app for football (soccer) analytics, powered by
classic machine learning models built on top of `scikit-learn`. It pulls
player and match data at runtime (falling back to synthetic data if the
sources are unreachable) and turns it into predictions, scores, clusters,
and recommendations — all wrapped in a dark, plasma-orange themed UI.

## Features

The app is organized into pages, accessible from the sidebar:

| Page | Description |
|---|---|
| 🏠 Dashboard | High-level KPIs and overview |
| ⚡ Match Predictor | Predicts match outcomes with a Random Forest classifier |
| 📈 Performance Scorer | Scores any player's performance with a Ridge regression model |
| 🗂️ Player Clusters | Groups players into archetypes with K-Means |
| 🤖 Player Recommender | Finds similar players using cosine similarity |
| 🌍 EDA Explorer | Explore the raw player/match data |
| 📊 League Table | Standings and head-to-head team stats |
| 🖼️ ML Chart Gallery | A gallery of all the ML-driven charts in one view |

**ML modules used:**
- Random Forest / Gradient Boosting — match outcome prediction
- Ridge / Linear Regression — player performance scoring
- K-Means — player archetype clustering
- Cosine similarity — player recommendation engine

## Data

On first run, the app tries to download FIFA player ratings and EPL match
results from a few public CSV sources on the internet. If none of those are
reachable, it automatically falls back to generating realistic synthetic
data, so the app always works out of the box.

## Getting Started

### Prerequisites
- Python 3.9+

### Installation

```bash
git clone https://github.com/BSujanK/football-analytics-ml.git
cd football-analytics-ml
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run football_app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

## Project Structure

```
football-analytics-ml/
├── football_app.py     # Main Streamlit app
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
├── .gitignore
└── README.md
```

## Tech Stack

Python · Streamlit · pandas · NumPy · scikit-learn · Matplotlib · Seaborn

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author

**B Sujan Kumar**
GitHub: [@BSujanK](https://github.com/BSujanK)
