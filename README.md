# ⚽ AI Football Scout

[[Streamlit App]](https://ai-football-scout-zuqzqe3auivnzsfmvavgdh.streamlit.app/)

A machine learning-powered tool that analyses player event data and identifies statistical fits for specific tactical systems. Upload your data, select a position and playing style, and the AI will recommend the best-matching players from your dataset.

## 🌟 Live Demo

Try it out without setting up anything locally:

👉 **[Launch AI Football Scout](YOUR_STREAMLIT_APP_LINK_HERE)**

## ✨ Features

- **Position-Aware Scouting** — Find fits for Defenders, Midfielders, and Attackers
- **Tactical System Matching** — Search for players suited to Possession, High Pressing, or Counter Attack styles
- **Confidence Scores** — Each recommendation comes with a percentage confidence rating
- **CSV Upload** — Simply upload your player data and get instant results
- **Built-in Demo** — Try the tool immediately with a sample dataset

## 🚀 How It Works

1. Upload a CSV file containing player event data (per 90 stats)
2. Select the position you're scouting and your preferred tactical system
3. The AI model predicts which players statistically match your criteria
4. Results are sorted by confidence score, with the best fits at the top

## 📊 Dataset Requirements

Your CSV must include the following identifier columns:

| Column | Description |
|--------|-------------|
| `player_name` | Player name |
| `team_name` | Team name |
| `minutes_played` | Minutes played |
| `primary_position` OR `position_group` | Position or group (Defender/Midfielder/Attacker) |

### Tactical Metrics by Position

The model was trained on advanced metrics derived from **StatsBomb** data. Your dataset should include the following features (per 90):

**Defenders:**
`total_passes_per90`, `pass_completion_pct`, `passes_into_final_third_per90`, `progressive_carries_per90`, `avg_defensive_distance`, `total_pressures_per90`, `counterpressures_per90`, `successful_long_passes_per90`, `pass_directness_ratio`

**Midfielders:**
`total_passes_per90`, `pass_completion_pct`, `passes_into_final_third_per90`, `pressure_receipt_pct`, `progressive_carries_per90`, `total_pressures_per90`, `final_third_pressures_per90`, `counterpressures_per90`, `high_ball_recoveries_per90`, `explosive_carries_per90`, `pass_directness_ratio`, `successful_long_passes_per90`

**Attackers:**
`touches_in_box_per90`, `pass_completion_pct`, `pressure_receipt_pct`, `final_third_pressures_per90`, `counterpressures_per90`, `high_ball_recoveries_per90`, `explosive_carries_per90`, `avg_shot_quality`, `aerial_duels_won_per90`, `avg_receipt_distance_from_goal`

> **Tip:** Filter out players with low minutes played before uploading for the most accurate results.

## 🛠️ Installation & Local Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/AI-Football-Scout.git
cd AI-Football-Scout

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## 📁 Project Structure

```
AI-Football-Scout/
├── app.py                     # Streamlit web application
├── scouting_models.pkl        # Saved ML models (position × style)
├── tactical_scalers.pkl       # Saved feature scalers
├── Combined_Dataset_Final.csv # Demo/sample dataset
├── requirements.txt           # Python dependencies
├── notebooks/
│   ├── 01_feature_engineering.ipynb
│   └── 02_model_development.ipynb
└── README.md
```

## 🧠 Model Details

- **Algorithm:** Supervised classification (trained on StatsBomb open data)
- **Approach:** Position-specific models for each tactical system (Possession, High Pressing, Counter Attack)
- **Scaling:** Feature standardisation via fitted scalers
- **Threshold:** Players are classified as a "fit" if the model confidence exceeds 50%

> **Note:** Training data includes StatsBomb open database records, primarily from the 2015/2016 season. Results may vary when applied to more modern datasets.

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---
