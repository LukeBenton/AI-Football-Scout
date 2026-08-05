import streamlit as st
import pandas as pd
import joblib

# Page Setup
st.set_page_config(page_title="AI Football Scout", layout="wide")
st.title("⚽ AI Football Scout")
st.markdown("Upload a CSV of player event data to instantly find statistical fits for your tactical system.")

# Initialise session state so demo dataset doesn't unload
if 'use_demo_data' not in st.session_state:
    st.session_state.use_demo_data = False

# Documentation and Instructions
with st.expander("📖 How to use this app & Dataset Requirements"):
    st.markdown("""
    ### Welcome to the AI Football Scout
    This tool uses Machine Learning to evaluate player event data and find perfect statistical fits for specific tactical systems.
    
    #### 📁 Dataset Requirements
    To use this app, please upload a CSV file containing player event data. Your dataset **must** include the following identifier columns:
    * `player_name` (or an unlabelled index column)
    * `team_name`
    * `minutes_played`
    * `primary_position` OR `position_group` (e.g., 'Defender', 'Midfielder', 'Attacker')
    
    #### 📊 Required Tactical Metrics (Per 90)
    The AI was trained on specific advanced metrics derived from StatsBomb data. Depending on the position you are scouting, your CSV needs these columns:
    
    **For Defenders:**
    `total_passes_per90`, `pass_completion_pct`, `passes_into_final_third_per90`, `progressive_carries_per90`, `avg_defensive_distance`, `total_pressures_per90`, `counterpressures_per90`, `successful_long_passes_per90`, `pass_directness_ratio`
    
    **For Midfielders:**
    `total_passes_per90`, `pass_completion_pct`, `passes_into_final_third_per90`, `pressure_receipt_pct`, `progressive_carries_per90`, `total_pressures_per90`, `final_third_pressures_per90`, `counterpressures_per90`, `high_ball_recoveries_per90`, `explosive_carries_per90`, `pass_directness_ratio`, `successful_long_passes_per90`
    
    **For Attackers:**
    `touches_in_box_per90`, `pass_completion_pct`, `pressure_receipt_pct`, `final_third_pressures_per90`, `counterpressures_per90`, `high_ball_recoveries_per90`, `explosive_carries_per90`, `avg_shot_quality`, `aerial_duels_won_per90`, `avg_receipt_distance_from_goal`
    
    *Tip: For the most accurate results, ensure players with low minutes played are filtered out before uploading.*
                
    *This model was trained using the Statsbomb open database which is very limited. A lot of the training data is from 2015/2016 so results may not be perfect if more modern datasets are loaded. The demo dataset also uses this data, and may copies of the same player from different seasons.*
    """)

# Initialize models in session state (lazy loading)
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.models = None
    st.session_state.scalers = None

def load_models_lazy():
    """Load models on first use, not at startup"""
    import os
    
    if st.session_state.models_loaded:
        return st.session_state.models, st.session_state.scalers
    
    with st.spinner("Loading AI models (this may take a moment)..."):
        try:
            model_path = "scouting_models.pkl"
            scaler_path = "tactical_scalers.pkl"
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"'{model_path}' not found")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"'{scaler_path}' not found")
            
            import time
            start = time.time()
            
            models = joblib.load(model_path)
            scalers = joblib.load(scaler_path)
            
            elapsed = time.time() - start
            st.session_state.models = models
            st.session_state.scalers = scalers
            st.session_state.models_loaded = True
            
            return models, scalers
            
        except Exception as e:
            raise Exception(f"Failed to load models: {e}")

# Remove the module-level loading - models are loaded lazily when user interacts

# Sidebar UI
st.sidebar.header("Scouting Parameters")

# Dropdowns for user selection
selected_position = st.sidebar.selectbox("Target Position", ["Defender", "Midfielder", "Attacker"])
selected_style = st.sidebar.selectbox("Tactical System", ["Possession", "High Pressing", "Counter Attack"])

st.sidebar.markdown("---")
# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload Player Data (CSV)", type=["csv"])

st.sidebar.markdown("**Don't have a dataset?**")
# Demo button
# use_demo_data = st.sidebar.button("📊 Load Demo Dataset")

# Demo button logic
if st.sidebar.button("📊 Load Demo Dataset"):
    # When clicked, flip the memory flag to true
    st.session_state.use_demo_data = True

# Feature definitions (same as in training)
positional_features = {
    "Defender": [
        "total_passes_per90", "pass_completion_pct", "passes_into_final_third_per90", 
        "progressive_carries_per90", "avg_defensive_distance", "total_pressures_per90", 
        "counterpressures_per90", "successful_long_passes_per90", "pass_directness_ratio"
    ],
    "Midfielder": [
        "total_passes_per90", "pass_completion_pct", "passes_into_final_third_per90", 
        "pressure_receipt_pct", "progressive_carries_per90", "total_pressures_per90", 
        "final_third_pressures_per90", "counterpressures_per90", "high_ball_recoveries_per90", 
        "explosive_carries_per90", "pass_directness_ratio", "successful_long_passes_per90"
    ],
    "Attacker": [
        "touches_in_box_per90", "pass_completion_pct", "pressure_receipt_pct",
        "final_third_pressures_per90", "counterpressures_per90", "high_ball_recoveries_per90", 
        "explosive_carries_per90", "avg_shot_quality", "aerial_duels_won_per90", 
        "avg_receipt_distance_from_goal"
    ]
}

# Main logic
if uploaded_file is not None or st.session_state.use_demo_data:

    if uploaded_file is not None:
        # Read the uploaded dataset
        df = pd.read_csv(uploaded_file)
        st.session_state.use_demo_data = False
    else:
        # Load the demo dataset
        try:
            df = pd.read_csv("Combined_Dataset_Final.csv")
        except FileNotFoundError:
            st.error("⚠️ Demo dataset 'Combined_Dataset_Final.csv' not found. Please upload your own CSV file.")
            st.stop()
        st.sidebar.success("Demo dataset loaded successfully!")

    # Deal with blank columns titles (e.g. player name not having a title like in the demo dataset)
    if "Unnamed: 0" in df.columns:
        # If player_name is missing rename it
        if "player_name" not in df.columns:
            df = df.rename(columns={"Unnamed: 0": "player_name"})
        # If player_name is already there just delete the blank column
        else:
            df = df.drop(columns=["Unnamed: 0"])
    
    st.write(f"### 🔍 Scouting {selected_position}s for a {selected_style} system...")
    
    # Position mapping if uploaded dataset is not correctly formatted (eg the demo dataset)
    if "position_group" not in df.columns:
        if "primary_position" in df.columns:
            st.info("🔄 'position_group' column not found. Automatically categorizing players based on 'primary_position'...")
            
            # The same mapping as from the model development stage
            position_mapping = {
                # Defenders
                "Right Back": "Defender", "Left Back": "Defender", "Center Back": "Defender",
                "Right Center Back": "Defender", "Left Center Back": "Defender",
                "Right Wing Back": "Defender", "Left Wing Back": "Defender",
                
                # Midfielders
                "Center Defensive Midfield": "Midfielder", "Right Defensive Midfield": "Midfielder", 
                "Left Defensive Midfield": "Midfielder", "Right Center Midfield": "Midfielder", 
                "Left Center Midfield": "Midfielder", "Right Midfield": "Midfielder", 
                "Left Midfield": "Midfielder", "Center Attacking Midfield": "Midfielder", 
                "Right Attacking Midfield": "Midfielder", "Left Attacking Midfield": "Midfielder",
                
                # Attackers
                "Center Forward": "Attacker", "Right Center Forward": "Attacker", 
                "Left Center Forward": "Attacker", "Right Wing": "Attacker", "Left Wing": "Attacker"
            }
            
            # Apply the mapping
            df["position_group"] = df["primary_position"].map(position_mapping)
        else:
            st.error("❌ Error: The dataset must contain either a 'position_group' or 'primary_position' column to continue.")
            st.stop()
        
    # Filter the dataset to only include the selected position
    position_df = df[df["position_group"] == selected_position].copy()
    
    if position_df.empty:
        st.warning(f"No {selected_position}s found in the uploaded dataset.")
    else:
        # Load models on first use (lazy loading)
        try:
            models, scalers = load_models_lazy()
        except Exception as e:
            st.error(f"⚠️ Could not load models: {e}")
            st.stop()
        
        # Grab the correct features, model, and scaler based on dropdowns
        active_features = positional_features[selected_position]
        active_model = models[selected_position][selected_style]
        active_scaler = scalers[selected_position]
        
        # Ensure all required features exist in the uploaded CSV
        missing_cols = [col for col in active_features if col not in position_df.columns]
        if missing_cols:
            st.error(f"Missing required columns in dataset: {missing_cols}")
            st.stop()
        
        # Isolate and scale the data
        X_raw = position_df[active_features]
        X_scaled = active_scaler.transform(X_raw)
        
        # Make the predictions
        # position_df["Fit_Prediction"] = active_model.predict(X_scaled)
        
        # Ask the model exactly which column holds the '1' (Fit) probability
        fit_class_index = list(active_model.classes_).index(1)
        position_df["Confidence_Score"] = active_model.predict_proba(X_scaled)[:, fit_class_index]

        # Force the prediction to be strictly based on the 50% confidence threshold
        position_df["Fit_Prediction"] = (position_df["Confidence_Score"] >= 0.50).astype(int)
        
        # 4. Filter to only show players the model says are a Fit
        recommended_players = position_df[position_df["Fit_Prediction"] == 1]
        
        # Sort them so the highest confidence fits are at the top
        recommended_players = recommended_players.sort_values(by="Confidence_Score", ascending=False)
        
        # Display the results
        if recommended_players.empty:
            st.info("No players perfectly matched this profile with over 50% confidence. Try scouting a different position or style.")
        else:
            st.success(f"Found {len(recommended_players)} players who fit this profile!")
            
            # Format the confidence score to look like a clean percentage
            recommended_players["Confidence_Score"] = (recommended_players["Confidence_Score"] * 100).round(1).astype(str) + "%"
            
            # Decide which columns to display 
            display_columns = ["player_name", "team_name", "minutes_played", "Confidence_Score"] + active_features
            
            # Handle cases where 'player_name' or 'team_name' might not be in the final CSV
            final_display = [col for col in display_columns if col in recommended_players.columns]
            
            # Render the finished dataframe
            st.dataframe(
                recommended_players[final_display],
                use_container_width=True,
                hide_index=True
            )

else:
    # Instructions displayed when the app first loads
    st.info("👈 Please upload your generated CSV file in the sidebar to begin.")