import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt
import plotly.express as px

# Load datasets
tournament_data = pd.read_csv("ASA All PGA Raw Data - Tourn Level.csv")
owgr_data = pd.read_csv("OWGR_Ranking.csv")
stats_data = pd.read_csv("stats.csv")

# Normalize names across datasets
tournament_data["player"] = tournament_data["player"].str.strip().str.lower()
stats_data["PLAYER"] = stats_data["PLAYER"].str.strip().str.lower()

# Extract player names from OWGR data
# Only extract top 100 players from OWGR
owgr_names = pd.read_csv("OWGR_Ranking.csv", usecols=["Name", "Average_Points"])
top100_names = owgr_names.sort_values(by="Average_Points", ascending=False).head(100)["Name"].str.strip().str.lower().tolist()

# Merge features
features = tournament_data.groupby("player")[[
    "sg_putt", "sg_arg", "sg_app", "sg_ott", "sg_t2g", "sg_total"
]].mean().reset_index()
features = features.merge(stats_data, left_on="player", right_on="PLAYER", how="left")
features = features[features["player"].isin(top100_names)]

# Extract meaningful features for model training
# Performance metrics from PGA data
performance_cols = ["sg_putt", "sg_arg", "sg_app", "sg_ott", "sg_t2g", "sg_total", "AVG"]

# Add current form indicators using recent tournament data
# For simulation, we'll use existing features plus some statistical transformations
features["recent_sg_total"] = features["sg_total"] * np.random.uniform(0.8, 1.2, size=len(features))  # Simulate recent form
features["masters_skill"] = (features["sg_putt"] * 0.3 + features["sg_app"] * 0.4 + 
                            features["sg_arg"] * 0.2 + features["sg_ott"] * 0.1)  # Weighted skill relevant to Masters

# Add simulated round scores - could be replaced with real data during the tournament
features["Round1"] = np.random.randint(68, 75, size=len(features))
features["Round2"] = np.random.randint(68, 75, size=len(features))
features["Round3"] = np.random.randint(68, 75, size=len(features))
features["Round4"] = np.random.randint(68, 75, size=len(features))
features["Cumulative_Score"] = features[["Round1", "Round2", "Round3", "Round4"]].sum(axis=1)

# Create target based on predicted performance rather than hardcoding
# Scale "masters_skill" (0-1) and invert so lower is better (like golf scores)
features["predicted_score"] = -1 * (features["masters_skill"] - features["masters_skill"].min()) / (features["masters_skill"].max() - features["masters_skill"].min())
features["predicted_score"] = features["predicted_score"] + np.random.normal(0, 0.1, size=len(features))  # Add randomness

# Create binary labels for top 20% of players (potential winners)
threshold = features["predicted_score"].quantile(0.2)  # Top 20% are labeled as potential winners
features["potential_winner"] = (features["predicted_score"] <= threshold).astype(int)

# Prepare training data
X = features.drop(columns=["PLAYER", "player", "potential_winner", "predicted_score"])
y = features["potential_winner"]
X = X.fillna(X.mean())

# Split data without stratification since we now have multiple positive examples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train ensemble model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
lr_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier()

ensemble = VotingClassifier(estimators=[
    ('xgb', xgb_model),
    ('lr', lr_model),
    ('rf', rf_model)
], voting='soft')

ensemble.fit(X_train, y_train)

# Streamlit App
st.title("🏆 Masters 2025: Winner Prediction Dashboard")

# Prediction and leaderboard
probs = ensemble.predict_proba(X)[:, 1]
results = pd.DataFrame({
    "Player": features["player"].str.title(),
    "Win Probability": probs,
    "Masters Skill Score": features["masters_skill"].round(2),
    "Recent Form": features["recent_sg_total"].round(2),
    "Cumulative Score": features["Cumulative_Score"]
}).sort_values(by="Win Probability", ascending=False)

st.subheader("🔮 Predicted Win Probabilities (Top 20)")
st.dataframe(results.head(20).reset_index(drop=True))

# Radar Chart (Plotly)
st.subheader("📊 Player Profile Radar Chart")
selected_player = st.selectbox("Select a Player", results["Player"].tolist())
player_data = features[features["player"] == selected_player.lower()]

radar_features = ["sg_putt", "sg_arg", "sg_app", "sg_ott", "sg_t2g", "sg_total"]
radar_fig = px.line_polar(r=pd.Series(player_data[radar_features].iloc[0].values),
                          theta=radar_features,
                          line_close=True)
radar_fig.update_traces(fill='toself')
st.plotly_chart(radar_fig)

# SHAP Explanation
st.subheader("🧠 Model Interpretation with SHAP")
explainer = shap.Explainer(ensemble.named_estimators_["xgb"])
shap_values = explainer(X)

shap.summary_plot(shap_values, X, show=False)
fig = plt.gcf()
st.pyplot(fig)

# Evaluation
st.subheader("📈 Model Evaluation Report")
y_pred = ensemble.predict(X_test)
st.text(classification_report(y_test, y_pred))
