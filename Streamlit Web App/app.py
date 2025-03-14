# app.py
import streamlit as st
import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import os
import time

# Configure FastF1 cache
cache_dir = './f1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# Track configuration
TRACKS = {
    'Melbourne': [2022, 2023],
    'Bahrain': [2022, 2023],
    'Silverstone': [2022, 2023],
    'Monza': [2022, 2023],
    'Suzuka': [2022, 2023]
}

@st.cache_data
def load_and_prepare_data():
    lap_data = []
    feature_columns = [
        'LapNumber', 'Tyre', 'AirTemp', 'TrackTemp',
        'Humidity', 'TyreAge', 'Driver', 'Track'
    ]
    
    for track_name, years in TRACKS.items():
        for year in years:
            try:
                session = fastf1.get_session(year, track_name, 'R')
                session.load()
                weather = session.weather_data
                avg_temp = weather['AirTemp'].mean()
                avg_track_temp = weather['TrackTemp'].mean()
                avg_humidity = weather['Humidity'].mean()

                for driver in session.drivers:
                    laps = session.laps.pick_driver(driver)
                    if len(laps) == 0: continue
                    
                    laps = laps.copy()
                    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
                    laps['Tyre'] = laps['Compound'].map({"SOFT":0, "MEDIUM":1, "HARD":2})
                    laps['Driver'] = pd.factorize(laps['Driver'])[0]
                    laps['Track'] = track_name
                    laps['AirTemp'] = avg_temp
                    laps['TrackTemp'] = avg_track_temp
                    laps['Humidity'] = avg_humidity
                    laps['TyreAge'] = laps['LapNumber']
                    
                    lap_data.append(laps[feature_columns + ['LapTimeSeconds']])

                print(f"✅ Loaded {track_name} {year}")
                
            except Exception as e:
                print(f"⚠️ Skipped {track_name} {year}: {str(e)[:100]}")
                continue
                
    df = pd.concat(lap_data, ignore_index=True)
    df['Track'] = pd.factorize(df['Track'])[0]
    return df.dropna()

# Load data
df = load_and_prepare_data()

# Train model
X = df.drop('LapTimeSeconds', axis=1)
y = df['LapTimeSeconds']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.1)
model.fit(X_train, y_train)

# Calculate metrics
start_time = time.time()
y_pred = model.predict(X_test)
pred_time = (time.time() - start_time)/len(X_test)*1e3
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

model.get_booster().feature_names = X.columns.tolist()

# Streamlit UI
st.title("F1 Strategy Predictor 🏎️💨 by Yash Vaman")
st.write("Race strategy simulation with tyre management analysis")

# Sidebar
with st.sidebar:
    st.header("Model Performance")
    col1, col2 = st.columns(2)
    with col1: st.metric("R² Score", f"{r2:.2f}")
    with col2: st.metric("MAE", f"{mae:.2f}s")
    st.caption(f"⚡ Prediction speed: {pred_time:.2f}ms/lap")
    st.caption(f"📊 Training data: {len(df):,} laps")
    st.divider()
    
    st.header("Simulation Parameters")
    track = st.selectbox("Circuit", list(TRACKS.keys()))
    air_temp = st.slider("Air Temp (°C)", 15, 45, 25)
    track_temp = st.slider("Track Temp (°C)", 25, 60, 40)
    humidity = st.slider("Humidity (%)", 20, 100, 45)
    weather = st.radio("Conditions", ["Dry", "Wet"])

def create_strategy(strategy_type):
    base = {
        'LapNumber': range(1,56),
        'AirTemp': air_temp,
        'TrackTemp': track_temp,
        'Humidity': humidity,
        'TyreAge': range(1,56),
        'Driver': 1,
        'Track': list(TRACKS.keys()).index(track)
    }
    
    if strategy_type == "1-stop":
        base['Tyre'] = [0]*25 + [2]*30
    elif strategy_type == "2-stop":
        base['Tyre'] = [0]*15 + [1]*25 + [2]*15
        
    return pd.DataFrame({col: base[col] for col in X.columns})

def plot_strategy(ax, data, times, title):
    tyre_colors = {
        0: ('Soft', '#FF3333'),
        1: ('Medium', '#FFD700'),
        2: ('Hard', '#FFFFFF')
    }
    
    # Plot lap times
    ax.plot(data['LapNumber'], times, color='#00D2BE', lw=2)
    
    # Tyre compound zones
    current_tyre = None
    start_lap = 1
    for lap, tyre in enumerate(data['Tyre'], start=1):
        if tyre != current_tyre:
            if current_tyre is not None:
                ax.axvspan(start_lap-0.5, lap-0.5, 
                          color=tyre_colors[current_tyre][1], alpha=0.2)
            current_tyre = tyre
            start_lap = lap
    ax.axvspan(start_lap-0.5, 55.5, color=tyre_colors[current_tyre][1], alpha=0.2)

    # Pit stop markers
    pit_laps = []
    for i in range(1, len(data)):
        if data['Tyre'].iloc[i] != data['Tyre'].iloc[i-1]:
            pit_laps.append(i+1)
    
    for plap in pit_laps:
        ax.axvline(plap-0.5, color='#FF8700', ls='--', lw=1.5, alpha=0.9)
        ax.text(plap-0.5+0.5, np.max(times)-2, 
                f'PIT STOP\n{tyre_colors[data["Tyre"].iloc[plap-2]][0]} → {tyre_colors[data["Tyre"].iloc[plap-1]][0]}',
                rotation=90, va='top', ha='left', fontsize=9,
                bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))

    # Formatting
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Lap Number", fontsize=10)
    ax.set_ylabel("Lap Time (s)", fontsize=10)
    ax.grid(alpha=0.15)
    ax.set_xlim(0.5, 55.5)
    ax.set_ylim(np.min(times)-1, np.max(times)+3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=v[1], alpha=0.2, label=v[0]) 
        for v in tyre_colors.values()
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

if st.button("🚦 Simulate Strategies"):
    # Generate strategies
    one_stop = create_strategy("1-stop")
    two_stop = create_strategy("2-stop")
    
    # Predict times
    one_stop_times = model.predict(one_stop)
    two_stop_times = model.predict(two_stop)
    
    # Weather penalty
    if weather == "Wet":
        one_stop_times += np.random.uniform(2, 5, 55)
        two_stop_times += np.random.uniform(2, 5, 55)
    
    # Calculate totals
    total_one_stop = one_stop_times.sum() + 25*1
    total_two_stop = two_stop_times.sum() + 25*2
    
    # Individual strategy plots
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(10,5))
        plot_strategy(ax1, one_stop, one_stop_times, "1-Stop Strategy Analysis")
        st.pyplot(fig1)
        st.metric("Total Race Time", f"{total_one_stop/60:.2f} mins", 
                 delta=f"Δ {total_one_stop-total_two_stop:.1f}s vs 2-Stop")

    with col2:
        fig2, ax2 = plt.subplots(figsize=(10,5))
        plot_strategy(ax2, two_stop, two_stop_times, "2-Stop Strategy Analysis")
        st.pyplot(fig2)
        st.metric("Total Race Time", f"{total_two_stop/60:.2f} mins")

    # Combined comparison plot
    st.divider()
    st.subheader("Direct Strategy Comparison")
    fig3, ax3 = plt.subplots(figsize=(12,4))
    ax3.plot(one_stop['LapNumber'], one_stop_times, label='1-Stop', color='#00D2BE')
    ax3.plot(two_stop['LapNumber'], two_stop_times, label='2-Stop', color='#DC0000')
    ax3.set_title("Lap Time Comparison Overview", fontsize=14)
    ax3.set_xlabel("Lap Number")
    ax3.set_ylabel("Lap Time (s)")
    ax3.grid(alpha=0.2)
    ax3.legend()
    st.pyplot(fig3)

st.caption("Note: Simulation includes 25s pit stop time. Data sourced from FastF1 historical records. Also Safety Car is not taken into consideration")
st.caption("MADE BY YASH VAMAN")