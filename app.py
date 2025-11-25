import os
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras  # pyright: ignore[reportMissingImports]

warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

LAG_FEATURES = [
    "footfall_lag_1",
    "footfall_lag_7",
    "footfall_lag_30",
    "footfall_rolling_7",
    "footfall_rolling_30",
]
WEATHER_OPTIONS = ["Clear", "Clouds", "Rain"]
SEASON_OPTIONS = ["Winter", "Summer", "Monsoon"]
TIME_STEPS = 7
WEATHERAPI_ENDPOINT = "https://api.weatherapi.com/v1/current.json"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"  # Using a reliable model that works

SCENARIO_PRESETS = {
    "Perfect Weather": {
        "weather_condition": "Clear",
        "temp_min": 20,
        "temp_max": 30,
        "humidity": 45,
        "pressure": 1012,
        "wind_speed": 2,
        "is_festival": False,
    },
    "Festival Boost": {
        "weather_condition": "Clear",
        "temp_min": 22,
        "temp_max": 32,
        "humidity": 50,
        "pressure": 1010,
        "wind_speed": 3,
        "is_festival": True,
    },
    "Monsoon Watch": {
        "weather_condition": "Rain",
        "temp_min": 18,
        "temp_max": 26,
        "humidity": 70,
        "pressure": 1004,
        "wind_speed": 5,
        "is_festival": False,
    },
}

PERIOD_OPTIONS = {
    "Last 3 Months": 3,
    "Last 6 Months": 6,
    "Last 12 Months": 12,
    "Year to Date": None,
}

CITY_COORDS = {
    "Jaipur": {"lat": 26.9124, "lon": 75.7873},
    "Udaipur": {"lat": 24.5854, "lon": 73.7125},
    "Jodhpur": {"lat": 26.2389, "lon": 73.0243},
    "Jaisalmer": {"lat": 26.9157, "lon": 70.9083},
    "Pushkar": {"lat": 26.4899, "lon": 74.5511},
}

st.set_page_config(
    page_title="Rajasthan Tourism Footfall Intelligence",
    page_icon="🕌",
    layout="wide",
)

# Global styling to deliver a polished, app-like feel
st.markdown(
    """
    <style>
        body {
            background-color: #05060a;
        }
        section.main > div {
            padding-top: 10px;
        }
        /* Typography */
        h1, h2, h3, h4, h5, h6, p, label, span, div, ul, li {
            color: rgba(255, 255, 255, 0.95) !important;
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        /* Inputs */
        .stTextInput input, .stSelectbox select, .stNumberInput input, .stDateInput input {
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.15);
        }
        /* Buttons */
        div[data-testid="stButton"] > button {
            border-radius: 12px;
            font-weight: 600;
            background: linear-gradient(90deg, #ff4b2b, #ff416c);
            border: none;
            color: white;
            padding: 0.75rem 1.2rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        div[data-testid="stButton"] > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(255, 65, 108, 0.35);
        }
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 16px;
            border-radius: 8px;
        }
        /* Metric cards */
        div[data-testid="stMetricValue"] {
            color: #e2e8f0 !important;
        }
        .metric-card {
            border-radius: 18px;
            padding: 18px 20px;
            background: linear-gradient(145deg, rgba(15,23,42,0.95), rgba(30,41,59,0.9));
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 18px 35px rgba(15, 23, 42, 0.45);
        }
        .result-card {
            border-radius: 18px;
            padding: 20px;
            background: rgba(15,23,42,0.8);
            border: 1px solid rgba(255,255,255,0.08);
            animation: fadeUp 0.4s ease forwards;
        }
        @keyframes fadeUp {
            from { opacity:0; transform: translateY(12px); }
            to { opacity:1; transform: translateY(0); }
        }
        .insight-chip {
            padding: 6px 12px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.2);
            margin-right: 8px;
            font-size: 0.85rem;
        }
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: #0b1220;
            border-right: 1px solid rgba(255,255,255,0.05);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_stat_card(title: str, value: str, delta: Optional[str] = None, icon: str = "📊"):
    delta_html = f"<span style='color:#22c55e'>{delta}</span>" if delta else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:0.9rem;opacity:0.8">{icon} {title}</div>
            <div style="font-size:2rem;font-weight:700;margin:4px 0">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )



@st.cache_data(show_spinner=False)
def load_processed_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "processed" / "merged_dataset.csv", parse_dates=["date"])
    return df


@st.cache_data(show_spinner=False)
def load_scaled_feature_store(feature_columns: List[str]) -> pd.DataFrame:
    processed_dir = DATA_DIR / "processed"
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    train_meta = pd.read_csv(processed_dir / "train_metadata.csv", parse_dates=["date"])
    test_meta = pd.read_csv(processed_dir / "test_metadata.csv", parse_dates=["date"])

    train_meta = train_meta.assign(split="train")
    test_meta = test_meta.assign(split="test")

    train_store = pd.concat([train_meta, X_train], axis=1)
    test_store = pd.concat([test_meta, X_test], axis=1)

    full_store = (
        pd.concat([train_store, test_store], ignore_index=True)
        .sort_values(["city", "date"])
        .reset_index(drop=True)
    )
    missing_cols = set(feature_columns) - set(full_store.columns)
    for col in missing_cols:
        full_store[col] = 0.0
    return full_store[["city", "date", "split"] + feature_columns]


@st.cache_data(show_spinner=False)
def load_results() -> Dict[str, pd.DataFrame]:
    comparison = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    rf_pred = pd.read_csv(RESULTS_DIR / "random_forest_predictions.csv")
    xgb_pred = pd.read_csv(RESULTS_DIR / "xgboost_predictions.csv")
    lstm_pred = pd.read_csv(RESULTS_DIR / "lstm_predictions.csv")
    return {
        "comparison": comparison,
        "rf": rf_pred,
        "xgb": xgb_pred,
        "lstm": lstm_pred,
    }


@st.cache_resource(show_spinner=False)
def load_prediction_assets():
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    feature_columns: List[str] = joblib.load(MODELS_DIR / "feature_columns.pkl")
    city_encoder: LabelEncoder = joblib.load(MODELS_DIR / "city_encoder.pkl")
    rf_model = joblib.load(MODELS_DIR / "random_forest_model.pkl")
    xgb_model = joblib.load(MODELS_DIR / "xgboost_model.pkl")
    
    # Try to load LSTM model, but make it optional
    lstm_model = None
    try:
        lstm_model = keras.models.load_model(MODELS_DIR / "lstm_model.h5")
    except Exception as e:
        st.warning("LSTM model could not be loaded. LSTM predictions will be unavailable.")
    
    return {
        "scaler": scaler,
        "features": feature_columns,
        "city_encoder": city_encoder,
        "rf_model": rf_model,
        "xgb_model": xgb_model,
        "lstm_model": lstm_model,
    }


@st.cache_data(show_spinner=False)
def load_events() -> pd.DataFrame:
    events_path = DATA_DIR / "raw" / "festivals_events.csv"
    if events_path.exists():
        return pd.read_csv(events_path)
    return pd.DataFrame(columns=["event_name", "city", "month", "duration_days"])


@st.cache_data(show_spinner=False)
def get_latest_city_state(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    latest = (
        df.sort_values("date")
        .groupby("city")
        .tail(1)
        .set_index("city")[LAG_FEATURES]
        .to_dict("index")
    )
    return latest


def get_env_credential(*keys: str) -> Optional[str]:
    """
    Resolve credentials in this order:
    1. Streamlit session state (allows runtime overrides via UI)
    2. Environment variables / .env
    """
    for key in keys:
        if key in st.session_state and st.session_state[key]:
            return st.session_state[key]
        value = os.getenv(key)
        if value:
            return value
    return None


def normalize_weather_condition(openweather_main: str) -> str:
    normalized = openweather_main.capitalize()
    if normalized not in WEATHER_OPTIONS:
        return "Clear"
    return normalized


@st.cache_data(show_spinner=False, ttl=900)
def fetch_live_weather(city: str) -> Dict[str, float]:
    api_key = get_env_credential("WEATHERAPI_KEY", "weatherapi_key", "OPENWEATHER_API_KEY")
    coords = CITY_COORDS.get(city)
    if not api_key or not coords:
        return {}

    # WeatherAPI.com uses "lat,lon" format for location
    location = f"{coords['lat']},{coords['lon']}"
    params = {
        "key": api_key,
        "q": location,
        "aqi": "no",
    }
    try:
        response = requests.get(WEATHERAPI_ENDPOINT, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        current = data.get("current", {})
        condition = current.get("condition", {})
        
        # WeatherAPI.com provides current temp, estimate min/max
        temp_c = float(current.get("temp_c", 0.0))
        feelslike_c = float(current.get("feelslike_c", temp_c))
        # Estimate min/max based on current temp and feels like
        temp_min = min(temp_c, feelslike_c) - 2.0
        temp_max = max(temp_c, feelslike_c) + 2.0
        
        # Convert pressure from mb to hPa (they're the same, but ensure consistency)
        pressure_mb = float(current.get("pressure_mb", 1010.0))
        
        # Convert wind from km/h to m/s
        wind_kph = float(current.get("wind_kph", 0.0))
        wind_ms = wind_kph / 3.6
        
        # Map WeatherAPI condition text to our weather options
        condition_text = condition.get("text", "Clear").lower()
        if "rain" in condition_text or "drizzle" in condition_text or "shower" in condition_text:
            weather_main = "Rain"
        elif "cloud" in condition_text or "overcast" in condition_text:
            weather_main = "Clouds"
        else:
            weather_main = "Clear"
        
        return {
            "temp_min": round(temp_min, 1),
            "temp_max": round(temp_max, 1),
            "humidity": float(current.get("humidity", 0.0)),
            "pressure": round(pressure_mb, 1),
            "wind_speed": round(wind_ms, 1),
            "weather_condition": normalize_weather_condition(weather_main),
        }
    except Exception as e:
        return {}


def request_groq_insight(
    *,
    city: str,
    target_date: date,
    rf_pred: float,
    xgb_pred: float,
    ensemble_pred: float,
    weather_condition: str,
    temp_min: float,
    temp_max: float,
    humidity: float,
    pressure: float,
    wind_speed: float,
    is_festival: bool,
) -> Optional[str]:
    api_key = get_env_credential("GROQ_API_KEY", "groq_api_key")
    if not api_key:
        return None

    prompt = (
        "You are an analyst for Rajasthan tourism. Provide a concise operational insight "
        "based on the predicted tourist demand and environment. "
        "Mention staffing/logistics tips in 2-3 sentences."
    )
    scenario_context = (
        f"City: {city}\n"
        f"Forecast date: {target_date.isoformat()}\n"
        f"Weather: {weather_condition}, temp {temp_min:.1f}°C - {temp_max:.1f}°C, "
        f"humidity {humidity:.0f}%, pressure {pressure:.0f} hPa, wind {wind_speed:.1f} m/s\n"
        f"Festival or major event: {'Yes' if is_festival else 'No'}\n"
        f"Random Forest prediction: {rf_pred:.0f}\n"
        f"XGBoost prediction: {xgb_pred:.0f}\n"
        f"Ensemble prediction: {ensemble_pred:.0f}\n"
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": scenario_context},
        ],
        "temperature": 0.2,
        "max_tokens": 250,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        
        # Extract content from response
        choices = data.get("choices", [])
        if not choices:
            return None
        
        message = choices[0].get("message", {})
        content = message.get("content", "").strip()
        
        if not content:
            return None
        return content
        
    except requests.exceptions.HTTPError as e:
        # HTTP error (4xx, 5xx)
        if hasattr(e, "response") and e.response is not None:
            try:
                error_data = e.response.json()
                error_msg = error_data.get("error", {}).get("message", str(e))
            except:
                error_msg = e.response.text[:200] if e.response.text else str(e)
        else:
            error_msg = str(e)
        # Return None - error details could be logged in production
        return None
    except requests.exceptions.RequestException as e:
        # Network/timeout errors
        return None
    except (KeyError, IndexError, ValueError) as e:
        # Response parsing errors
        return None
    except Exception as e:
        # Any other unexpected errors
        return None


def request_city_brief(city: str, df: pd.DataFrame) -> Optional[str]:
    api_key = get_env_credential("GROQ_API_KEY", "groq_api_key")
    if not api_key:
        return None

    city_df = df[df["city"] == city].copy().sort_values("date")
    if city_df.empty:
        return None

    avg = city_df["tourist_footfall"].mean()
    latest = city_df.iloc[-1]["tourist_footfall"]
    monthly = city_df.groupby(city_df["date"].dt.month)["tourist_footfall"].mean()
    peak_month = int(monthly.idxmax())
    lows = int(monthly.idxmin())
    volatility = city_df["tourist_footfall"].rolling(30).std().mean()

    prompt = (
        "You are an AI tourism strategist. Provide actionable recommendations (3 bullet points) "
        "covering demand outlook, operational readiness, and marketing focus for the city. "
        "Use concise business language."
    )
    context = (
        f"City: {city}\n"
        f"Average daily visitors: {avg:.0f}\n"
        f"Latest footfall: {latest:.0f}\n"
        f"Peak month (numeric): {peak_month}\n"
        f"Slow month (numeric): {lows}\n"
        f"Volatility index: {volatility:.2f}\n"
    )
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": context},
        ],
        "temperature": 0.3,
        "max_tokens": 220,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            return None
        return choices[0].get("message", {}).get("content", "").strip()
    except Exception:
        return None


def get_lstm_sequence(
    city: str,
    feature_store: pd.DataFrame,
    feature_columns: List[str],
    time_steps: int = TIME_STEPS,
) -> Optional[np.ndarray]:
    city_history = feature_store[feature_store["city"] == city].sort_values("date")
    if len(city_history) < time_steps:
        return None
    seq = city_history.tail(time_steps)[feature_columns].values
    return seq


def derive_season(selected_month: int) -> str:
    if selected_month in (11, 12, 1, 2):
        return "Winter"
    if selected_month in (3, 4, 5, 6):
        return "Summer"
    return "Monsoon"


def build_feature_payload(
    *,
    city: str,
    target_date: date,
    weather_condition: str,
    temp_min: float,
    temp_max: float,
    humidity: float,
    pressure: float,
    wind_speed: float,
    is_festival: bool,
    manual_lags: Dict[str, float],
    city_state: Dict[str, Dict[str, float]],
    feature_columns: List[str],
    city_encoder: LabelEncoder,
) -> pd.DataFrame:
    dt = datetime.combine(target_date, datetime.min.time())
    season = derive_season(dt.month)
    temp_avg = (temp_min + temp_max) / 2
    temp_range = temp_max - temp_min

    feature_values: Dict[str, float] = {col: 0.0 for col in feature_columns}
    feature_values.update(
        {
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "day_of_week": dt.weekday(),
            "is_weekend": int(dt.weekday() >= 5),
            "month_sin": np.sin(2 * np.pi * dt.month / 12),
            "month_cos": np.cos(2 * np.pi * dt.month / 12),
            "dow_sin": np.sin(2 * np.pi * dt.weekday() / 7),
            "dow_cos": np.cos(2 * np.pi * dt.weekday() / 7),
            "temp_avg": temp_avg,
            "temp_min": temp_min,
            "temp_max": temp_max,
            "temp_range": temp_range,
            "humidity": humidity,
            "pressure": pressure,
            "wind_speed": wind_speed,
            "is_festival": int(is_festival),
            "city_encoded": float(city_encoder.transform([city])[0]),
            f"weather_{weather_condition}": 1.0,
            f"season_{season}": 1.0,
        }
    )

    lag_source = manual_lags or city_state.get(city, {})
    for lag_col in LAG_FEATURES:
        feature_values[lag_col] = lag_source.get(lag_col, 0.0)

    feature_df = pd.DataFrame([feature_values])[feature_columns].astype(float)
    return feature_df


def render_overview_tab(df: pd.DataFrame, events_df: pd.DataFrame, default_city: str):
    st.subheader("Enhanced Data Overview")
    period = st.selectbox("Analysis period", list(PERIOD_OPTIONS.keys()), index=1, key="overview_period")
    months = PERIOD_OPTIONS[period]
    end_date = df["date"].max()
    start_date = end_date - pd.DateOffset(months=months) if months else df["date"].min()
    filtered = df[df["date"] >= start_date].copy()
    lookback = end_date - start_date
    prev_start = max(df["date"].min(), start_date - lookback) if months else start_date
    prev_df = df[(df["date"] >= prev_start) & (df["date"] < start_date)].copy()

    daily = filtered.groupby("date")["tourist_footfall"].sum().reset_index()
    daily["ma7"] = daily["tourist_footfall"].rolling(7).mean()
    daily["ma14"] = daily["tourist_footfall"].rolling(14).mean()
    daily["volatility"] = daily["tourist_footfall"].rolling(7).std()

    total_footfall = filtered["tourist_footfall"].sum()
    avg_daily = daily["tourist_footfall"].mean()
    latest_ma = daily["ma7"].iloc[-1] if not daily["ma7"].isna().all() else avg_daily
    latest_volatility = daily["volatility"].iloc[-1] if not daily["volatility"].isna().all() else 0
    prev_avg = prev_df.groupby("date")["tourist_footfall"].sum().mean() if not prev_df.empty else None
    growth_delta = (
        f"{(avg_daily - prev_avg) / prev_avg * 100:+.1f}% vs prior window"
        if prev_avg and prev_avg > 0
        else None
    )

    card_cols = st.columns(4)
    with card_cols[0]:
        render_stat_card("Total visitors", f"{total_footfall:,.0f}", icon="👥")
    with card_cols[1]:
        render_stat_card("Avg daily", f"{avg_daily:,.0f}", delta=growth_delta, icon="📈")
    with card_cols[2]:
        render_stat_card("7-day moving avg", f"{latest_ma:,.0f}", icon="🔁")
    with card_cols[3]:
        render_stat_card("Volatility (σ)", f"{latest_volatility:,.0f}", icon="⚠️")

    st.markdown("#### Interactive trend analysis")
    trend_view = st.radio(
        "Trend focus",
        ["Footfall volume", "Growth momentum"],
        horizontal=True,
        key="overview_trend_focus",
    )
    selected_city = st.selectbox(
        "Select city",
        sorted(df["city"].unique()),
        index=sorted(df["city"].unique()).index(default_city) if default_city in df["city"].unique() else 0,
        key="overview_city",
    )
    city_df = filtered[filtered["city"] == selected_city].sort_values("date")
    city_df["ma7"] = city_df["tourist_footfall"].rolling(7).mean()
    city_df["volatility"] = city_df["tourist_footfall"].rolling(7).std()
    if trend_view == "Footfall volume":
        trend_fig = go.Figure()
        trend_fig.add_trace(
            go.Scatter(
                x=city_df["date"],
                y=city_df["tourist_footfall"],
                mode="lines",
                name="Daily footfall",
                line=dict(color="#fb7185"),
            )
        )
        trend_fig.add_trace(
            go.Scatter(
                x=city_df["date"],
                y=city_df["ma7"],
                mode="lines",
                name="7-day MA",
                line=dict(color="#38bdf8", dash="dash"),
            )
        )
        trend_fig.update_layout(title=f"{selected_city} - Footfall & Moving Averages", yaxis_title="Visitors")
    else:
        city_df["growth_pct"] = city_df["tourist_footfall"].pct_change() * 100
        trend_fig = px.area(
            city_df,
            x="date",
            y="growth_pct",
            color_discrete_sequence=["#a78bfa"],
            title=f"{selected_city} - Growth momentum",
        )
        trend_fig.update_layout(yaxis_title="Growth %")
    st.plotly_chart(trend_fig, use_container_width=True)

    st.markdown("#### City performance statistics")
    city_stats = filtered.groupby("city")["tourist_footfall"].agg(["sum", "mean"]).rename(
        columns={"sum": "Total", "mean": "Avg daily"}
    )
    if not prev_df.empty:
        prev_city = prev_df.groupby("city")["tourist_footfall"].mean().rename("Prev avg")
        city_stats = city_stats.join(prev_city, how="left")
        city_stats["Growth %"] = ((city_stats["Avg daily"] - city_stats["Prev avg"]) / city_stats["Prev avg"]) * 100
    city_stats = city_stats.fillna(0).sort_values("Avg daily", ascending=False)
    st.dataframe(
        city_stats.style.format({"Total": "{:,.0f}", "Avg daily": "{:,.0f}", "Growth %": "{:+.1f}%"}),
        use_container_width=True,
    )

    st.markdown("#### Volatility dashboard")
    vol_fig = go.Figure()
    vol_fig.add_trace(
        go.Bar(
            x=city_stats.index,
            y=df[df["date"] >= start_date].groupby("city")["tourist_footfall"].std(),
            marker_color="#f97316",
            name="Std dev",
        )
    )
    vol_fig.update_layout(yaxis_title="Std dev", showlegend=False)
    st.plotly_chart(vol_fig, use_container_width=True)

    st.markdown("#### Upcoming cultural calendar")
    if events_df.empty:
        st.info("Festival calendar not available.")
    else:
        st.dataframe(events_df, use_container_width=True, hide_index=True)


def render_performance_tab(results: Dict[str, pd.DataFrame]):
    comparison = results["comparison"].copy()
    st.subheader("Advanced Model Performance")
    comparison["Confidence score"] = (
        (comparison["R²"].clip(lower=0, upper=1) * 0.6)
        + (1 / (1 + comparison["RMSE"] / 800)) * 0.25
        + (1 / (1 + comparison["MAE"] / 600)) * 0.15
    ) * 100
    st.dataframe(comparison.style.format({"RMSE": "{:.1f}", "MAE": "{:.1f}", "R²": "{:.3f}", "Confidence score": "{:.1f}"}))

    viz_option = st.selectbox(
        "Metric visualisation style",
        ["Grouped bar", "Radar", "Heatmap"],
        key="performance_chart",
    )
    metric_data = comparison.melt(id_vars="Model", value_vars=["RMSE", "MAE", "R²"], var_name="Metric", value_name="Value")
    if viz_option == "Grouped bar":
        metrics_fig = px.bar(
            metric_data,
            x="Model",
            y="Value",
            color="Metric",
            barmode="group",
            color_discrete_sequence=["#0ea5e9", "#22c55e", "#f97316"],
            title="Evaluation metrics overview",
        )
    elif viz_option == "Radar":
        radar_df = comparison.set_index("Model")
        metrics_fig = go.Figure()
        for model, row in radar_df.iterrows():
            metrics_fig.add_trace(
                go.Scatterpolar(
                    r=[row["RMSE"], row["MAE"], row["R²"] * 100],
                    theta=["RMSE", "MAE", "R² (x100)"],
                    fill="toself",
                    name=model,
                )
            )
        metrics_fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Radar comparison")
    else:
        pivot = metric_data.pivot(index="Metric", columns="Model", values="Value")
        metrics_fig = px.imshow(
            pivot,
            color_continuous_scale="Magma",
            labels=dict(color="Score"),
            title="Metric heatmap",
        )
    st.plotly_chart(metrics_fig, use_container_width=True)

    rf_pred = results["rf"]
    xgb_pred = results["xgb"]
    lstm_pred = results["lstm"]

    st.markdown("#### Error diagnostics")
    diag_cols = st.columns(2)
    model_choice = diag_cols[0].selectbox("Residual model", ["Random Forest", "XGBoost", "LSTM"], key="residual_model")
    pred_map = {
        "Random Forest": rf_pred,
        "XGBoost": xgb_pred,
        "LSTM": lstm_pred,
    }
    current = pred_map[model_choice].copy()
    # Calculate error if not present
    if "error" not in current.columns:
        current["error"] = current["y_true"] - current["y_pred"]
    residual_fig = go.Figure()
    residual_fig.add_trace(
        go.Histogram(
            x=current["error"],
            nbinsx=40,
            marker_color="#fb7185",
            name="Residuals",
        )
    )
    residual_fig.update_layout(title=f"{model_choice} residual distribution", xaxis_title="Error", yaxis_title="Frequency")
    diag_cols[0].plotly_chart(residual_fig, use_container_width=True)

    scatter_fig = px.scatter(
        current.sample(min(1000, len(current)), random_state=42),
        x="y_true",
        y="y_pred",
        trendline="ols",
        labels={"y_true": "Actual Footfall", "y_pred": "Predicted Footfall"},
        color_discrete_sequence=["#6366F1"],
        title=f"{model_choice}: Actual vs Predicted",
    )
    axis_min = min(current["y_true"].min(), current["y_pred"].min())
    axis_max = max(current["y_true"].max(), current["y_pred"].max())
    scatter_fig.add_trace(
        go.Scatter(
            x=[axis_min, axis_max],
            y=[axis_min, axis_max],
            mode="lines",
            line=dict(color="#94a3b8", dash="dash"),
            showlegend=False,
        )
    )
    diag_cols[1].plotly_chart(scatter_fig, use_container_width=True)

    st.markdown("#### Feature importance lab")
    feature_importance_path = RESULTS_DIR / "random_forest_feature_importance.csv"
    fi_chart_type = st.selectbox("Visualisation type", ["Horizontal bar", "Treemap", "Data table"], key="fi_chart")
    if feature_importance_path.exists():
        fi_df = pd.read_csv(feature_importance_path).sort_values("importance", ascending=False)
        if fi_chart_type == "Horizontal bar":
            fi_fig = px.bar(
                fi_df.head(20).sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="viridis",
                title="Random Forest - Feature importance",
            )
            fi_fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fi_fig, use_container_width=True)
        elif fi_chart_type == "Treemap":
            fi_fig = px.treemap(
                fi_df.head(25),
                path=["feature"],
                values="importance",
                color="importance",
                color_continuous_scale="Bluered",
                title="Importance treemap",
            )
            st.plotly_chart(fi_fig, use_container_width=True)
        else:
            st.dataframe(fi_df.head(30).style.format({"importance": "{:.4f}"}), use_container_width=True)
    else:
        st.info("Feature importance file not found.")


def render_prediction_tab(
    df: pd.DataFrame,
    city_state: Dict[str, Dict[str, float]],
    assets: Dict[str, object],
    feature_store: pd.DataFrame,
):
    st.subheader("Scenario Simulator")
    st.markdown(
        "Design future scenarios for Rajasthan's key cities. "
        "The simulator reuses the trained Random Forest and XGBoost models."
    )

    cities = sorted(df["city"].unique())
    city = st.selectbox("Destination city", cities)
    default_date = (df["date"].max() + pd.Timedelta(days=1)).date()
    target_date = st.date_input("Forecast date", value=default_date, min_value=date(2024, 11, 1))

    # Initialize default weather state
    default_weather_state = {
        "weather_condition_select": WEATHER_OPTIONS[0],
        "temp_min_input": 18.0,
        "temp_max_input": 32.0,
        "humidity_slider": 52,
        "pressure_input": 1010.0,
        "wind_speed_input": 3.5,
    }
    for key, default in default_weather_state.items():
        if key not in st.session_state:
            st.session_state[key] = default

    if "is_festival_flag" not in st.session_state:
        st.session_state["is_festival_flag"] = False

    # Check if weather data was loaded in previous run (before any widgets)
    if "pending_weather_data" in st.session_state and st.session_state["pending_weather_data"]:
        weather_data = st.session_state["pending_weather_data"]
        st.session_state["temp_min_input"] = weather_data.get("temp_min", st.session_state["temp_min_input"])
        st.session_state["temp_max_input"] = weather_data.get("temp_max", st.session_state["temp_max_input"])
        st.session_state["humidity_slider"] = weather_data.get("humidity", st.session_state["humidity_slider"])
        st.session_state["pressure_input"] = weather_data.get("pressure", st.session_state["pressure_input"])
        st.session_state["wind_speed_input"] = weather_data.get("wind_speed", st.session_state["wind_speed_input"])
        st.session_state["weather_condition_select"] = weather_data.get("weather_condition", st.session_state["weather_condition_select"])
        st.session_state["is_festival_flag"] = weather_data.get("is_festival", st.session_state["is_festival_flag"])
        # Clear pending data
        st.session_state["pending_weather_data"] = None

    col_weather, col_env, col_special = st.columns(3)

    # Button to load live weather
    load_weather_btn = col_weather.button("Load live weather", use_container_width=True, key="load_weather_btn")
    
    if load_weather_btn:
        live_weather = fetch_live_weather(city)
        if not live_weather:
            st.warning(
                "Unable to fetch live weather. Ensure WEATHERAPI_KEY is configured and valid."
            )
        else:
            # Store weather data to be applied on next run
            st.session_state["pending_weather_data"] = {
                "temp_min": round(live_weather["temp_min"], 1),
                "temp_max": round(live_weather["temp_max"], 1),
                "humidity": int(round(live_weather["humidity"])),
                "pressure": round(live_weather["pressure"], 1),
                "wind_speed": round(live_weather["wind_speed"], 1),
                "weather_condition": live_weather["weather_condition"],
                "is_festival": st.session_state["is_festival_flag"],
            }
            st.toast("Live weather data will be applied.")
            st.rerun()

    preset = col_weather.selectbox(
        "Quick scenario presets",
        ["Custom"] + list(SCENARIO_PRESETS.keys()),
        key="scenario_preset",
    )
    if preset != "Custom" and col_weather.button("Apply preset", key="apply_preset"):
        preset_data = SCENARIO_PRESETS[preset]
        st.session_state["pending_weather_data"] = {
            "temp_min": preset_data["temp_min"],
            "temp_max": preset_data["temp_max"],
            "humidity": preset_data["humidity"],
            "pressure": preset_data["pressure"],
            "wind_speed": preset_data["wind_speed"],
            "weather_condition": preset_data["weather_condition"],
            "is_festival": preset_data["is_festival"],
        }
        st.toast(f"{preset} scenario loaded")
        st.rerun()

    weather_condition = col_weather.selectbox(
        "Sky condition",
        WEATHER_OPTIONS,
        key="weather_condition_select",
    )
    temp_min = col_weather.number_input(
        "Min temperature (°C)",
        value=st.session_state["temp_min_input"],
        min_value=-5.0,
        max_value=50.0,
        key="temp_min_input",
    )
    temp_max = col_weather.number_input(
        "Max temperature (°C)",
        value=st.session_state["temp_max_input"],
        min_value=-5.0,
        max_value=55.0,
        key="temp_max_input",
    )

    humidity = col_env.slider(
        "Relative humidity (%)",
        min_value=10,
        max_value=100,
        value=st.session_state["humidity_slider"],
        key="humidity_slider",
    )
    pressure = col_env.number_input(
        "Pressure (hPa)",
        value=st.session_state["pressure_input"],
        min_value=900.0,
        max_value=1100.0,
        key="pressure_input",
    )
    wind_speed = col_env.number_input(
        "Wind speed (m/s)",
        value=st.session_state["wind_speed_input"],
        min_value=0.0,
        max_value=20.0,
        step=0.1,
        key="wind_speed_input",
    )

    is_festival = col_special.checkbox(
        "Festival / major event?",
        value=st.session_state.get("is_festival_flag", False),
        key="is_festival_flag",
    )
    manual_lag_mode = col_special.checkbox("Override lag features?", value=False)
    confidence_level = col_special.selectbox(
        "Confidence level",
        ["Standard", "High Precision", "Conservative"],
    )
    chart_style = col_special.selectbox(
        "Scenario theme",
        ["Executive Summary", "Operator Focus", "Risk Aware"],
    )

    manual_lags = {}
    if manual_lag_mode:
        st.markdown("Enter recent footfall statistics (leave defaults if unknown).")
        lag_cols = LAG_FEATURES
        lag_values = city_state.get(city, {col: 0.0 for col in lag_cols})
        col_lag1, col_lag2, col_lag3 = st.columns(3)
        manual_lags["footfall_lag_1"] = col_lag1.number_input(
            "Footfall lag 1 day", value=float(lag_values.get("footfall_lag_1", 0.0))
        )
        manual_lags["footfall_lag_7"] = col_lag2.number_input(
            "Footfall lag 7 days", value=float(lag_values.get("footfall_lag_7", 0.0))
        )
        manual_lags["footfall_lag_30"] = col_lag3.number_input(
            "Footfall lag 30 days", value=float(lag_values.get("footfall_lag_30", 0.0))
        )
        col_roll1, col_roll2 = st.columns(2)
        manual_lags["footfall_rolling_7"] = col_roll1.number_input(
            "Rolling 7-day average", value=float(lag_values.get("footfall_rolling_7", 0.0))
        )
        manual_lags["footfall_rolling_30"] = col_roll2.number_input(
            "Rolling 30-day average", value=float(lag_values.get("footfall_rolling_30", 0.0))
        )
    else:
        auto_lags = city_state.get(city)
        if auto_lags:
            st.success(
                "Using the latest historical lags from the feature store "
                f"(up to {df[df['city']==city]['date'].max().date()})."
            )
        else:
            st.warning("Lag features unavailable for this city. Consider entering them manually.")

    if st.button("🚀 Generate AI Forecast", type="primary", use_container_width=True):
        assets_loaded = assets
        feature_df = build_feature_payload(
            city=city,
            target_date=target_date,
            weather_condition=weather_condition,
            temp_min=temp_min,
            temp_max=temp_max,
            humidity=humidity,
            pressure=pressure,
            wind_speed=wind_speed,
            is_festival=is_festival,
            manual_lags=manual_lags if manual_lag_mode else {},
            city_state=city_state,
            feature_columns=assets_loaded["features"],
            city_encoder=assets_loaded["city_encoder"],
        )

        scaled = assets_loaded["scaler"].transform(feature_df)
        rf_pred = assets_loaded["rf_model"].predict(scaled)[0]
        xgb_pred = assets_loaded["xgb_model"].predict(scaled)[0]
        ensemble_pred = float(np.mean([rf_pred, xgb_pred]))
        confidence_map = {
            "Standard": 1.0,
            "High Precision": 0.95,
            "Conservative": 0.9,
        }
        confidence_pred = ensemble_pred * confidence_map[confidence_level]

        st.markdown("### Forecast Results")
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Random Forest", f"{rf_pred:,.0f} visitors")
        res_col2.metric("XGBoost", f"{xgb_pred:,.0f} visitors")
        res_col3.metric("Bi-model ensemble", f"{ensemble_pred:,.0f} visitors")

        confidence_indicator = min(0.99, confidence_pred / max(ensemble_pred, 1))
        st.metric(
            label=f"{confidence_level} forecast",
            value=f"{confidence_pred:,.0f} visitors",
            delta=f"{confidence_pred - ensemble_pred:+.0f}",
        )
        st.progress(confidence_indicator)
        st.markdown(
            f"""
            <div class="result-card">
                <h4 style="margin-top:0">{chart_style}</h4>
                <p style="opacity:0.85">
                    Based on current weather context and lag structure, the blended forecast expects 
                    <strong>{ensemble_pred:,.0f}</strong> visitors. Adjusting for the {confidence_level.lower()} setting, 
                    target operations for <strong>{confidence_pred:,.0f}</strong> guests. Weather outlook: {weather_condition} with
                    {humidity:.0f}% humidity and wind {wind_speed:.1f} m/s.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        theme_messages = {
            "Executive Summary": "Balanced view for leadership dashboards with emphasis on YoY movement.",
            "Operator Focus": "Highlights staffing and logistics readiness for operations teams.",
            "Risk Aware": "Adds cautionary context for compliance and crowd management teams.",
        }
        st.caption(theme_messages[chart_style])

        st.markdown("#### Feature Vector (scaled input sent to models)")
        preview = pd.DataFrame(scaled, columns=assets_loaded["features"]).T.rename(columns={0: "value"})
        st.dataframe(preview, use_container_width=True)

        # Check if Groq API key is available
        groq_api_key = get_env_credential("GROQ_API_KEY", "groq_api_key")
        if not groq_api_key:
            st.info("Groq API key not configured. AI insights will not be available.")
        else:
            with st.spinner("Generating AI insight..."):
                groq_insight = request_groq_insight(
                    city=city,
                    target_date=target_date,
                    rf_pred=rf_pred,
                    xgb_pred=xgb_pred,
                    ensemble_pred=ensemble_pred,
                    weather_condition=weather_condition,
                    temp_min=temp_min,
                    temp_max=temp_max,
                    humidity=humidity,
                    pressure=pressure,
                    wind_speed=wind_speed,
                    is_festival=is_festival,
                )
            if groq_insight:
                st.markdown("#### AI Insight (Groq)")
                st.write(groq_insight)
            else:
                st.warning(
                    "Unable to generate Groq insight. The API may be temporarily unavailable or there was an error. "
                    "Please check your API key and try again."
                )

    st.divider()
    st.markdown("#### LSTM Rolling-Window Forecast")
    st.caption(
        "Leverages the last 7 recorded days for the selected city to predict the next day's demand. "
        "Scenario overrides are not applied because the sequential model expects actual history."
    )
    latest_city_date = feature_store[feature_store["city"] == city]["date"].max()
    if pd.notna(latest_city_date):
        st.info(f"Latest available observation for {city}: {latest_city_date.date()}")

    if st.button("Predict via LSTM", key="lstm_forecast_btn"):
        if assets["lstm_model"] is None:
            st.error("LSTM model is not available. Please check the model file.")
        else:
            seq = get_lstm_sequence(city, feature_store, assets["features"])
            if seq is None:
                st.warning(f"Need at least {TIME_STEPS} historical days for {city} to use the LSTM model.")
            else:
                seq_input = np.expand_dims(seq, axis=0)
                lstm_pred = assets["lstm_model"].predict(seq_input, verbose=0)[0][0]
                st.metric("LSTM next-day prediction", f"{lstm_pred:,.0f} visitors")


def render_insights_tab(
    df: pd.DataFrame,
    results: Dict[str, pd.DataFrame],
    events_df: pd.DataFrame,
):
    st.subheader("Interactive City Analytics")
    city_col, period_col = st.columns([2, 1])
    selected_city = city_col.selectbox("Select destination", sorted(df["city"].unique()), key="insights_city")
    period_options = {
        "Last 3 Months": 3,
        "Last 6 Months": 6,
        "Last 12 Months": 12,
        "Full History": None,
    }
    selected_period = period_col.selectbox("Time period", list(period_options.keys()), index=1, key="insights_period")
    months = period_options[selected_period]
    city_df = df[df["city"] == selected_city].copy().sort_values("date")
    if months:
        cutoff = city_df["date"].max() - pd.DateOffset(months=months)
        city_df = city_df[city_df["date"] >= cutoff]
    city_chart = px.area(
        city_df,
        x="date",
        y="tourist_footfall",
        title=f"{selected_city} - Daily Footfall Trend",
        color_discrete_sequence=["#fb7185"],
    )
    city_chart.update_yaxes(title="Visitors")
    st.plotly_chart(city_chart, use_container_width=True)

    st.markdown("### Tourism Trend Intelligence")
    trend_cols = st.columns(2)
    analysis_period = trend_cols[0].selectbox("Analysis period", ["Last 6 Months", "Last 12 Months", "Full History"], key="trend_period")
    trend_metric = trend_cols[1].selectbox("Trend metric", ["Growth Rate", "Footfall Volume"], key="trend_metric")
    months_map = {"Last 6 Months": 6, "Last 12 Months": 12, "Full History": None}
    months_filter = months_map[analysis_period]
    monthly_df = df.copy()
    monthly_df["month"] = monthly_df["date"].dt.to_period("M").dt.to_timestamp()
    grouped = monthly_df.groupby(["month", "city"])["tourist_footfall"].sum().reset_index()
    grouped = grouped.sort_values("month")
    if trend_metric == "Growth Rate":
        grouped["value"] = grouped.groupby("city")["tourist_footfall"].pct_change() * 100
        y_title = "Growth %"
    else:
        grouped["value"] = grouped["tourist_footfall"]
        y_title = "Footfall"
    if months_filter:
        cutoff = grouped["month"].max() - pd.DateOffset(months=months_filter)
        grouped = grouped[grouped["month"] >= cutoff]
    trend_fig = px.line(
        grouped,
        x="month",
        y="value",
        color="city",
        markers=True,
        title="Monthly trend comparison",
    )
    trend_fig.update_layout(yaxis_title=y_title)
    st.plotly_chart(trend_fig, use_container_width=True)

    st.markdown("### Prediction Analysis Studio")
    model_map = {
        "Random Forest": results.get("rf"),
        "XGBoost": results.get("xgb"),
        "LSTM": results.get("lstm"),
    }
    analysis_cols = st.columns(2)
    selected_model = analysis_cols[0].selectbox("Select model for analysis", list(model_map.keys()))
    sample_size = analysis_cols[1].slider("Sample size", min_value=200, max_value=2000, value=800, step=100)
    model_df = model_map[selected_model].copy()
    model_sample = model_df.sample(min(sample_size, len(model_df)), random_state=42)
    analysis_fig = px.scatter(
        model_sample,
        x="y_true",
        y="y_pred",
        color_discrete_sequence=["#34d399"],
        labels={"y_true": "Actual", "y_pred": "Predicted"},
        title=f"{selected_model} - Actual vs Predicted",
    )
    analysis_fig.add_trace(
        go.Scatter(
            x=[model_sample["y_true"].min(), model_sample["y_true"].max()],
            y=[model_sample["y_true"].min(), model_sample["y_true"].max()],
            mode="lines",
            line=dict(color="#f87171", dash="dash"),
            showlegend=False,
        )
    )
    st.plotly_chart(analysis_fig, use_container_width=True)

    st.markdown("### Feature Spotlight")
    feature_cols = st.columns(2)
    top_n = feature_cols[0].slider("Top N features", min_value=5, max_value=25, value=15, step=1)
    chart_type = feature_cols[1].selectbox("Chart type", ["Horizontal Bar", "Heatmap"])
    fi_path = RESULTS_DIR / "random_forest_feature_importance.csv"
    if fi_path.exists():
        fi_df = pd.read_csv(fi_path).sort_values("importance", ascending=False).head(top_n)
        if chart_type == "Horizontal Bar":
            fi_fig = px.bar(
                fi_df.sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Bluered",
                title="Random Forest feature importance",
            )
            fi_fig.update_layout(yaxis_title="")
        else:
            heatmap_df = fi_df.set_index("feature").rename(columns={"importance": "score"})
            fi_fig = go.Figure(
                data=go.Heatmap(
                    z=heatmap_df["score"].values.reshape(-1, 1),
                    x=["Importance"],
                    y=heatmap_df.index,
                    colorscale="Tealrose",
                )
            )
            fi_fig.update_layout(title="Random Forest feature importance", yaxis_title="")
        st.plotly_chart(fi_fig, use_container_width=True)
    else:
        st.info("Feature importance file not available.")

    st.markdown("### Cultural Events Calendar")
    if events_df.empty:
        st.info("Festivals calendar not available.")
    else:
        event_city = st.selectbox("Filter by city", ["All"] + sorted(events_df["city"].unique()))
        event_data = events_df.copy()
        if event_city != "All":
            event_data = event_data[event_data["city"] == event_city]
        st.dataframe(event_data, use_container_width=True, hide_index=True)


def main():
    assets = load_prediction_assets()
    data_df = load_processed_data()
    events_df = load_events()
    results = load_results()
    city_state = get_latest_city_state(data_df)
    feature_store = load_scaled_feature_store(assets["features"])

    # Sidebar with quick controls and status
    with st.sidebar:
        st.markdown("### 🎛️ Quick Controls")
        st.markdown("---")

        st.markdown("#### 🔐 Configure API keys")
        st.caption("Keys entered here persist only for this session. Use .env for long-term storage.")
        if "groq_api_key_input" not in st.session_state:
            st.session_state["groq_api_key_input"] = get_env_credential("GROQ_API_KEY", "groq_api_key") or ""
        if "weather_api_key_input" not in st.session_state:
            st.session_state["weather_api_key_input"] = get_env_credential("WEATHERAPI_KEY", "weatherapi_key") or ""

        groq_input = st.text_input("Groq API key", type="password", key="groq_api_key_input")
        if groq_input.strip():
            st.session_state["GROQ_API_KEY"] = groq_input.strip()

        weather_input = st.text_input("Weather API key", type="password", key="weather_api_key_input")
        if weather_input.strip():
            st.session_state["WEATHERAPI_KEY"] = weather_input.strip()

        st.markdown("---")
        st.markdown("#### 📡 API Status")
        weather_key = get_env_credential("WEATHERAPI_KEY", "weatherapi_key")
        groq_key = get_env_credential("GROQ_API_KEY", "groq_api_key")
        weather_status = "🟢 Active" if weather_key else "🔴 Not Configured"
        groq_status = "🟢 Active" if groq_key else "🔴 Not Configured"
        st.markdown(f"**Weather API:** {weather_status}")
        st.markdown(f"**Groq AI:** {groq_status}")

        st.markdown("---")
        st.markdown("#### 📊 Quick Stats")
        st.metric("Total Records", f"{len(data_df):,}")
        st.metric("Cities", data_df["city"].nunique())
        st.metric("Date Range", f"{data_df['date'].min().date()} to {data_df['date'].max().date()}")

        st.markdown("---")
        st.markdown("#### 🎨 Display Options")
        show_animations = st.checkbox("Enable animations", value=True, key="animations")
        compact_mode = st.checkbox("Compact mode", value=False, key="compact")

    st.title("🕌 Rajasthan Tourist Footfall Intelligence Hub")
    st.caption(
        "Forecast demand, inspect model diagnostics, and explore Rajasthan's tourism dynamics "
        "through an integrated Streamlit experience."
    )

    tab_overview, tab_performance, tab_predict, tab_insights = st.tabs(
        ["Data Overview", "Model Performance", "Scenario Forecasting", "Insights Studio"]
    )

    with tab_overview:
        default_city = sorted(data_df["city"].unique())[0] if len(data_df["city"].unique()) > 0 else "Jaipur"
        render_overview_tab(data_df, events_df, default_city)

    with tab_performance:
        render_performance_tab(results)

    with tab_predict:
        render_prediction_tab(data_df, city_state, assets, feature_store)

    with tab_insights:
        render_insights_tab(data_df, results, events_df)


if __name__ == "__main__":
    main()

