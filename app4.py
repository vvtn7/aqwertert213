import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
import plotly.graph_objs as go

# --- 1. Загрузка астрологических данных ---
@st.cache_data
def load_astro_data():
    df = pd.read_csv("astro_data.csv", parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    return df

# --- 2. Загрузка и расчёт финансовых данных ---
def stock_par(tick_nm, start_dt='2020-01-01', end_dt=datetime.today()):
    temptick = yf.Ticker(tick_nm)
    data = temptick.history(start=start_dt, end=end_dt, interval="1d")
    data = data.reset_index()
    data['Date'] = data['Date'].dt.tz_localize(None)
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
    data['cumulative_return'] = data['log_return'].fillna(0).cumsum().apply(np.exp)
    data['EMA_100'] = data['Close'].ewm(span=100, adjust=False).mean()
    return data

# --- 3. Интерфейс выбора тикера ---
st.title("🪐📈 Астрология и Рынки")
tick_nm = st.text_input("Введите тикер (например GC=F)", value="GC=F")

df_astro = load_astro_data()
df_stock = stock_par(tick_nm)

# --- 4. Временная шкала выбора даты ---
st.subheader("📆 Выберите дату (влияет на график и астроданные)")

min_date = df_stock['Date'].min().date()
max_date = df_stock['Date'].max().date()

slider_date = st.slider("📅 Дата", min_value=min_date, max_value=max_date, value=max_date)
st.session_state["selected_date"] = slider_date

selected_date = st.session_state["selected_date"]
st.markdown(f"**📅 Выбранная дата:** `{selected_date}`")

# --- 5. График цены и EMA + подсветка выбранной точки ---
st.subheader("📊 График цены и EMA с подсветкой даты")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_stock['Date'], y=df_stock['Close'],
    mode='lines+markers', name='Цена'
))
fig.add_trace(go.Scatter(
    x=df_stock['Date'], y=df_stock['EMA_100'],
    mode='lines', name='EMA 100'
))

# Подсветка выбранной даты
highlight_point = df_stock[df_stock['Date'].dt.date == selected_date]
if not highlight_point.empty:
    fig.add_trace(go.Scatter(
        x=highlight_point['Date'],
        y=highlight_point['Close'],
        mode='markers',
        marker=dict(color='red', size=10, symbol='star'),
        name='Выбранная дата'
    ))

fig.update_layout(
    title=f"{tick_nm}: Цена и EMA 100",
    xaxis_title="Дата",
    yaxis_title="Цена",
    hovermode="x unified",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# --- 6. Поиск данных на выбранную дату ---
date_str = selected_date.strftime("%Y-%m-%d")
astro_row = df_astro.loc[df_astro.index.strftime("%Y-%m-%d") == date_str]

# --- 7. Аспекты и ретроградность ---
aspects_allowed = {"Conjunction", "Sextile", "Square", "Trine", "Opposition"}

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔗 Аспекты (без Chiron)")
    if not astro_row.empty:
        row = astro_row.iloc[0]
        aspect_cols = [col for col in df_astro.columns if col.endswith("_x") and "Chiron" not in col]
        filtered_aspects = row[aspect_cols].dropna()
        filtered_aspects = filtered_aspects[filtered_aspects.isin(aspects_allowed)]

        if not filtered_aspects.empty:
            st.dataframe(filtered_aspects.to_frame().style.applymap(lambda v: "background-color: lightblue"))
        else:
            st.info("Нет подходящих аспектов.")
    else:
        st.warning("Дата отсутствует в астрологических данных.")

with col2:
    st.subheader("🌀 Ретроградные / Стационарные планеты")
    if not astro_row.empty:
        status_cols = [col for col in df_astro.columns if col.endswith("_status")]
        status_today = astro_row.iloc[0][status_cols]
        filtered_status = status_today[status_today.isin(["Stationary", "Retrograde"])]

        if not filtered_status.empty:
            st.dataframe(filtered_status.to_frame().style.applymap(
                lambda v: "background-color: orange" if v == "Stationary" else "background-color: lightpink"
            ))
        else:
            st.info("Нет ретроградных или стационарных планет.")
    else:
        st.warning("Нет данных на эту дату.")

# --- 8. Астрологический круг ---
df_planets = pd.read_csv('df_planets.csv')
df_planets = df_planets.set_index(['Date'])

def get_planet_positions(date, df_planets):
    date = pd.to_datetime(date).normalize()
    df_planets.index = pd.to_datetime(df_planets.index).normalize()
    if date not in df_planets.index:
        return {planet.replace('_lon', ''): None for planet in df_planets.columns if planet.endswith('_lon')}

    row = df_planets.loc[date]
    positions = {col.replace('_lon', ''): row[col] for col in df_planets.columns if col.endswith('_lon')}
    return positions

st.subheader("🧭 Положение планет на круге")

positions = get_planet_positions(pd.to_datetime(selected_date), df_planets)
fig2, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi / 2)
ax.set_yticklabels([])

for planet, angle in positions.items():
    if angle is not None:
        theta = np.deg2rad(angle)
        ax.plot([theta], [1], 'o', label=planet)
        ax.text(theta, 1.1, planet, ha='center', va='center', fontsize=9)

ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.2))
st.pyplot(fig2)

