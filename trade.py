import numpy as np
import pandas as pd
import yfinance as yf
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime

# Liste des tickers
tickers = [
    "AAPL", "NVDA", "MSFT", "AMZN", "AVGO", "0H59.IL", "META", "TSLA",
    "GOOGL", "MC.PA", "TTE", "SU.PA", "LORA.F", "TMUS", "CSCO", "ADBE",
    "AMD", "PEP", "TXN", "MRNA", "ARM", "SMCI", "ABNB", "PYPL", "INTU",
    "AIR.PA", "BN.PA","BTC-USD"
]

# Fonction pour calculer le RSI
def calculate_rsi(data, window=7):
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Fonction pour calculer le MACD
def calculate_macd(data, short_window=6, long_window=13, signal_window=9):
    ema_short = data["Close"].ewm(span=short_window, adjust=False).mean()
    ema_long = data["Close"].ewm(span=long_window, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

# Fonction pour récupérer les données avec yfinance
def get_data(tickers, period="6mo", interval="1d"):
    data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        df["SMA_20"] = df["Close"].rolling(window=5).mean()  # SMA 20 jours
        df["EMA_20"] = df["Close"].ewm(span=5, adjust=False).mean()  # EMA 20 jours
        df["RSI"] = calculate_rsi(df)  # RSI
        df["MACD"], df["Signal"], df["Histogram"] = calculate_macd(df)  # MACD
        data[ticker] = df
    return data

# Charger les données
data = get_data(tickers)

def predict_prices_with_prophet(df, periods=60):
    # Préparation des données pour Prophet
    df_prophet = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds']).dt.tz_localize(None)  # Supprimer le fuseau horaire

    # Initialisation et ajustement du modèle Prophet
    model = Prophet()
    model.fit(df_prophet)

    # Création d'un DataFrame pour les dates futures
    future_dates = model.make_future_dataframe(periods=periods)

    # Génération des prévisions
    forecast = model.predict(future_dates)

    # Extraction des dates et des prix prévus
    future_dates = forecast['ds']
    future_prices = forecast['yhat']

    return future_dates, future_prices

# Initialiser l'application Dash
app = dash.Dash(__name__)
app.title = "Tableau de bord des actions"

# Mise en page de l'application
app.layout = html.Div([
    html.H1("Tableau de bord des cours d'actions", style={"textAlign": "center"}),

    html.Div([
        html.Label("Choisissez une entreprise :", style={"fontWeight": "bold"}),
        dcc.Dropdown(
            id="ticker-dropdown",
            options=[{"label": ticker, "value": ticker} for ticker in tickers],
            value="AAPL",  # Valeur par défaut
            multi=False,
            style={"width": "50%"}
        )
    ], style={"textAlign": "center", "marginBottom": "20px"}),

    dcc.Graph(id="price-chart"),
    dcc.Graph(id="rsi-chart"),
    dcc.Graph(id="macd-chart"),
    dcc.Graph(id="prediction-chart")  # Nouveau graphique pour les prédictions
])

# Callback pour mettre à jour les graphiques
@app.callback(
    [
        Output("price-chart", "figure"),
        Output("rsi-chart", "figure"),
        Output("macd-chart", "figure"),
        Output("prediction-chart", "figure")  # Nouveau graphique pour les prédictions
    ],
    [Input("ticker-dropdown", "value")]
)
def update_charts(ticker):
    df = data[ticker]
    df = df.reset_index()

    # Prédictions avec Prophet
    n_steps = 60
    future_dates, future_prices = predict_prices_with_prophet(df, periods=n_steps)

    # Graphique des cours et moyennes mobiles
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(x=df['Date'], y=df["Close"], mode="lines", name="Cours", line=dict(color="blue")))
    price_fig.add_trace(go.Scatter(x=df['Date'], y=df["SMA_20"], mode="lines", name="SMA 20", line=dict(color="orange")))
    price_fig.add_trace(go.Scatter(x=df['Date'], y=df["EMA_20"], mode="lines", name="EMA 20", line=dict(color="green")))
    price_fig.update_layout(
        title=f"Cours et moyennes mobiles de {ticker}",
        xaxis_title="Date",
        yaxis_title="Prix ($)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Graphique du RSI
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df['Date'], y=df["RSI"], mode="lines", name="RSI", line=dict(color="purple")))
    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Suracheté", annotation_position="top left")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Survendu", annotation_position="bottom left")
    rsi_fig.update_layout(
        title=f"RSI (Relative Strength Index) de {ticker}",
        xaxis_title="Date",
        yaxis_title="RSI",
        template="plotly_white"
    )

    # Graphique du MACD
    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=df['Date'], y=df["MACD"], mode="lines", name="MACD", line=dict(color="blue")))
    macd_fig.add_trace(go.Scatter(x=df['Date'], y=df["Signal"], mode="lines", name="Signal", line=dict(color="orange")))
    macd_fig.add_trace(go.Bar(x=df['Date'], y=df["Histogram"], name="Histogram", marker_color="gray"))
    macd_fig.update_layout(
        title=f"MACD (Moving Average Convergence Divergence) de {ticker}",
        xaxis_title="Date",
        yaxis_title="Valeur",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Graphique des prédictions avec Prophet
    prediction_fig = go.Figure()
    prediction_fig.add_trace(go.Scatter(x=df['Date'], y=df["Close"], mode="lines", name="Cours historique", line=dict(color="blue")))
    prediction_fig.add_trace(go.Scatter(x=future_dates, y=future_prices, mode="lines", name="Prédictions", line=dict(color="red", dash="dash")))
    prediction_fig.update_layout(
        title=f"Prédictions avec Prophet pour {ticker}",
        xaxis_title="Date",
        yaxis_title="Prix ($)",
        template="plotly_white"
    )

    return price_fig, rsi_fig, macd_fig, prediction_fig

# Lancer le serveur Dash
if __name__ == "__main__":
    app.run_server(debug=True)