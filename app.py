%%writefile app.py
import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title('📈 S&P 500 Interactive Dashboard')

st.markdown("""
This app retrieves the list of the *S&P 500* (from Wikipedia) and its corresponding *stock data*! 📊

*Python libraries:* base64, pandas, streamlit, numpy, matplotlib, seaborn, plotly, sklearn, yfinance

*Data source:* [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
""")

st.sidebar.header('🔧 User Input Features')

@st.cache_data
def load_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header=0)
    df = html[0]
    return df

df = load_data()

sorted_sector_unique = sorted(df['GICS Sector'].unique())
selected_sector = st.sidebar.multiselect('Select Sector(s)', sorted_sector_unique, sorted_sector_unique)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
num_company = st.sidebar.slider('Number of Companies to Display', 1, 10, 5)

show_ma = st.sidebar.checkbox("Show Moving Averages", True)
show_volume = st.sidebar.checkbox("Show Volume Chart", True)
show_prediction = st.sidebar.checkbox("Show 7-Day Price Prediction", True)

filtered_df = df[df['GICS Sector'].isin(selected_sector)]

if filtered_df.empty:
    st.error("🚫 No companies found for the selected sector(s). Please choose at least one sector.")
    st.stop()

if start_date >= end_date:
    st.error("🚫 Invalid date range. End Date must be after Start Date.")
    st.stop()

tickers = list(filtered_df.Symbol.unique())[:num_company]

st.subheader('📋 Companies in Selected Sectors')
st.write(f"{filtered_df.shape[0]} companies found.")
st.dataframe(filtered_df)

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">📥 Download CSV File</a>'
    return href

st.markdown(filedownload(filtered_df), unsafe_allow_html=True)

# Download stock data
data = yf.download(
    tickers=tickers,
    start=start_date,
    end=end_date,
    interval="1d",
    group_by='ticker',
    auto_adjust=True,
    threads=True
)

st.subheader('📉 Stock Dashboard')

def candlestick_chart(symbol):
    df = data[symbol].copy().reset_index()
    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])
    fig.update_layout(title=f"{symbol} Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

def moving_averages(symbol):
    df = data[symbol]['Close'].dropna().copy()
    df = df.to_frame().reset_index()
    df['MA7'] = df['Close'].rolling(window=7).mean()
    # df['MA14'] = df['Close'].rolling(window=14).mean()
    # df['MA30'] = df['Close'].rolling(window=30).mean()
    fig = px.line(df, x='Date', y=['Close', 'MA7'], title=f'{symbol} - Moving Averages')
    st.plotly_chart(fig)

def volume_chart(symbol):
    df = data[symbol].copy().reset_index()
    fig = px.area(df, x='Date', y='Volume', title=f'{symbol} - Trading Volume', template="plotly_dark")
    st.plotly_chart(fig)

def daily_returns(symbol):
    df = data[symbol]['Close'].dropna().pct_change().dropna()
    fig = px.line(df, title=f'{symbol} - Daily Returns', labels={'value': 'Return'}, template="plotly_white")
    st.plotly_chart(fig)

def risk_metrics(symbol):
    df = pd.DataFrame(data[symbol]['Close']).dropna()
    df['Return'] = df['Close'].pct_change()
    volatility = np.std(df['Return']) * np.sqrt(252)
    sharpe_ratio = np.mean(df['Return']) / np.std(df['Return']) * np.sqrt(252)
    st.markdown(f"*Volatility (Annualized):* {volatility:.2%}")
    st.markdown(f"*Sharpe Ratio:* {sharpe_ratio:.2f}")

def summary_table(symbol):
    df = data[symbol]['Close'].dropna()
    st.dataframe(df.describe().T)

def correlation_heatmap(tickers_list):
    df_close = pd.DataFrame({sym: data[sym]['Close'] for sym in tickers_list if 'Close' in data[sym]})
    df_returns = df_close.pct_change().dropna()
    corr = df_returns.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    plt.title("Correlation Heatmap of Daily Returns")
    st.pyplot(fig)

def predict_price(symbol):
    df = pd.DataFrame(data[symbol]['Close']).dropna().reset_index()
    df['Day'] = range(len(df))
    X = df[['Day']]
    y = df['Close']
    model = LinearRegression()
    model.fit(X, y)
    future_days = pd.DataFrame({'Day': range(len(df), len(df)+7)})
    predictions = model.predict(future_days)
    st.markdown("📈 Predicted Prices for Next 7 Days**")
    st.line_chart(np.append(y.values, predictions))

if st.button('📊 Generate Stock Dashboard'):
    for symbol in tickers:
        st.markdown(f"## 📌 {symbol}")
        candlestick_chart(symbol)
        if show_ma:
            moving_averages(symbol)
        if show_volume:
            volume_chart(symbol)
        daily_returns(symbol)
        risk_metrics(symbol)
        summary_table(symbol)
        if show_prediction:
            predict_price(symbol)
        st.markdown("---")

    correlation_heatmap(tickers)
    # sector_avg_returns(filtered_df, tickers)

    st.info("⚠ For educational purposes only. Predictions are based on linear regression and may not reflect actual market conditions.")
