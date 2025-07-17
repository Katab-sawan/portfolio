import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
import plotly.graph_objects as go

st.set_page_config(page_title="Implied Volatility Surface", page_icon=":chart_with_upwards_trend:", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #D3D3D3;
    }
    .css-1d391kg {
        background-color: #0E1117;
    }
    </style>
    """, unsafe_allow_html=True)

st.title('Implied Volatility Surface & Greeks')

def bs_call_price(S, K, T, r, sigma, q=0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_greeks(S, K, T, r, sigma, q=0):
    if T <= 0:
        return {'delta': np.nan, 'gamma': np.nan, 'vega': np.nan, 'theta': np.nan, 'rho': np.nan}
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = np.exp(-q * T) * norm.cdf(d1)
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100
    theta = (-S * norm.pdf(d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    theta /= 365
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}

def implied_volatility(price, S, K, T, r, q=0):
    if T <= 0 or price <= 0:
        return np.nan
    def objective(sigma):
        return bs_call_price(S, K, T, r, sigma, q) - price
    try:
        return brentq(objective, 1e-6, 5)
    except:
        return np.nan

st.sidebar.header('Model Parameters')
r = st.sidebar.number_input('Risk-Free Rate', value=0.015, format="%.4f")
q = st.sidebar.number_input('Dividend Yield', value=0.013, format="%.4f")
y_axis_option = st.sidebar.selectbox('Select Y-axis:', ('Strike Price ($)', 'Moneyness'))
ticker_symbol = st.sidebar.text_input('Enter Ticker Symbol', value='SPY').upper()
min_strike_pct = st.sidebar.number_input('Min Strike % of Spot', 50.0, 199.0, 80.0)
max_strike_pct = st.sidebar.number_input('Max Strike % of Spot', 51.0, 200.0, 120.0)

if min_strike_pct >= max_strike_pct:
    st.sidebar.error('Min % must be < Max %')
    st.stop()

try:
    ticker = yf.Ticker(ticker_symbol)
    expirations = ticker.options
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

today = pd.Timestamp.today().normalize()
exp_dates = [pd.Timestamp(e) for e in expirations if pd.Timestamp(e) > today + timedelta(days=7)]

if not exp_dates:
    st.error('No valid expirations.')
    st.stop()

option_data = []
for exp in exp_dates:
    try:
        calls = ticker.option_chain(exp.strftime('%Y-%m-%d')).calls
        calls = calls[(calls['bid'] > 0) & (calls['ask'] > 0)]
        for _, row in calls.iterrows():
            mid = (row['bid'] + row['ask']) / 2
            option_data.append({
                'expirationDate': exp,
                'strike': row['strike'],
                'mid': mid
            })
    except:
        continue

if not option_data:
    st.error('No valid call options.')
    st.stop()

options_df = pd.DataFrame(option_data)
spot_price = ticker.history(period='5d')['Close'].iloc[-1]
options_df['daysToExpiration'] = (options_df['expirationDate'] - today).dt.days
options_df['timeToExpiration'] = options_df['daysToExpiration'] / 365
options_df = options_df[
    (options_df['strike'] >= spot_price * (min_strike_pct / 100)) &
    (options_df['strike'] <= spot_price * (max_strike_pct / 100))
]

options_df['iv'] = options_df.apply(lambda row: implied_volatility(
    row['mid'], spot_price, row['strike'], row['timeToExpiration'], r, q), axis=1)

options_df.dropna(subset=['iv'], inplace=True)
options_df['iv'] *= 100
options_df['moneyness'] = options_df['strike'] / spot_price

options_df[['delta', 'gamma', 'vega', 'theta', 'rho']] = options_df.apply(
    lambda row: pd.Series(bs_greeks(spot_price, row['strike'], row['timeToExpiration'], r, row['iv'] / 100, q)),
    axis=1
)

display_df = options_df.copy()
Y = display_df['strike'].values if y_axis_option == 'Strike Price ($)' else display_df['moneyness'].values
y_label = 'Strike Price ($)' if y_axis_option == 'Strike Price ($)' else 'Moneyness'
X = display_df['timeToExpiration'].values
Z = display_df['iv'].values

T, K = np.meshgrid(np.linspace(X.min(), X.max(), 50), np.linspace(Y.min(), Y.max(), 50))
Zi = griddata((X, Y), Z, (T, K), method='linear')
Zi = np.ma.array(Zi, mask=np.isnan(Zi))

fig = go.Figure(data=[go.Surface(x=T, y=K, z=Zi, colorscale='Viridis', colorbar_title='IV (%)')])
fig.update_layout(title=f'Implied Volatility Surface - {ticker_symbol}',
                  scene=dict(xaxis_title='Time to Expiry (Years)',
                             yaxis_title=y_label,
                             zaxis_title='IV (%)'),
                  width=900, height=800)
st.plotly_chart(fig)

# 3D Greeks
greek_names = ['delta', 'gamma', 'vega', 'theta', 'rho']
with st.spinner("Interpolating Greeks..."):
    for greek in greek_names:
        G = display_df[greek].values
        Gi = griddata((X, Y), G, (T, K), method='linear')
        Gi = np.ma.array(Gi, mask=np.isnan(Gi))

        fig_greek = go.Figure(data=[go.Surface(x=T, y=K, z=Gi, colorscale='Plasma', colorbar_title=greek.title())])
        fig_greek.update_layout(title=f'{greek.title()} Surface - {ticker_symbol}',
                                scene=dict(xaxis_title='Time to Expiry (Years)',
                                           yaxis_title=y_label,
                                           zaxis_title=greek.title()),
                                width=900, height=800)
        st.plotly_chart(fig_greek)

st.write("---")
st.markdown("Created by Katab Sawan  |  [LinkedIn](https://www.linkedin.com/in/katab-sawan-60a2a8260/)  |  [GitHub](https://github.com/Katab-sawan/portfolio/tree/main/personal_projects)")
