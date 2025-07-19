import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.sparse import csc_matrix
import math

st.title("Options Pricing App")

# Inputs
st.sidebar.header("Parameters")
S = st.sidebar.number_input("Spot Price (S)", value=100.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0)
T = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0)
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05)
sigma = st.sidebar.number_input("Volatility (sigma)", value=0.2)
N = st.sidebar.slider("Steps in Binomial Tree", min_value=3, max_value=50, value=5)

# Black-Scholes for European Options
def black_scholes_call_put(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    put = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call, put

call_price, put_price = black_scholes_call_put(S, K, T, r, sigma)
st.subheader("Black-Scholes European Option Pricing")
st.write(f"Call Price: **{call_price:.4f}**")
st.write(f"Put Price: **{put_price:.4f}**")

# Binomial model for American Call Option
def binomial_tree_american_call(S, K, T, r, sigma, N):
    dt = T / (N - 1)
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    stock_prices = csc_matrix((N, N))
    call_prices = csc_matrix((N, N))

    stock_prices[0, 0] = S
    for i in range(1, N):
        M = i + 1
        stock_prices[i, 0] = d * stock_prices[i - 1, 0]
        for j in range(1, M):
            stock_prices[i, j] = u * stock_prices[i - 1, j - 1]

    # At expiration
    expiration = stock_prices[:, -1].toarray() - K
    expiration = np.where(expiration >= 0, expiration, 0)
    call_prices[-1, :] = expiration

    # Backward induction
    for i in range(N - 2, -1, -1):
        for j in range(i + 1):
            call_prices[i, j] = np.exp(-r * dt) * (
                (1 - p) * call_prices[i + 1, j] + p * call_prices[i + 1, j + 1]
            )

    return call_prices[0, 0], call_prices

binomial_price, call_matrix = binomial_tree_american_call(S, K, T, r, sigma, N)
st.subheader("Binomial Model American Call Option")
st.write(f"Price: **{binomial_price:.4f}**")

# Visualize
fig, ax = plt.subplots()
ax.spy(call_matrix, markersize=5)
st.subheader("Call Option Pricing Matrix (Binomial)")
st.pyplot(fig)
