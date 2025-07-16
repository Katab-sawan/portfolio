print("Hello, this is a placeholder for the Options Pricing project which will theoretically price Eurpoean options using Blach-Scholes and American options using the Binomial method.")
from scipy.stats import norm
from enum import Enum
from abc import ABC, abstractmethod
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.stats import bernoulli

# Black-Scholes Model for European Options

class OPTION_TYPE(Enum):
    CALL = "call"
    PUT = "PUT"

class EuropeanOptionModel(ABC):

    def calculate_optio_price(self, option_type):
        if option_type == OPTION_TYPE.CALL:
            return self.calclulate_call_price()
        elif option_type == OPTION_TYPE.PUT:
            return self.calculate_put_price()
        else:
            return -1
    @classmethod
    @abstractmethod
    def calculate_call_option_price(cls):
        pass

    @classmethod
    @abstractmethod
    def calculate_put_option_price(cls):
        pass

class BlackScholesModel(EuropeanOptionModel):
    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def calculate_call_option_price(self):
        d1 = (math.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * math.sqrt(self.T))
        d2 = d1 - self.sigma * math.sqrt(self.T)
        return (self.S * norm.cdf(d1) - self.K * math.exp(-self.r * self.T) * norm.cdf(d2))

    def calculate_put_option_price(self):
        d1 = (math.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * math.sqrt(self.T))
        d2 = d1 - self.sigma * math.sqrt(self.T)
        return (self.K * math.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1))
    
model = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2)
call = model.calculate_call_option_price()
put = model.calculate_put_option_price()
print(f"Call Option Price: {call}")
print(f"Put Option Price: {put}")
    
# Binomial Model for American Options

S0 = 100
t = 1
u = 1.05
d = 1/u
p = 0.6

S_u = u * S0
S_d = d * S0
K = 102.5
r = 0.01

C_u = S_d - K
C_d = 0

print(S_u, S_d)

C = np.exp(-r * t) * (p*C_u + (1-p)*C_d)
print(C)
        


N = 5

t = 1
t = t / (N-1)

S0 = 100
r = 0.01

sigma = 0.04
u = np.exp(sigma * np.sqrt(t))
u = 1/d
p = (np.exp(r*t) - d) / (u - d)

stock_prices = csc_matrix((N,N))
call_prices = csc_matrix((N,N))

stock_prices[0,0] = S0

for i in range(1,N):
  M = i + 1
  stock_prices[i,0] = d * stock_prices[i-1,0]
  for j in range(1,M):
    stock_prices[i,j] = u * stock_prices[i-1,j-1]

expiration = stock_prices[:,-1].toarray() - K
expiration_shape = (expiration.size, )
expiration = np.where(expiration >= 0, expiration, 0)

call_prices[-1,:] = expiration

for i in range(N - 2, -1, -1):
  for j in range(i+1):
    call_prices[i,j] = np.exp(-r * t) * ((1-p) * call_prices[i+1,j] + p * call_prices[i+1,j+1])


plt.spy(call_prices)
print(call_prices[0,0])