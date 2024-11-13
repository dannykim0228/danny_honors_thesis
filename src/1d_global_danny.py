import jax
import jax.numpy as jnp
from jax import random, grad, jit
import pandas as pd
import numpy as np
from pypomp import Pomp
#import pypomp as pp
#import multiprocessing as mp

# Data Manipulation
sp500_raw = pd.read_csv("SPX.csv")
sp500 = sp500_raw.copy()

# Convert Date column: string -> date type
sp500['date'] = pd.to_datetime(sp500['Date'])

# Calculate difference in days and create 'time' column
sp500['diff_days'] = (sp500['date'] - sp500['date'].min()).dt.days
sp500['time'] = sp500['diff_days'].astype(float)

# Calculate log returns (y)
sp500['y'] = np.log(sp500['Close'] / sp500['Close'].shift(1))

# Drop missing values
sp500 = sp500.dropna(subset=['y'])[['time', 'y']]

# Process model: how state evolves over time
def process_model(state, covaryt, params, key):
    V, S = state
    mu, kappa, theta, xi, rho = params

    # Wiener process (random noise)
    dZ = random.normal(key)  
    dWs = (covaryt - mu + 0.5 * V) / jnp.sqrt(V)

    # dWv with correlation
    dWv = rho * dWs + jnp.sqrt(1 - rho ** 2) * dZ

    # Update state variables
    S = S + S * (mu + jnp.sqrt(jnp.maximum(V, 0.0)) * dWs)
    V = V + xi * jnp.sqrt(V) * dWv

    # Feller condition to keep V positive
    V = jnp.maximum(V, 1e-32)
    return V, S

# Initialization model: initial state setup
def rinit_model(params):
    V_0 = params["V_0"]
    # Initial price
    S_0 = 1105  
    return V_0, S_0

# Measurement model: how we measure state
def dmeasure(y, state, params):
    V, S = state
    mu = params["mu"]
    return jax.scipy.stats.norm.logpdf(y, mu - 0.5 * V, jnp.sqrt(V))

# Initial parameter values
initial_params = {
    "mu": 0.01, "kappa": 0.1, "theta": 0.05, "xi": 0.1, "rho": 0.5, "V_0": 0.1
}

# Initialize POMP model
sp500_model = Pomp(
    rinit=rinit_model,
    rproc=process_model,
    dmeas=dmeasure,

    # Observed log returns
    ys=sp500['y'].values,

    # Initial parameters
    theta=initial_params,

    # Covariates(time)
    covars=sp500['time'].values  
)