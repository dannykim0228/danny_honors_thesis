import jax
import jax.numpy as jnp
from jax import random, grad, jit
import pandas as pd
import numpy as np
import multiprocessing as mp

# Data Manipulation
sp500_raw = pd.read_csv("../data/SPX.csv")
sp500 = sp500_raw.copy()
sp500['date'] = pd.to_datetime(sp500['Date'])
sp500['diff_days'] = (sp500['date'] - sp500['date'].min()).dt.days
sp500['time'] = sp500['diff_days'].astype(float)
sp500['y'] = np.log(sp500['Close'] / sp500['Close'].shift(1))
sp500 = sp500.dropna(subset=['y'])[['time', 'y']]

# Name of States and Parmeters
sp500_statenames = ["V", "S"]
sp500_rp_names = ["mu", "kappa", "theta", "xi", "rho"]
sp500_ivp_names = ["V_0"]
sp500_parameters = sp500_rp_names + sp500_ivp_names
sp500_covarnames = ["covaryt"]

def rproc(state, covaryt, params, key):
    V, S = state
    mu, kappa, theta, xi, rho = params
    
    # Wiener process generation (Gaussian noise)
    dZ = random.normal(key)
    
    # Calculate dWs
    dWs = (covaryt - mu + 0.5 * V) / jnp.sqrt(V)
    
    # dWv with correlation
    dWv = rho * dWs + jnp.sqrt(1 - rho ** 2) * dZ
    
    # Update state variables
    S = S + S * (mu + jnp.sqrt(jnp.maximum(V, 0.0)) * dWs)
    V = V + xi * jnp.sqrt(V) * dWv
    
    # Feller condition to keep V positive
    V = jnp.maximum(V, 1e-32)
    
    return V, S

# Initialization Model
def rinit(params):
    V_0 = params["V_0"]
    S_0 = 1105  # Initial price
    return V_0, S_0

