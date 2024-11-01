"""pypomp implementation of Weizhe's 1d_global_search.R code."""
import jax
import jax.numpy as jnp
from jax import random
import pandas as pd
import numpy as np
import datetime
import pypomp
import pypomp.fit
import pypomp.pomp_class
#import multiprocessing as mp

print("Current system time:", datetime.datetime.now())

ncores = 2
print(ncores)
RUN_LEVEL = 1
match RUN_LEVEL:
    case 1:
        NP_FITR = 2
        NFITR = 2
        NREPS_FITR = ncores
        NP_EVAL = 2
        NREPS_EVAL = ncores
        NREPS_EVAL2 = ncores
        print("Running at level 1")
    case 2:
        NP_FITR = 1000
        NFITR = 200
        NREPS_FITR = ncores
        NP_EVAL = 5000
        NREPS_EVAL = ncores
        NREPS_EVAL2 = ncores*8
        print("Running at level 2")

# Data Manipulation
sp500_raw = pd.read_csv("data/SPX.csv")
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

def rproc(state, params, key, covars = None):
    """Process simulator for Weizhe model."""
    V, S, t = state
    mu, kappa, theta, xi, rho, V_0 = params
    mu = jnp.exp(mu)
    xi = jnp.exp(xi)
    rho = -1 + 2/(1 + jnp.exp(-rho))
    t = t.astype(int)
    
    # Wiener process generation (Gaussian noise)
    dZ = random.normal(key)
    
    # Calculate dWs
    dWs = (covars[t] - mu + 0.5 * V) / jnp.sqrt(V)

    # dWv with correlation
    dWv = rho * dWs + jnp.sqrt(1 - rho ** 2) * dZ
    
    # Update state variables
    S = S + S * (mu + jnp.sqrt(jnp.maximum(V, 0.0)) * dWs)
    #S = S + S * (mu + jnp.sqrt(V) * dWs)
    V = V + xi * jnp.sqrt(V) * dWv
    t += 1
    
    # Feller condition to keep V positive
    V = jnp.maximum(V, 1e-32)
    
    return jnp.array([V, S, t])

# Initialization Model
def rinit(params, J, covars = None):
    """Initial state process simulator for Weizhe model."""
    V_0 = jnp.exp(params[5])
    S_0 = 1105  # Initial price
    t = 0
    return jnp.tile(jnp.array([V_0, S_0, t]), (J,1))

# Measurement model: how we measure state
def dmeasure(y, state, params):
    """Measurement model distribution for Weizhe model."""
    V, S, t = state
    mu = jnp.exp(params[0])
    return jax.scipy.stats.norm.logpdf(y, mu - 0.5 * V, jnp.sqrt(V))

initial_params = jnp.array([
    jnp.log(0.01), # mu
    jnp.log(0.1), # kappa
    jnp.log(0.05), # theta
    jnp.log(0.1), # xi
    jnp.log((1 + 0.5)/(1 - 0.5)), # rho 
    jnp.log(0.1) # V_0
])

# Initialize POMP model
sp500_model = pypomp.pomp_class.Pomp(
    rinit = rinit,
    rproc = rproc,
    dmeas = dmeasure,
    # Observed log returns
    ys = jnp.array(sp500['y'].values),
    # Initial parameters
    theta = initial_params,
    # Covariates(time)
    covars = jnp.insert(sp500['y'].values, 0, 0)
)

fit_out = pypomp.fit.fit(
    pomp_object = sp500_model,
    J = NP_FITR,
    Jh = 5,
    M = NREPS_FITR,
    a = 0.5,
    itns = 2,
    sigmas = 0.001,
    sigmas_init = 1e-20,
    mode = "IF2"
)

np.exp(fit_out[1][0])