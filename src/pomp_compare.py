import os
import pickle
import datetime
import jax
import jax.numpy as jnp
from jax import random
import pandas as pd
import numpy as np
import pypomp
import pypomp.fit
import pypomp.pfilter
import pypomp.pomp_class
import global_1d as g

# Check 1: Testing rinit
params = jnp.array([jnp.log(0.0001), jnp.log(0.01), jnp.log(0.00015), jnp.log(0.001), 0.0, jnp.log(1e-6)])
J = 10  # Number of particles
rinit_states = g.rinit(params, J)

print("rinit states(pypomp):", rinit_states)

# Test dmeasure
state = jnp.array([1e-6, 1105, 0])  # Example state: [V, S, t]
y = 0.002  # Example observation
dmeasure_value = g.dmeasure(y, state, params)

print("dmeasure value(pypomp):", dmeasure_value)

#--------------------------------------------------------------

# Check 2: Removing noise from rprocess
def rproc_without_noise(state, params, key, covars=None):
    V, S, t = state
    mu, kappa, theta, xi, rho, V_0 = params
    mu = jnp.exp(mu)
    xi = jnp.exp(xi)
    rho = -1 + 2 / (1 + jnp.exp(-rho))
    t = t.astype(int)
    dWs = (covars[t] - mu + 0.5 * V) / jnp.sqrt(V)
    S = S + S * (mu + jnp.sqrt(jnp.maximum(V, 0.0)) * dWs)
    V = V + xi * jnp.sqrt(V) * dWs  # Remove dWv noise
    t += 1
    V = jnp.maximum(V, 1e-32)
    return jnp.array([V, S, t])

# Replaced rproc ver.
sp500_model_no_noise = pypomp.pomp_class.Pomp(
    rinit = g.rinit,
    rproc = rproc_without_noise,
    dmeas = g.dmeasure,
    ys = jnp.array(g.sp500['y'].values),
    theta = jnp.array(g.initial_params_df.iloc[0]),
    covars = jnp.insert(g.sp500['y'].values, 0, 0)
)

# Running pfilter with no-noise model
pf_out_no_noise = pypomp.pfilter.pfilter(
    pomp_object = sp500_model_no_noise,
    J = g.NP_EVAL,
    thresh = 0,
    key = random.PRNGKey(g.MAIN_SEED)
)

print("LL without noise(pypomp):", pf_out_no_noise)



