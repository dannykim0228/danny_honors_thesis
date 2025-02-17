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

print("Current system time:", datetime.datetime.now())

out_dir = os.environ.get("out_dir")
if out_dir is None:
    SAVE_RESULTS_TO = "output/default_output/1d_global_mop_out.pkl"
else:
    SAVE_RESULTS_TO = out_dir + "1d_global_mop_out.pkl"

MAIN_SEED = 631409
np.random.seed(MAIN_SEED)
RUN_LEVEL = 3
match RUN_LEVEL:
    case 1:
        NP_FITR = 2
        #NFITR = 2
        NREPS_FITR = 3
        NP_EVAL = 2
        NREPS_EVAL = 5
        print("Running at level 1")
    case 2:
        NP_FITR = 1000
        #NFITR = 20
        NREPS_FITR = 3
        NP_EVAL = 1000
        NREPS_EVAL = 5
        print("Running at level 2")
    case 3:
        NP_FITR = 1000
        #NFITR = 100
        NREPS_FITR = 20
        NP_EVAL = 5000
        NREPS_EVAL = 20
        print("Running at level 3")

# Data Manipulation
sp500_raw = pd.read_csv("C:/Users/ravis/OneDrive/Documents/danny_honors_thesis/data/SPX.csv")
#sp500_raw = pd.read_csv("/home/kimdanny/danny_honors_thesis/data/SPX.csv")
sp500 = sp500_raw.copy()
sp500['date'] = pd.to_datetime(sp500['Date'])
sp500['diff_days'] = (sp500['date'] - sp500['date'].min()).dt.days
sp500['time'] = sp500['diff_days'].astype(float)
sp500['y'] = np.log(sp500['Close'] / sp500['Close'].shift(1))
sp500 = sp500.dropna(subset = ['y'])[['time', 'y']]

# Name of States and Parmeters
sp500_statenames = ["V", "S"]
sp500_rp_names = ["mu", "kappa", "theta", "xi", "rho"]
sp500_ivp_names = ["V_0"]
sp500_parameters = sp500_rp_names + sp500_ivp_names
sp500_covarnames = ["covaryt"]

def rproc(state, params, key, covars = None):
    V, S, t = state
    mu, kappa, theta, xi, rho, V_0 = params
    mu, kappa, theta, xi = jnp.exp(mu), jnp.exp(kappa), jnp.exp(theta), jnp.exp(xi)
    rho = -1 + 2 / (1 + jnp.exp(-rho))
    t = t.astype(int)
    dZ = random.normal(key)
    dWs = (covars[t] - mu + 0.5 * V) / jnp.sqrt(V)
    dWv = rho * dWs + jnp.sqrt(1 - rho ** 2) * dZ
    S = S + S * (mu + jnp.sqrt(jnp.maximum(V, 0.0)) * dWs)
    V = V + kappa * (theta - V) + xi * jnp.sqrt(jnp.maximum(V, 0.0)) * dWv
    t += 1
    V = jnp.maximum(V, 1e-32)
    return jnp.array([V, S, t])

def rinit(params, J, covars = None):
    V_0 = jnp.exp(params[5])
    #V_0 = jnp.exp(jnp.clip(params[5], a_min=-10, a_max=0))
    S_0, t = 1105, 0
    return jnp.tile(jnp.array([V_0, S_0, t]), (J, 1))

def dmeasure(y, state, params):
    V, S, t = state
    mu = jnp.exp(params[0])
    return jax.scipy.stats.norm.logpdf(y, mu - 0.5 * V, jnp.sqrt(V))

def funky_transform(lst):
    return [np.log((1 + x) / (1 - x)) for x in lst]

sp500_box = pd.DataFrame({
    "mu": np.log([1e-6, 1e-4]),
    "kappa": np.log([1e-8, 0.1]),
    "theta": np.log([0.000075, 0.0002]),
    "xi": np.log([5e-4, 1e-2]),
    "rho": funky_transform([0.5, 0.9]),
    "V_0": np.log([1e-6, 1e-4])
})


def runif_design(box, n_draws):
    draw_frame = pd.DataFrame()
    for param in box.columns:
        draw_frame[param] = np.random.uniform(box[param][0], box[param][1], n_draws)
    return draw_frame

initial_params_df = runif_design(sp500_box, NREPS_FITR)

# POMP model using MOP Gradient-Based Optimization
start_time = datetime.datetime.now()
key = random.key(MAIN_SEED)
fit_out_gd = []
pf_out = []

for rep in range(NREPS_FITR):
    theta_check = jnp.array(initial_params_df.iloc[rep])
    # sp500_model is initialized using theta_check drawn randomly from initial_params_df
    sp500_model = pypomp.pomp_class.Pomp(
        rinit = rinit, 
        rproc = rproc, 
        dmeas = dmeasure,
        ys = jnp.array(sp500['y'].values),
        theta = theta_check,
        covars = jnp.insert(sp500['y'].values, 0, 0)
    )

    # Stores optimized parameter estimates in fit_out_gd
    fit_result = pypomp.fit.fit(
        pomp_object = sp500_model,
        J = NP_FITR,
        method = "Newton",
        itns = 20,
        mode = "GD"
    )
    print("fit_result type:", type(fit_result))
    print(fit_result)
    fit_out_gd.append(fit_result)
    # Step size-paramters for paramters
    # doesn't support AD
    # Return: tuple - (LL array, optimized parameters array)

    """
    pf_out.append(pypomp.pfilter.pfilter(
        pomp_object = sp500_model,
        J = NP_EVAL,
        thresh = 0
    ))
    """
    # initial pfiltering performed on unoptimized parameters
    # filtering-needs after optimization step using fitted parameters
    
    # Create new POMP model using optimized parameters
    model_for_pfilter = pypomp.pomp_class.Pomp(
        rinit = rinit,
        rproc = rproc,
        dmeas = dmeasure,
        ys = jnp.array(sp500['y'].values),
        theta = fit_result[1],  # Use optimized parameters from fit()
        covars = jnp.insert(sp500['y'].values, 0, 0)
    )
    
    pf_out2 = []
    for pf_rep in range(NREPS_EVAL): 
        key, subkey = random.split(key)
        pf_out2.append(pypomp.pfilter.pfilter(
            pomp_object = model_for_pfilter,
            # assigning pomp_object to JAX array, not POMP model
            J = NP_EVAL,
            thresh = 0,
            key = subkey
        ))
    pf_out.append([np.mean(pf_out2), np.std(pf_out2)])
    print("pf type:", type(pf_out))
    print(pf_out)
    print("pf2 type:", type(pf_out2))
    print(pf_out2)


results_out = {"fit_out": fit_out_gd, "pf_out": pf_out}
end_time = datetime.datetime.now()
print(end_time - start_time)
print(pf_out)
pickle.dump(results_out, open(SAVE_RESULTS_TO, "wb"))