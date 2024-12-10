"""pypomp implementation of Weizhe's 1d_global_search.R code."""
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
    SAVE_RESULTS_TO = "output/default_output/1d_global_out.pkl"
else:
    SAVE_RESULTS_TO = out_dir + "1d_global_out.pkl"

#SJNN = os.environ.get("SLURM_JOB_NUM_NODES")
#SGON = os.environ.get("SLURM_GPUS_ON_NODE")
MAIN_SEED = 631409
np.random.seed(MAIN_SEED)
GPUS = 1
print("gpus:", GPUS)
RUN_LEVEL = 2
match RUN_LEVEL:
    case 1:
        NP_FITR = 2
        NFITR = 2
        NREPS_FITR = 3
        NP_EVAL = 2
        NREPS_EVAL = 5
        NREPS_EVAL2 = 5
        print("Running at level 1")
    case 2:
        NP_FITR = 1000
        NFITR = 20
        NREPS_FITR = 3
        NP_EVAL = 1000
        NREPS_EVAL = 5
        NREPS_EVAL2 = 5
        print("Running at level 2")
    case 3:
        NP_FITR = 1000
        NFITR = 200
        NREPS_FITR = GPUS
        NP_EVAL = 5000
        NREPS_EVAL = GPUS
        NREPS_EVAL2 = GPUS*8
        print("Running at level 3")
RW_SD = 0.001
RW_SD_INIT = 0.01
COOLING_RATE = 0.987

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
    # Transform parameters onto natural scale
    mu = jnp.exp(mu)
    xi = jnp.exp(xi)
    rho = -1 + 2/(1 + jnp.exp(-rho))
    # Make sure t is cast as an int
    t = t.astype(int)
    # Wiener process generation (Gaussian noise)
    dZ = random.normal(key)
    dWs = (covars[t] - mu + 0.5 * V) / jnp.sqrt(V)
    # dWv with correlation
    dWv = rho * dWs + jnp.sqrt(1 - rho ** 2) * dZ
    S = S + S * (mu + jnp.sqrt(jnp.maximum(V, 0.0)) * dWs)
    V = V + xi * jnp.sqrt(V) * dWv
    t += 1
    # Feller condition to keep V positive
    V = jnp.maximum(V, 1e-32)
    # Results must be returned as a JAX array
    return jnp.array([V, S, t])

# Initialization Model
def rinit(params, J, covars = None):
    """Initial state process simulator for Weizhe model."""
    # Transform V_0 onto natural scale
    V_0 = jnp.exp(params[5])
    S_0 = 1105  # Initial price
    t = 0
    # Result must be returned as a JAX array. For rinit, the states must be replicated
    # for each particle. 
    return jnp.tile(jnp.array([V_0, S_0, t]), (J,1))

# Measurement model: how we measure state
def dmeasure(y, state, params):
    """Measurement model distribution for Weizhe model."""
    V, S, t = state
    # Transform mu onto the natural scale
    mu = jnp.exp(params[0])
    return jax.scipy.stats.norm.logpdf(y, mu - 0.5 * V, jnp.sqrt(V))

def funky_transform(lst):
    """Transform rho to perturbation scale"""
    out = [np.log((1 + x)/(1 - x)) for x in lst]
    return out

sp500_box = pd.DataFrame({
    # Parameters are transformed onto the perturbation scale
    "mu": np.log([1e-6, 1e-4]),
    "kappa": np.log([1e-8, 0.1]),
    "theta": np.log([0.000075, 0.0002]),
    "xi": np.log([1e-8, 1e-2]),
    "rho": funky_transform([1e-8, 1-1e-8]),
    "V_0": np.log([1e-10, 1e-4])
})

def runif_design(box, n_draws):
    """Draws parameters from a given box."""
    draw_frame = pd.DataFrame()
    for param in box.columns:
        draw_frame[param] = np.random.uniform(box[param][0], box[param][1], n_draws)
    return draw_frame

initial_params_df = runif_design(sp500_box, NREPS_FITR)

# Fit POMP model
start_time = datetime.datetime.now()
key = random.key(MAIN_SEED)
fit_out = []
pf_out = []
for rep in range(NREPS_FITR):
    # Apparently the theta argument for pypomp.fit doesn't override whatever is
    # already saved in the model object, so we need to remake the model object each rep.
    sp500_model = pypomp.pomp_class.Pomp(
        rinit = rinit,
        rproc = rproc,
        dmeas = dmeasure,
        # Observed log returns
        ys = jnp.array(sp500['y'].values),
        # Initial parameters
        theta = jnp.array(initial_params_df.iloc[rep]),
        # Covariates(time)
        covars = jnp.insert(sp500['y'].values, 0, 0)
    )

    fit_out.append(pypomp.fit.fit(
        pomp_object = sp500_model,
        #theta = jnp.array(initial_params_df.iloc[rep]),
        J = NP_FITR,
        M = NFITR,
        a = COOLING_RATE,
        sigmas = RW_SD,
        sigmas_init = RW_SD_INIT,
        mode = "IF2",
        thresh_mif = 0
    ))

    # Apparently the theta argument for pypomp.pfilter doesn't override whatever is
    # already saved in the model object, so we need to remake the model object.
    model_for_pfilter = pypomp.pomp_class.Pomp(
        rinit = rinit,
        rproc = rproc,
        dmeas = dmeasure,
        # Observed log returns
        ys = jnp.array(sp500['y'].values),
        # Initial parameters
        theta = fit_out[rep][1][-1].mean(axis = 0),
        # Covariates(time)
        covars = jnp.insert(sp500['y'].values, 0, 0)
    )
    pf_out2 = []
    for pf_rep in range(NREPS_EVAL):
        # JAX seed needs to be changed manually
        key, subkey = random.split(key = key)
        pf_out2.append(pypomp.pfilter.pfilter(
            pomp_object = model_for_pfilter,
            J = NP_EVAL,
            thresh = 0,
            key = subkey
        ))
    pf_out.append([np.mean(pf_out2), np.std(pf_out2)])
results_out = {
    "fit_out": fit_out,
    "pf_out": pf_out,
}
end_time = datetime.datetime.now()
print(end_time - start_time) # run time
print(pf_out) # Print LL estimates
pickle.dump(results_out, open(SAVE_RESULTS_TO, "wb"))
