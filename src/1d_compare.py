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
    SAVE_RESULTS_TO = "output/default_output/1d_global_if2_out.pkl"
else:
    SAVE_RESULTS_TO = out_dir + "1d_global_if2_out.pkl"

#SJNN = os.environ.get("SLURM_JOB_NUM_NODES")
#SGON = os.environ.get("SLURM_GPUS_ON_NODE")
MAIN_SEED = 631409
np.random.seed(MAIN_SEED)
RUN_LEVEL = 3
match RUN_LEVEL:
    case 1:
        NP_FITR = 2
        NFITR = 2
        NREPS_FITR = 3
        NP_EVAL = 2
        NREPS_EVAL = 5
        print("Running at level 1")
    case 2:
        NP_FITR = 1000
        NFITR = 20
        NREPS_FITR = 3
        NP_EVAL = 1000
        NREPS_EVAL = 5
        print("Running at level 2")
    case 3:
        NP_FITR = 10000
        NFITR = 10
        NREPS_FITR = 20
        NP_EVAL = 10000
        NREPS_EVAL = 24
        print("Running at level 3")
RW_SD = 0.02
RW_SD_INIT = 0.1
COOLING_RATE = 0.987

# Data Manipulation
sp500_raw = pd.read_csv("/home/kimdanny/danny_honors_thesis/data/SPX.csv")
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
    # Transform parameters onto natural scale
    mu = jnp.exp(mu)
    kappa = jnp.exp(kappa)
    theta = jnp.exp(theta)
    xi = jnp.exp(xi)
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


# Initialization Model
def rinit(params, J, covars = None):
    # Transform V_0 onto natural scale
    V_0 = 7.86e-3 ** 2
    S_0 = 1105
    t = 0
    return jnp.tile(jnp.array([V_0, S_0, t]), (J, 1))


# Measurement model: how we measure state
def dmeasure(y, state, params):
    V, S, t = state
    # Transform mu onto the natural scale
    mu = jnp.exp(params[0])
    return jax.scipy.stats.norm.logpdf(y, mu - 0.5 * V, jnp.sqrt(V))


def funky_transform(lst):
    """Transform rho to perturbation scale"""
    out = [np.log((1 + x) / (1 - x)) for x in lst]
    return out

"""
sp500_box = pd.DataFrame({
    "mu": [3.71e-4, 3.71e-4],
    "kappa": [3.25e-2, 3.25e-2],
    "theta": [1.09e-4, 1.09e-4],
    "xi": [2.22e-3, 2.22e-3],
    "rho": [-7.29e-1, -7.29e-1],
    "V_0": [(7.86e-3)**2, (7.86e-3)**2]
})
"""
sp500_box = pd.DataFrame({
    # Parameters are transformed onto the perturbation scale
    "mu": [1e-6, 1e-4],
    "kappa": [1e-8, 0.1],
    "theta": [0.000075, 0.0002],
    "xi": [1e-8, 1e-2],
    "rho": [1e-8, 1],
    "V_0": [(7.86e-3)**2, (7.86e-3)**2]
})

def runif_design(box, n_draws):
    """Draws parameters from a given box."""
    draw_frame = pd.DataFrame()
    for param in box.columns:
        draw_frame[param] = np.random.uniform(box[param][0], box[param][1], n_draws)
    
    draw_frame["mu"] = np.log(draw_frame["mu"])
    draw_frame["kappa"] = np.log(draw_frame["kappa"])
    draw_frame["theta"] = np.log(draw_frame["theta"])
    draw_frame["xi"] = np.log(draw_frame["xi"])
    draw_frame["rho"] = funky_transform(draw_frame["rho"])
    draw_frame["V_0"] = np.log(draw_frame["V_0"])

    return draw_frame

#initial_params_df = runif_design(sp500_box, NREPS_FITR)
N_STARTS = 20
initial_params_df = runif_design(sp500_box, N_STARTS)

initial_params_df.to_csv("initial_params.csv", index = False)
print("Saved 20 initial parameter sets to py csv")


# Fit POMP model using IF
start_time = datetime.datetime.now()
key = random.key(MAIN_SEED)
fit_out = []
pf_out = []
for rep in range(NREPS_FITR):
    fit_out_rep = []

    sp500_model = pypomp.pomp_class.Pomp(
        rinit = rinit,
        rproc = rproc,
        dmeas = dmeasure,
        ys = jnp.array(sp500['y'].values),
        theta = jnp.array(initial_params_df.iloc[rep]),
        covars = jnp.insert(sp500['y'].values, 0, 0)
    )      
    fit_result = pypomp.fit.fit(
        pomp_object = sp500_model,
        J = NP_FITR,
        M = NFITR,
        a = COOLING_RATE,
        sigmas = RW_SD,
        sigmas_init = RW_SD_INIT,
        mode = "IF2",
        thresh_mif = 0
    )
    loglik_trace = [-ll for ll in fit_result[0]]  # Convert negative LL to positive LL
    fit_out_rep.append(loglik_trace)
    fit_out.append(fit_out_rep)

    model_for_pfilter = pypomp.pomp_class.Pomp(
        rinit = rinit,
        rproc = rproc,
        dmeas = dmeasure,
        ys = jnp.array(sp500['y'].values),
        theta = fit_out[rep][1][-1].mean(axis = 0),
        covars = jnp.insert(sp500['y'].values, 0, 0)
    )
    pf_out2 = []
    for pf_rep in range(NREPS_EVAL): 
        key, subkey = random.split(key)
        pf_out2.append(pypomp.pfilter.pfilter(
            pomp_object = model_for_pfilter,
            J = NP_EVAL,
            thresh = 0,
            key = subkey
        ))
    pf_out.append([np.mean(pf_out2), np.std(pf_out2)])

pd.DataFrame(fit_out).to_csv("if2_ll_python.csv", index = False)
print("Saved IF2 LL traces to csv")


results_out = {
    "fit_out": fit_out,
    "pf_out": pf_out,
}
end_time = datetime.datetime.now()
print(end_time - start_time)
print(pf_out)
pickle.dump(results_out, open(SAVE_RESULTS_TO, "wb"))