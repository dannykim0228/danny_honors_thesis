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
        NP_FITR = 5000
        NFITR = 100
        NREPS_FITR = 20
        NP_EVAL = 5000
        NREPS_EVAL = 24
        print("Running at level 3")
RW_SD = 0.02
RW_SD_INIT = 0.1
COOLING_RATE = 0.987

# Data Manipulation
#sp500_raw = pd.read_csv("C:/Users/ravis/OneDrive/Documents/danny_honors_thesis/data/SPX.csv")
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
    mu = jnp.exp(mu)
    kappa = jnp.exp(kappa)
    theta = jnp.exp(theta)
    xi = jnp.exp(xi)
    rho = -1 + 2 / (1 + jnp.exp(-rho))
    t = t.astype(int)
    dZ = random.normal(key)
    dWs = (covars[t] - mu + 0.5 * V) / jnp.sqrt(V)
    # dWv with correlation
    dWv = rho * dWs + jnp.sqrt(1 - rho ** 2) * dZ
    S = S + S * (mu + jnp.sqrt(jnp.maximum(V, 0.0)) * dWs)
    V = V + kappa * (theta - V) + xi * jnp.sqrt(jnp.maximum(V, 0.0)) * dWv
    t += 1
    V = jnp.maximum(V, 1e-32)
    return jnp.array([V, S, t])


# Initialization Model
def rinit(params, J, covars = None):
    #V_0 = jnp.exp(params[5])
    #V_0 = jnp.exp(jnp.clip(params[5], a_min=-10, a_max=0))
    V_0 = 7.86e-3 ** 2
    S_0 = 1105
    t = 0
    return jnp.tile(jnp.array([V_0, S_0, t]), (J, 1))


# Measurement model: how we measure state
def dmeasure(y, state, params):
    V, S, t = state
    mu = jnp.exp(params[0])
    return jax.scipy.stats.norm.logpdf(y, mu - 0.5 * V, jnp.sqrt(V))


def funky_transform(lst):
    """Transform rho to perturbation scale"""
    out = [np.log((1 + x) / (1 - x)) for x in lst]
    return out

sp500_box = pd.DataFrame({
    "mu": np.log([3.71e-4, 3.71e-4]),
    "kappa": np.log([3.25e-2, 3.25e-2]),
    "theta": np.log([1.09e-4, 1.09e-4]),
    "xi": np.log([2.22e-3, 2.22e-3]),
    "rho": funky_transform([-7.29e-1, -7.29e-1]),
    "V_0": np.log([(7.86e-3)**2, (7.86e-3)**2])
})

def runif_design(box, n_draws):
    """Draws parameters from a given box."""
    draw_frame = pd.DataFrame()
    for param in box.columns:
        draw_frame[param] = np.random.uniform(box[param][0], box[param][1], n_draws)
    return draw_frame

initial_params_df = runif_design(sp500_box, NREPS_FITR)
#print("Checking initial_params_df values (first 5 rows):") # Checking parameters are in perturbation scale
#print(initial_params_df.head()) # For Debugging


start_time = datetime.datetime.now()
key = random.key(MAIN_SEED)
fit_out_if2 = []
fit_out_ifad = []
pf_out = []
for rep in range(NREPS_FITR):
    theta_check = jnp.array(initial_params_df.iloc[rep])
    #print(f"Checking theta values before running POMP model for rep {rep}:")
    #print("e-transformed positive parameters:", jnp.exp(theta_check[:4]))  # > 0
    #print("Transformed rho:", -1 + 2 / (1 + jnp.exp(-theta_check[4])))  # (-1,1)
    sp500_model = pypomp.pomp_class.Pomp(
        rinit = rinit,
        rproc = rproc,
        dmeas = dmeasure,
        ys = jnp.array(sp500['y'].values),
        theta = theta_check,
        covars = jnp.insert(sp500['y'].values, 0, 0)
    )
    # IF2 Fit      
    fit_out_if2.append(pypomp.fit.fit(
        pomp_object = sp500_model,
        J = NP_FITR,
        M = NFITR,
        a = COOLING_RATE,
        sigmas = RW_SD,
        sigmas_init = RW_SD_INIT,
        mode = "IF2",
        thresh_mif = 0
    ))

    # IFAD Fit
    theta_if2_final = fit_out_if2[rep][1][-1].mean(axis = 0)
    fit_out_ifad.append(pypomp.fit.fit(
        pomp_object = sp500_model, 
        theta = theta_if2_final, 
        J = NP_FITR, 
        M = 0,
        sigmas = RW_SD,
        sigmas_init = RW_SD_INIT, 
        mode = "IFAD"
    ))

    model_for_pfilter = pypomp.pomp_class.Pomp(
        rinit = rinit,
        rproc = rproc,
        dmeas = dmeasure,
        ys = jnp.array(sp500['y'].values),
        theta = theta_if2_final,
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

results_out = {
    "fit_out_if2": fit_out_if2,
    "fit_out_ifad": fit_out_ifad,
    "pf_out": pf_out,
}
end_time = datetime.datetime.now()
print(end_time - start_time)
print(pf_out)
pickle.dump(results_out, open(SAVE_RESULTS_TO, "wb"))