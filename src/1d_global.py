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
        NP_FITR = 2 # Number of particles for filtering
        NFITR = 2 # Number of iterated filtering steps
        NREPS_FITR = 3 # Replicates for each step
        NP_EVAL = 2
        NREPS_EVAL = 5 # Replicates for each step
        print("Running at level 1")
    case 2:
        NP_FITR = 1000
        NFITR = 20
        NREPS_FITR = 3
        NP_EVAL = 1000
        NREPS_EVAL = 5
        print("Running at level 2")
    case 3:
        NP_FITR = 1000
        NFITR = 100
        NREPS_FITR = 20
        NP_EVAL = 5000
        NREPS_EVAL = 20
        print("Running at level 3")
    
#RW_SD = 0.02 # SD for random walk parameter perturbations
#RW_SD_INIT = 0.1
RW_SD = 0.0003
RW_SD_INIT = 0.004
# COOLING_RATE = 0.98623  # This number raised to 50 is approx 0.5, so equivalent to cooling.fraction.50 = 0.5 in R
# Cooling fraction for controlling parameter perturbations
COOLING_RATE = 0.987

# Data Manipulation
sp500_raw = pd.read_csv("C:/Users/ravis/OneDrive/Documents/danny_honors_thesis/data/SPX.csv")
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


# ----------------------------------------------------------------
def rproc(state, params, key, covars = None):
    V, S, t = state
    mu, kappa, theta, xi, rho, V_0 = params
    # Transform parameters onto natural scale
    mu = jnp.exp(mu)
    #mu = 3.71e-4
    kappa = jnp.exp(kappa)
    theta = jnp.exp(theta)
    xi = jnp.exp(xi)
    rho = -1 + 2 / (1 + jnp.exp(-rho))
    # Make sure t is cast as an int
    t = t.astype(int)
    # Wiener process generation (Gaussian noise)
    dZ = random.normal(key)
    dWs = (covars[t] - mu + 0.5 * V) / jnp.sqrt(V)
    # dWv with correlation
    dWv = rho * dWs + jnp.sqrt(1 - rho ** 2) * dZ
    S = S + S * (mu + jnp.sqrt(jnp.maximum(V, 1e-32)) * dWs)
    V = V + kappa * (theta - V) + xi * jnp.sqrt(jnp.maximum(V, 0.0)) * dWv
    t += 1
    # Feller condition to keep V positive
    V = jnp.maximum(V, 1e-32)
    # Results must be returned as a JAX array
    return jnp.array([V, S, t])
# Potential Issue: Ensure indexing for covars[t] in Python aligns with R's covaryt
# ----------------------------------------------------------------


# Initialization Model
def rinit(params, J, covars = None):
    # Transform V_0 onto natural scale
    #V_0 = jnp.exp(params[5])
    V_0 = jnp.exp(jnp.clip(params[5], a_min=-10, a_max=0))
    S_0 = 1105  # Initial price
    t = 0
    # Result must be returned as a JAX array. For rinit, the states must be replicated
    # for each particle
    return jnp.tile(jnp.array([V_0, S_0, t]), (J, 1))


# ----------------------------------------------------------------
# Measurement model: how we measure state
def dmeasure(y, state, params):
    V, S, t = state
    # Transform mu onto the natural scale
    mu = jnp.exp(params[0])
    #mu = 3.71e-4
    return jax.scipy.stats.norm.logpdf(y, mu - 0.5 * V, jnp.sqrt(V))
# Potential Issue: Ensure T of mu aligns with R’s scale
# ----------------------------------------------------------------


def funky_transform(lst):
    """Transform rho to perturbation scale"""
    out = [np.log((1 + x) / (1 - x)) for x in lst]
    return out

# ----------------------------------------------------------------
sp500_box = pd.DataFrame({
    # Parameters are transformed onto the perturbation scale
    "mu": np.log([1e-6, 1e-4]), 
    "kappa": np.log([1e-8, 0.1]),
    "theta": np.log([0.000075, 0.0002]),
    #"xi": np.log([1e-8, 1e-2]),-Aaron Check
    #"rho": funky_transform([1e-8, 1 - 1e-8]),
    #"V_0": np.log([1e-10, 1e-4])
    "xi": np.log([5e-4, 1e-2]),
    #"rho": funky_transform([0.5, 0.9]),
    "rho": funky_transform(np.clip([-0.9, 0.9], -0.95, 0.95)),
    "V_0": np.log([1e-6, 1e-4])
})
"""
sp500_box = pd.DataFrame({
    # Parameters are transformed onto the perturbation scale
    "mu": [1e-6, 1e-4],
    "kappa": [1e-8, 0.1],
    "theta": [0.000075, 0.0002],
    "xi": [1e-8, 1e-2],
    "rho": [-0.9, 0.9],
    "V_0": [1e-6, 1e-4]
})

def runif_design(box, n_draws):
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
"""


def runif_design(box, n_draws):
    """Draws parameters from a given box."""
    draw_frame = pd.DataFrame()
    for param in box.columns:
        draw_frame[param] = np.random.uniform(box[param][0], box[param][1], n_draws)
    '''
    # Apply Feller to constrain xi
    # Transform kappa, theta to natural scale
    draw_frame["kappa"] = np.exp(draw_frame["kappa"])
    draw_frame["theta"] = np.exp(draw_frame["theta"])

    # Compute upbound for xi based on Feller
    xi_upper_bound = np.sqrt(2 * draw_frame["kappa"] * draw_frame["theta"])
    
    if (xi_upper_bound <= 0).any() or xi_upper_bound.isna().any():
        print("Warning: Invalid xi_upper_bound values detected!")
        print(xi_upper_bound)

    # Draw xi uniformly
    draw_frame["xi"] = np.random.uniform(0, xi_upper_bound)

    # transform kappa, theta, xi back to log 
    draw_frame["kappa"] = np.log(draw_frame["kappa"])
    draw_frame["theta"] = np.log(draw_frame["theta"])
    draw_frame["xi"] = np.log(draw_frame["xi"])
    print("Initial parameter values after applying Feller's condition:")
    print(draw_frame.describe())
    '''
    return draw_frame

initial_params_df = runif_design(sp500_box, NREPS_FITR)
 # Potential Issue: Ensure ranges of transformed parameters match ranges in R after applying inv T
# ----------------------------------------------------------------

# Fit POMP model using IF
start_time = datetime.datetime.now()
key = random.key(MAIN_SEED)
fit_out_if2 = []
fit_out_ifad = []
pf_out = []
for rep in range(NREPS_FITR):
    # Apparently the theta argument for pypomp.fit doesn't override whatever is
    # already saved in the model object, so we need to remake the model object each rep.
    # Initialize POMP model
    sp500_model = pypomp.pomp_class.Pomp(
        rinit = rinit,
        rproc = rproc,
        dmeas = dmeasure,
        # Observed log returns
        ys = jnp.array(sp500['y'].values),
        # Initial parameters
        #theta = jnp.array(initial_params_df.iloc[0]),-Aaron Check
        theta = jnp.array(initial_params_df.iloc[rep]),
        # Covariates(time)
        covars = jnp.insert(sp500['y'].values, 0, 0)
    )
    # IF2 Fit      
    fit_out_if2.append(pypomp.fit.fit(
        pomp_object = sp500_model,
        # theta = jnp.array(initial_params_df.iloc[rep]),
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
        #sigmas = RW_SD,
        #sigmas_init = RW_SD_INIT,
        # increase number of iteration
        mode = "IFAD"
        """
        M = 10, 
        a = 0.9, 
        method = 'Newton', 
        itns = 20, 
        beta = 0.9, 
        eta = 0.0025, 
        c = 0.1, 
        max_ls_itn = 10,
        thresh_mif = 100, 
        thresh_tr = 100, 
        verbose = False, 
        scale = False, 
        ls = False, 
        alpha = 0.1, 
        monitor = True, 
        mode = "IFAD"
        """
    ))

    # Potential Issue: Verify sigmas(perturbation scale) is equivalent in both implementations

    
    # Apparently the theta argument for pypomp.pfilter doesn't override whatever is
    # already saved in the model object, so we need to remake the model object
    # Evaluate model using PF
    model_for_pfilter = pypomp.pomp_class.Pomp(
        rinit = rinit,
        rproc = rproc,
        dmeas = dmeasure,
        # Observed log returns
        ys = jnp.array(sp500['y'].values),
        # Initial parameters
        theta = theta_if2_final,
        # Covariates(time)
        covars = jnp.insert(sp500['y'].values, 0, 0)
    )
    # TODO: Make sure multiple pfilters use different seeds
    pf_out2 = []
    for pf_rep in range(NREPS_EVAL): # split R.N.G
        # JAX seed needs to be changed manually
        key, subkey = random.split(key) # Fixed
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
print(end_time - start_time) # run time
print(pf_out) # Print LL estimates
pickle.dump(results_out, open(SAVE_RESULTS_TO, "wb"))