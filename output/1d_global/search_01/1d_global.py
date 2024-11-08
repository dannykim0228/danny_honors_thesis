"""pypomp implementation of Weizhe's 1d_global_search.R code."""
import jax
import jax.numpy as jnp
from jax import random
import pandas as pd
import numpy as np
import datetime
import pypomp
import pypomp.fit
import pypomp.pfilter
import pypomp.pomp_class
import pickle
import os

print("Current system time:", datetime.datetime.now())

<<<<<<< HEAD:src/1d_global_danny.py
SAVE_RESULTS_TO = "output/default_output/1d_global_danny_out.pkl"
=======
out_dir = os.environ.get("out_dir")
if out_dir is None:
    SAVE_RESULTS_TO = "output/default_output/1d_global_danny_out.pkl"
else:
    SAVE_RESULTS_TO = out_dir

gpus = 1
print(gpus)
>>>>>>> 2f1d321d505ea80141f002a2c8bc545e01a90e59:output/1d_global/search_01/1d_global.py
RUN_LEVEL = 2
match RUN_LEVEL:
    case 1:
        NP_FITR = 2
        NFITR = 2
<<<<<<< HEAD:src/1d_global_danny.py
        NREPS_FITR = 2
        NP_EVAL = 2
        NREPS_EVAL = 2
=======
        NREPS_FITR = gpus
        NP_EVAL = 2
        NREPS_EVAL = gpus
        NREPS_EVAL2 = gpus
>>>>>>> 2f1d321d505ea80141f002a2c8bc545e01a90e59:output/1d_global/search_01/1d_global.py
        print("Running at level 1")
    case 2:
        NP_FITR = 100
        NFITR = 20
<<<<<<< HEAD:src/1d_global_danny.py
        NREPS_FITR = 4
        NP_EVAL = 100
        NREPS_EVAL = 4
=======
        NREPS_FITR = gpus
        NP_EVAL = 100
        NREPS_EVAL = gpus
        NREPS_EVAL2 = gpus
>>>>>>> 2f1d321d505ea80141f002a2c8bc545e01a90e59:output/1d_global/search_01/1d_global.py
        print("Running at level 2")
    case 3:
        NP_FITR = 1000
        NFITR = 200
<<<<<<< HEAD:src/1d_global_danny.py
        NREPS_FITR = 36
        NP_EVAL = 5000
        NREPS_EVAL = 36
=======
        NREPS_FITR = gpus
        NP_EVAL = 5000
        NREPS_EVAL = gpus
        NREPS_EVAL2 = gpus*8
>>>>>>> 2f1d321d505ea80141f002a2c8bc545e01a90e59:output/1d_global/search_01/1d_global.py
        print("Running at level 3")
RW_SD = 0.001
RW_SD_INIT = 0.01
COOLING_RATE = 0.5

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

def funky_transform(list):
    """Transform rho to perturbation scale"""
    out = [np.log((1 + x)/(1 - x)) for x in list]
    return(out)

sp500_box = pd.DataFrame({
    "mu": np.log([1e-6, 1e-4]),
    "kappa": np.log([1e-8, 0.1]),
    "theta": np.log([0.000075, 0.0002]),
    "xi": np.log([1e-8, 1e-2]),
    "rho": funky_transform([1e-8, 1-1e-8]),
    "V_0": np.log([1e-10, 1e-4])
})

def runif_design(box, n_draws):
    """Draws parameters from a given box."""
    draw_list = []
    draw_frame = pd.DataFrame()
    for param in box.columns:
        draw_frame[param] = np.random.uniform(box[param][0], box[param][1], n_draws)

    return draw_frame

initial_params_df = runif_design(sp500_box, NREPS_FITR)

# Initialize POMP model
sp500_model = pypomp.pomp_class.Pomp(
    rinit = rinit,
    rproc = rproc,
    dmeas = dmeasure,
    # Observed log returns
    ys = jnp.array(sp500['y'].values),
    # Initial parameters
    theta = jnp.array(initial_params_df.iloc[1]),
    # Covariates(time)
    covars = jnp.insert(sp500['y'].values, 0, 0)
)

# Fit POMP model
start_time = datetime.datetime.now()
fit_out = []
pf_out = []
for rep in range(NREPS_FITR):
    fit_out.append(pypomp.fit.fit(
        pomp_object = sp500_model,
        theta = jnp.array(initial_params_df.iloc[rep]),
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
        ys = jnp.array(sp500['y'].values),
        # Grab final parameter estimate from fit results
        theta = fit_out[rep][1][-1].mean(axis = 0),
        covars = jnp.insert(sp500['y'].values, 0, 0)
    )
    # TODO: pfilter multiple times AND get a different result each time
    pf_out2 = []
    for pf_rep in range(NREPS_EVAL):
        pf_out2.append(pypomp.pfilter.pfilter(
            pomp_object = model_for_pfilter,
            J = NP_EVAL,
            thresh = 0
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
