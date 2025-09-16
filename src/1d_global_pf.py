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

#SJNN = os.environ.get("SLURM_JOB_NUM_NODES")
#SGON = os.environ.get("SLURM_GPUS_ON_NODE")
MAIN_SEED = 631409
np.random.seed(MAIN_SEED)

RUN_LEVEL = 3
match RUN_LEVEL:
    case 1:
        NP_EVAL = 2
        NREPS_EVAL = 5
        print("Running at level 1")
    case 2:
        NP_EVAL = 100
        NREPS_EVAL = 4
        print("Running at level 2")
    case 3:
        NP_EVAL = 1000
        NREPS_EVAL = 1000 #24
        print("Running at level 3")

RW_SD = 0.02
RW_SD_INIT = 0.1
COOLING_RATE = 0.98623

# Data Manipulation
sp500_raw = pd.read_csv("SPX.csv")
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
    """Process simulator for Weizhe model."""
    V, S, t = state
    kappa, theta, xi, rho, V_0 = params
    # Transform parameters onto natural scale
    mu = 3.857704e-04
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
    S = S + S * (mu + jnp.sqrt(jnp.maximum(V, 0.0)) * dWs)
    V = V + kappa * (theta - V) + xi * jnp.sqrt(jnp.maximum(V, 0.0)) * dWv
    t += 1
    # Feller condition to keep V positive
    V = jnp.maximum(V, 1e-32)
    # Results must be returned as a JAX array
    return jnp.array([V, S, t])

# Initialization Model
def rinit(params, J, covars = None):
    """Initial state process simulator for Weizhe model."""
    # Transform V_0 onto natural scale
    V = 2.819792e-05
    S = 1105  # Initial price
    t = 0
    # Result must be returned as a JAX array. For rinit, the states must be replicated
    # for each particle. 
    return jnp.tile(jnp.array([V, S, t]), (J,1))

# Measurement model: how we measure state
def dmeasure(y, state, params):
    """Measurement model distribution for Weizhe model."""
    V, S, t = state
    # Transform mu onto the natural scale
    mu = 3.857704e-04
    return jax.scipy.stats.norm.logpdf(y, mu - 0.5 * V, jnp.sqrt(V))

def funky_transform(lst):
    """Transform rho to perturbation scale"""
    out = [np.log((1 + x) / (1 - x)) for x in lst]
    return out

weizhe_values = jnp.array([
    np.log(3.220282e-02), 
    np.log(1.064001e-04), 
    np.log(2.265493e-03), 
    funky_transform([-7.231781e-01])[0], 
    #np.log((7.86e-3)**2)
    np.log(2.819792e-05)
    
])

# Fit POMP model
start_time = datetime.datetime.now()
key = random.key(MAIN_SEED)
pf_out = []

# Apparently the theta argument for pypomp.pfilter doesn't override whatever is
# already saved in the model object, so we need to remake the model object.
model_for_pfilter = pypomp.pomp_class.Pomp(
    rinit = rinit,
    rproc = rproc,
    dmeas = dmeasure,
    # Observed log returns
    ys = jnp.array(sp500['y'].values),
    # Initial parameters
    theta = weizhe_values,
    # Covariates(time)
    covars = jnp.insert(sp500['y'].values, 0, 0)
)
pf_out2 = []
for pf_rep in range(NREPS_EVAL):
    # JAX seed needs to be changed manually
    key, subkey = random.split(key)
    pf_out2.append(pypomp.pfilter.pfilter(
        pomp_object = model_for_pfilter,
        J = NP_EVAL,
        thresh = 0,
        key = subkey
    ))
#pf_out.append([np.mean(pf_out2), np.std(pf_out2)])
pf_out.append(pf_out2)
pf_out2_df = pd.DataFrame({
    "loglikelihood": pf_out2
})

pf_out2_df.to_csv("pf_val_py.csv", index=False)

print("Saved particle filter loglikelihoods to pf_val_py.csv")
end_time = datetime.datetime.now()
print("Run time:", end_time - start_time)
#print("Particle filter output:", pf_out) # Print LL estimates
#pickle.dump(pf_out, open(SAVE_RESULTS_TO, "wb"))


#Particle filter output: [[np.float32(-11848.294), np.float32(2.7826576)]]
#paired t-test: 


# Results: LL -11849.65, sd 1.1217372
# Run level 1: Particle filter output: [[np.float32(4.380655e+28), np.float32(inf)]]
# Run level 2: Particle filter output: [[np.float32(-11833.059), np.float32(1.114732)]]
# Run level 3: Particle filter output: [[np.float32(-11847.36), np.float32(2.0073755)]]