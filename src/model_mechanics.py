import jax
import jax.numpy as np

class ModelMechanics:
    """Class specifies the mechanics of a model."""
    rproc = None
    dmeas = None
    rinit = None
    state_names = None
    theta_names = None

class ToyMechanics001(ModelMechanics):
    """Class with attributes for an i.i.d. normal toy model."""
    state_names = ["X"]
    theta_names = ["mu", "sigma"]
    def __init__(self):
        def rproc(state, theta,  key, covars = None):
            return 0
        
        def rinit(theta, J, covars = None):
            return 0
        
        def dmeas(y, theta):
            mu, sigma = theta
            return jax.scipy.stats.multivariate_normal.logpdf(y, mu, sigma)

        self.rproc = rproc
        self.dmeas = dmeas
        self.rinit = rinit

class ModelMechanics001(ModelMechanics):
    """Class with attributes of measles model 1."""
    state_names = ["S", "E", "I", "R"]
    theta_names = ["beta", "gamma", "sigma", "alpha", "R0", "iota", "rho", "sigmaSE",
                   "psi", "cohort", "amplitude", "S_0", "E_0", "I_0", "R_0"]
    def __init__(self):
        def rproc(state, theta,  key, covars = None):
            return None
        
        def rinit(theta, J, covars = None):
            return None
        
        def dmeas(y, preds, theta):
            # A, C, Q, R = get_thetas(theta)
            return None

        self.rproc = rproc
        self.dmeas = dmeas
        self.rinit = rinit