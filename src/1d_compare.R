library(pomp)
library(mvtnorm)
library(doParallel)
library(foreach)
library(doRNG)
library(tidyverse)

cores <-  as.numeric(Sys.getenv('SLURM_NTASKS_PER_NODE', unset=NA))
if(is.na(cores)) cores <- detectCores()  
registerDoParallel(cores)
registerDoRNG(34118892)


sp500_raw <- read.csv("SPX.csv")  
sp500 <- sp500_raw%>% 
  mutate(date = as.Date(Date)) %>% 
  mutate(diff_days = difftime(date, min(date), units = 'day')) %>% 
  mutate(time = as.numeric(diff_days)) %>% 
  mutate(y = log(Close / lag(Close))) %>%  #log returns are y
  select(time, y) %>% 
  drop_na()


sp500_statenames <- c("V", "S")
sp500_rp_names <- c("mu", "kappa", "theta", "xi", "rho") 
#sp500_ivp_names <- c("V_0")
#sp500_parameters <- c(sp500_rp_names, sp500_ivp_names)
sp500_ivp_names <- c()
sp500_parameters <- sp500_rp_names
sp500_covarnames <- "covaryt"


rproc1 <- "
  double dWv, dZ, dWs, rt;
  
  rt = covaryt;
  dWs = (rt - mu + 0.5 * V) / (sqrt(V));
  dZ = rnorm(0, 1); // Generate standard normal noise for the stochastic process
  dWv = rho * dWs + sqrt(1 - rho * rho) * dZ;

  S += S * (mu + sqrt(fmax(V, 0.0)) * dWs);
  V += kappa * (theta - V) + xi * sqrt(fmax(V, 0.0)) * dWv;
  // S & V are updated based on this process, ensuring V stays positive
  if (V <= 0) {
    V = 1e-32;
  } 
"


sp500_rinit <- "
  V = 7.86e-3 * 7.86e-3;
  S = 1105;
"


sp500_rmeasure_filt <- "
  y = exp(covaryt);
"

sp500_rmeasure_sim <- "
  y = (mu - 0.5 * V) + sqrt(V); 
"

sp500_dmeasure <- "
   lik = dnorm(y, mu - 0.5 * V, sqrt(V), give_log); 
"


my_ToTrans <- "
     T_xi = log(xi);
     T_kappa = log(kappa);
     T_theta = log(theta);
     // T_V_0 = log(V_0);
     T_mu = log(mu);
     T_rho = log((rho + 1) / (1 - rho));
  "


my_FromTrans <- "
    kappa = exp(T_kappa);
    theta = exp(T_theta);
    xi = exp(T_xi);
    // V_0 = exp(T_V_0);
    mu = exp(T_mu);
    rho = -1 + 2 / (1 + exp(-T_rho));
  "

sp500_partrans <- parameter_trans(
  toEst = Csnippet(my_ToTrans),
  fromEst = Csnippet(my_FromTrans)
)


# Construct Filter Object
# Defines full POMP model
sp500.filt <- pomp(
  data = data.frame(
    y = sp500$y, time = 1:length(sp500$y)
  ),
  statenames = sp500_statenames,
  paramnames = sp500_parameters,
  covarnames = sp500_covarnames,
  times = "time",
  t0 = 0,
  covar = covariate_table(
    time = 0:length(sp500$y),
    covaryt = c(0, sp500$y),
    times = "time"
  ),
  rmeasure = Csnippet(sp500_rmeasure_filt),
  dmeasure = Csnippet(sp500_dmeasure),
  rprocess = discrete_time(step.fun = Csnippet(rproc1), delta.t = 1),
  rinit = Csnippet(sp500_rinit),
  partrans = sp500_partrans
)

# Fitting POMP to data
sp500_rw.sd_rp <- 0.02
sp500_rw.sd_ivp <- 0.1
sp500_cooling.fraction50 <- 0.5


# Filter POMP to data
sp500_rw.sd <- rw_sd(
  mu = sp500_rw.sd_rp,
  theta = sp500_rw.sd_rp,
  kappa = sp500_rw.sd_rp,
  xi = sp500_rw.sd_rp,
  rho = sp500_rw.sd_rp,
  lambda = sp500_rw.sd_rp,
  V_0 = ivp(sp500_rw.sd_ivp)
)

run_level <- 4
sp500_Np <-           switch(run_level, 2,  200, 500, 10000) # Matched with NP_FITR, NP_EVAL
sp500_Nmif <-         switch(run_level,  2,  25,  50, 10) # Matched with NFITR
sp500_Nreps_eval <-   switch(run_level,   4,  7,   10,  24) # Matched with NREPS_EVAL, Not using
# sp500_Nreps_local <-  switch(run_level,  10,  15,  20,  24) # Not used
sp500_Nreps_global <- switch(run_level,  3,  15,  20, 20) # Matched with NREPS_FITR

"""
sp500_box <- rbind(
  mu = c(3.71e-4, 3.71e-4), 
  theta = c(1.09e-4, 1.09e-4),
  kappa = c(3.25e-2, 3.25e-2),
  xi = c(2.22e-3, 2.22e-3),
  rho = c(-7.29e-1, -7.29e-1),
  V_0 = c((7.86e-3)**2, (7.86e-3)**2),
)

sp500_box <- rbind(
  mu = c(1e-6, 1e-4), 
  theta = c(0.000075, 0.0002),
  kappa = c(1e-8, 0.1),
  xi = c(1e-8, 1e-2),
  rho = c(1e-8, 1),
  V_0 = c((7.86e-3)**2, (7.86e-3)**2),
)
# Generate random initial parameter sets for each global search iteration
global_starts <- pomp::runif_design(
  lower = sp500_box[ , 1],
  upper = sp500_box[ , 2],
  nseq = sp500_Nreps_global
)

# Feller's Condition
global_starts$xi <- runif(n = nrow(global_starts), min = 0, max = sqrt(global_starts$kappa * global_starts$theta *2)) #kappa<2*xi*theta
"""

initial_params <- read.csv("initial_params.csv")

# Ensure same number of replicates
sp500_Nreps_global <- nrow(initial_params)


colnames(initial_params) <- c("mu", "kappa", "theta", "xi", "rho", "V_0")
initial_params$V_0 <- exp(initial_params$V_0)

# Each iteration starts with different set of initial parameters from global_starts
# pfilter evaluates LL across replications and logmeanexp calculates average LL with SE
"""
stew(file = sprintf("ENTER_FILE_NAME.rds"), {
  t.box <- system.time({
    if.box <- foreach(i = 1:sp500_Nreps_global, .packages = 'pomp', .combine = c,
                      .options.multicore = list(set.seed = TRUE)) %dopar%  {
                        mif2(
                          sp500.filt,
                          Nmif = sp500_Nmif,
                          rw.sd = sp500_rw.sd,
                          cooling.fraction.50 = sp500_cooling.fraction50,
                          Np = sp500_Np,
                          params = unlist(initial_params[i, ]) # Use fixed initial values
                        )
                      } # if.box contains all estimates of parameters (list of mif objects)
    
    L.box <- foreach(i = 1:sp500_Nreps_global, .packages = 'pomp', .combine = rbind,
                     .options.multicore = list(set.seed = TRUE)) %dopar% {
                       logmeanexp(
                         replicate(sp500_Nreps_eval,
                                   logLik(pfilter(sp500.filt,params = coef(if.box[[i]]), Np = sp500_Np))
                         ), 
                         se = TRUE)
                     } # matrix containing logLik and SE for each time 
  })
})
"""
if.box <- foreach(i = 1:nrow(initial_params), .packages = 'pomp', .combine = c) %dopar% {
  mif2(
    sp500.filt,
    Nmif = sp500_Nmif,
    rw.sd = sp500_rw.sd,
    cooling.fraction.50 = sp500_cooling.fraction50,
    Np = sp500_Np,
    params = unlist(initial_params[i, ])
  )

}
write.csv(if.box, "if2_ll_r.csv", row.names = FALSE)
print("Saved all IF2 results in if2_ll_r.csv")

saveRDS(if.box, "if2_results.rds")
print("Saved IF2 results to if2_results.rds")