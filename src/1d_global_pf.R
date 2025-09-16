library(pomp)
library(mvtnorm) #multivariate distributions
library(doParallel) #parallel processing
library(foreach) #parallel processing
library(doRNG) #reproducible random numbers
library(tidyverse)

#Determines number of CPU cores to use, either from environment variables or by detecting available cores
cores <-  as.numeric(Sys.getenv('SLURM_NTASKS_PER_NODE', unset=NA))
if(is.na(cores)) cores <- detectCores()  
registerDoParallel(cores) #Registers parallel processing with specified number of cores
registerDoRNG(34118892) #Ensures that parallel operations are reproducible by setting seed for RNG


# Data Manipulation 
# Calculates log returns, which represent percentage changes in closing prices
sp500_raw <- read.csv("SPX.csv")  
sp500 <- sp500_raw%>% 
  mutate(date = as.Date(Date)) %>% 
  mutate(diff_days = difftime(date, min(date), units = 'day')) %>% 
  mutate(time = as.numeric(diff_days)) %>% 
  mutate(y = log(Close / lag(Close))) %>%  #log returns are y
  select(time, y) %>% 
  drop_na() #Rows with missing values are removed


# Name of States and Parmeters 
sp500_statenames <- c("V", "S") # Define state variables V (volatility) and S (stock price)
sp500_rp_names <- c("mu", "kappa", "theta", "xi", "rho") 
sp500_ivp_names <- c("V_0")
sp500_parameters <- c(sp500_rp_names, sp500_ivp_names)
sp500_covarnames <- "covaryt" # Name covariate variable covaryt, representing time-varying inputs


# rprocess 
rproc1 <- "
  double dWv, dZ, dWs, rt;
  
  rt = covaryt;
  dWs = (rt - mu + 0.5 * V) / (sqrt(V));
  dZ = rnorm(0, 1); // Generate standard normal noise for the stochastic process
  
  dWv = rho * dWs + sqrt(1 - rho * rho) * dZ;

  S += S * (mu + sqrt(fmax(V, 0.0)) * dWs);
  V += kappa * (theta - V) + xi * sqrt(fmax(V, 0.0)) * dWv;
  // S & V are updated based on this process, ensuring V stays positive
  if (V<=0) {
    V=1e-32;
  } 
"


# Initialization Model 
sp500_rinit <- "
  V = V_0; // V_0 is a parameter as well
  S = 1105; // 1105 is the starting price
"


# rmeasure 
# Define filtered observations for y
sp500_rmeasure_filt <- "
  y = exp(covaryt);
"

# Define simulated observations for y
sp500_rmeasure_sim <- "
  y = (mu - 0.5 * V) + sqrt(V); 
"


# dmeasure 
# Calculate likelihood of observing y based on current V & mu values
sp500_dmeasure <- "
   lik = dnorm(y, mu - 0.5 * V, sqrt(V), give_log); 
"


# Parameter Transformation
# Transform parameters to log-scale or other constrained scales for estimation
my_ToTrans <- "
     T_xi = log(xi);
     T_kappa = log(kappa);
     T_theta = log(theta);
     T_V_0 = log(V_0);
     T_mu = log(mu);
     T_rho = log((rho + 1) / (1 - rho));
  "

# Transform parameters back to original scale
my_FromTrans <- "
    kappa = exp(T_kappa);
    theta = exp(T_theta);
    xi = exp(T_xi);
    V_0 = exp(T_V_0);
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
# Set random walk SD for perturbing parameters during IF2
sp500_rw.sd <- rw_sd(
  mu = sp500_rw.sd_rp,
  theta = sp500_rw.sd_rp,
  kappa = sp500_rw.sd_rp,
  xi = sp500_rw.sd_rp,
  rho = sp500_rw.sd_rp,
  V_0 = ivp(sp500_rw.sd_ivp)
)

run_level <- 4
sp500_Np <-           switch(run_level, 100,  200, 500, 1000)
sp500_Nmif <-         switch(run_level,  10,  25,  50, 200)
sp500_Nreps_eval <-   switch(run_level,   4,  7,   10,  1000)
sp500_Nreps_global <- switch(run_level,  10,  15,  20, 120)


mle_params <- c(
  mu = 3.857704e-04,
  kappa = 3.220282e-02,
  theta = 1.064001e-04,
  xi = 2.265493e-03,
  rho = -7.231781e-01,
  V_0 = 2.819792e-05
)

log_likelihoods <- replicate(sp500_Nreps_eval, {
  pfilter_result <- pfilter(
    sp500.filt,
    params = mle_params,
    Np = sp500_Np
  )
  logLik(pfilter_result)
})

# Save results into CSV
pf_val_r <- data.frame(
  loglikelihood = log_likelihoods
)

write.csv(pf_val_r, "pf_val_r.csv", row.names = FALSE)
cat("Saved particle filter loglikelihoods to pf_val_r.csv\n")
