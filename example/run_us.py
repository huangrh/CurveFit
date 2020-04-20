#%%
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


#%%
import numpy as np 
import pandas as pd 

import sys
import pandas
import numpy
import scipy as sp
import pdb

import curvefit
from curvefit.core.model import CurveModel

#%%
social_dist = pd.read_excel(open("G:/My Drive/Dev/covid19/social_distance/social_distance.xlsx", 'rb'),sheet_name="us_stay_home")

#%%
death_dat = pd.read_csv("G:/My Drive/Dev/covid19/death_rate/covid19_us_death_rate.csv")
death_dat['one'] = 1.0
death_dat['death_rate'] = death_dat['death_rate'] / np.max(death_dat['death_rate'])

death_dat.head()
np.max(death_dat['death_rate'])


# ==============================
# %%
import math
n_data    = 21                   # number simulated measurements to generate
b_true    = 20.0                 # b used to simulate data
a_true    = math.log(2.0/b_true) # a used to simulate data
c_true    = 1.0 / b_true         # c used to simulate data
phi_true  = math.log(0.1)        # phi used to simulate data
rel_tol   = 1e-5          # relative tolerance used to check optimal solution

# --------------------------------
# %%
# model for the mean of the data
def generalized_error_function(t, params) :
    alpha = params[0]
    beta  = params[1]
    p     = params[2]
    return 0.5 * p * ( 1.0 + sp.special.erf( alpha * ( t - beta ) ) )
#
# link function used for beta
def identity_fun(x) :
    return x
#
# link function used for alpha, p
def exp_fun(x) :
    return numpy.exp(x)
#
# inverse of function used for alpha, p
def log_fun(x) :
    return numpy.log(x)
#
# true value for fixed effects
fe_true = numpy.array( [ a_true, b_true, c_true, phi_true ] )
num_fe  = len(fe_true)
# -----------------------------------------------------------------------
#%% data_frame
independent_var = numpy.array(range(n_data)) * b_true / (n_data-1)
social_distance = numpy.zeros(n_data, dtype = float)
params_true     = numpy.zeros((n_data, 3), dtype = float)
alpha_true      = numpy.exp( a_true)
p_true          = numpy.exp( phi_true )
for i in range(n_data) :
    social_distance[i] = 0 if i < n_data / 2.0 else 1
    beta_true          = b_true + c_true * social_distance[i]
    params_true[i]     = [alpha_true, beta_true, p_true ]
params_true       = numpy.transpose(params_true)
measurement_value = generalized_error_function(independent_var, params_true)
measurement_std   = n_data * [ 0.1 ]
cov_one           = n_data * [ 1.0 ]
data_group        = n_data * [ 'world' ]
data_dict         = {
    'independent_var'   : independent_var   ,
    'measurement_value' : measurement_value ,
    'measurement_std'   : measurement_std   ,
    'cov_one'           : cov_one           ,
    'social_distance'   : social_distance   ,
    'data_group'        : data_group        ,
}
data_frame        = pandas.DataFrame(data_dict)
# ------------------------------------------------------------------------
#%% curve_model
data_frame   = death_dat
col_t        = 'day'
col_obs      = 'death_rate'
col_covs     = [ ['one'], ['one', 'social_dist'], ['one'] ]
col_group    = 'state'
param_names  = [ 'alpha', 'beta',       'p'     ]
link_fun     = [ exp_fun, identity_fun, exp_fun ]
var_link_fun = num_fe * [ identity_fun ]
fun          = generalized_error_function
col_obs_se   = 'death_rate_std'
#
curve_model = curvefit.core.model.CurveModel(
    data_frame,
    col_t,
    col_obs,
    col_covs,
    col_group,
    param_names,
    link_fun,
    var_link_fun,
    fun,
    col_obs_se
)
# -------------------------------------------------------------------------
#%% fit_params
#
num_group = len(np.unique(death_dat['state']))
# fe_init   = fe_true / 3.0
fe_init = np.array([2,20,1,1],dtype = "float") # for alpha, beta, social_distance_fator, p


# re_init   = numpy.zeros( num_fe )
num_fe = len(fe_init)
re_init = np.ones(num_fe * num_group)
fe_bounds = [ [-numpy.inf, numpy.inf] ] * num_fe
# re_bounds = [ [0.0, 0.0] ] * num_fe
re_bounds = [[-np.inf, np.inf]] * num_fe
options   = {
    'ftol' : 1e-12,
    'gtol' : 1e-12,
}
#
curve_model.fit_params(
    fe_init,
    re_init,
    fe_bounds,
    re_bounds,
    options=options
)
fe_estimate = curve_model.result.x[:num_fe]
fe_estimate
# -------------------------------------------------------------------------
# %%check result
for i in range(num_fe) :
    rel_error = fe_estimate[i] / fe_true[i] - 1.0
    assert abs(rel_error) < rel_tol
#





# %%
