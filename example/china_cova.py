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

# model for the mean of the data
def generalized_error_function(t, params) :
    alpha = params[0]
    beta  = params[1]
    p     = params[2]
    return 0.5 * p * ( 1.0 + scipy.special.erf( alpha * ( t - beta ) ) )
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

#   %% china death rate
china_death_rate = pd.read_csv('G:/My Drive/Dev/covid19/death_rate/covid19_china_death_rate.csv')
china_death_rate['one'] = 1
china_death_rate['death_rate'] = china_death_rate['death_rate']/np.max(china_death_rate['death_rate'])
china_death_rate.head()
np.max(china_death_rate['death_rate'])


hubei = china_death_rate.query('(state in "Hubei")')

#=========================================================
# %% simple model
# ------------------------------------------------------------------------
# curve_model
col_t        = 'day'
col_obs      = 'death_rate'
col_covs     = [ ['one'], ['one', 'social_dist'], ['one'] ]
col_group    = 'state'
param_names  = [ 'alpha', 'beta',       'p'     ]
link_fun     = [ exp_fun, identity_fun, exp_fun ]
var_link_fun = 3 * [ identity_fun ]
fun          = generalized_error_function
col_obs_se   = 'measurement_std'
#
curve_model = curvefit.core.model.CurveModel(
    hubei,
    col_t,
    col_obs,
    col_covs,
    col_group,
    param_names,
    link_fun,
    var_link_fun,
    fun
)
# -------------------------------------------------------------------------
# %%fit_params
#
num_fe = 4
fe_init   = np.array([1,1,1,1], dtype='float')
re_init   = numpy.zeros( num_fe )
fe_bounds = [ [-numpy.inf, numpy.inf] ] * num_fe
re_bounds = [ [0.0, 0.0] ] * num_fe
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
fe_estimate = curve_model.result.x[:3]

curve_model.result


# ------------------------------------------------------------------------
#%% curve_model - random effect + covarite model
data_frame   = china_death_rate
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
    fun
)
# -------------------------------------------------------------------------
#%% fit_params
#
num_group = len(np.unique(china_death_rate['state']))
# fe_init   = fe_true / 3.0
fe_init = np.array([1,10,0.2,3],dtype = "float") # for alpha, beta, social_distance_fator, p


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
curve_model.result
fe_estimate
# -------------------------------------------------------------------------

