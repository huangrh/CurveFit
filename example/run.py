#%%
import math
n_time       = 21    # number of time points used in the simulation
n_group      = 4     # number of groups
rel_tol      = 1e-4  # relative tolerance used to check optimal solution
# simulation values used for b_0, ..., b_4
b_true       = [ 20.0 , -2.0, -1.0, +1.0, +2.0 ]
# simulation values used for a_0, ..., a_4
a_true       = [ math.log(2.0) / b_true[0], -0.2, -0.1, +0.1, +0.2]
# simulation values used for phi_0, ..., phi_4
phi_true     = [ math.log(0.1), -0.3, -0.15, +0.15, +0.3 ]

#%%
fe_gprior = [
    [ a_true[0],   a_true[0]   / 100.0 ],
    [ b_true[0],   b_true[0]   / 100.0 ],
    [ phi_true[0], phi_true[0] / 100.0 ],
]


#%%
# -------------------------------------------------------------------------
import scipy
import sys
import pandas
import numpy
# import sandbox
# sandbox.path()
import curvefit
#
# number of parameters, fixed effects, random effects
num_params   = 3
num_fe       = 3
num_re       = num_fe * n_group
#
# f(t, alpha, beta, p)
def generalized_logistic(t, params) :
    alpha = params[0]
    beta  = params[1]
    p     = params[2]
    return p / ( 1.0 + numpy.exp( - alpha * ( t - beta ) ) )
#
# identity function
def identity_fun(x) :
    return x
#
# link function used for alpha, p
def exp_fun(x) :
    return numpy.exp(x)
#
# -----------------------------------------------------------------------
# %%data_frame
num_data           = n_time * n_group
time_grid          = numpy.array(range(n_time)) * b_true[0] / (n_time-1)
independent_var    = numpy.zeros(0, dtype=float)
measurement_value  = numpy.zeros(0, dtype=float)
data_group         = list()
for j in range(1, n_group + 1) :
    group_j  = 'group_' + str(j)
    alpha_j  = math.exp(a_true[0] + a_true[j])
    beta_j   = b_true[0] + b_true[j]
    p_j      = math.exp(phi_true[0] + phi_true[j])
    y_j      = generalized_logistic(time_grid, [alpha_j, beta_j, p_j])
    independent_var   = numpy.append(independent_var, time_grid)
    measurement_value = numpy.append(measurement_value, y_j)
    data_group += n_time * [ group_j ]
constant_one    = num_data * [ 1.0 ]
measurement_std = num_data * [ 0.1 ]
data_dict         = {
    'independent_var'   : independent_var   ,
    'measurement_value' : measurement_value ,
    'measurement_std'   : measurement_std   ,
    'constant_one'      : constant_one      ,
    'data_group'        : data_group        ,
}
data_frame        = pandas.DataFrame(data_dict)
# %%------------------------------------------------------------------------
# curve_model
col_t        = 'independent_var'
col_obs      = 'measurement_value'
col_covs     = num_fe *[ [ 'constant_one' ] ]
col_group    = 'data_group'
param_names  = [ 'alpha', 'beta',       'p'     ]
link_fun     = [ exp_fun, identity_fun, exp_fun ]
var_link_fun = num_fe * [ identity_fun ]
fun          = generalized_logistic
col_obs_se   = 'measurement_std'
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
# %%fit_params
#
# initialize fixed effects so correspond to true parameters divided by three
fe_init   = numpy.array( [ a_true[0], b_true[0], phi_true[0] ] ) / 3.0
re_init   = numpy.zeros( num_re )
fe_bounds = [ [-numpy.inf, numpy.inf] ] * num_fe
re_bounds = [ [-numpy.inf, numpy.inf] ] * num_fe
options={
    'disp'    : 0,
    'maxiter' : 200,
    'ftol'    : 1e-8,
    'gtol'    : 1e-8,
}
#
curve_model.fit_params(
    fe_init,
    re_init,
    fe_bounds,
    re_bounds,
    fe_gprior,
    options=options
)
fe_estimate = curve_model.result.x[0 : num_fe]
re_estimate = curve_model.result.x[num_fe :].reshape(n_group, num_fe)
# -------------------------------------------------------------------------
# %%check fixed effects
fe_truth    = [ a_true[0], b_true[0], phi_true[0] ]
for i in range(num_fe) :
    rel_error = fe_estimate[i] / fe_truth[i] - 1.0
    assert abs(rel_error) < rel_tol
for j in range(n_group) :
    re_truth = [ a_true[j+1], b_true[j+1], phi_true[j+1] ]
    for i in range(num_fe) :
        rel_err  = re_estimate[j,i] / re_truth[i]
        assert abs(rel_error) < rel_tol
#
print('random_effect.py: OK')

# %%
