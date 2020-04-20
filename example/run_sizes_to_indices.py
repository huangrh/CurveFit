#! /usr/bin/env python3
#%%
import sys
import numpy
import curvefit
#
sizes   = [ 2, 4, 3 ]
indices = curvefit.core.utils.sizes_to_indices(sizes)
assert all( indices[0] == numpy.array([0, 1]) )
assert all( indices[1] == numpy.array([2, 3, 4, 5]) )
assert all( indices[2] == numpy.array([6, 7, 8]) )
print('sizes_to_indices.py: OK')



# %%
