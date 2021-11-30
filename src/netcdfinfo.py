import numpy as np
import pandas as pd
import xarray as xr

ncfilename = '/data/projects/PA3C3/Input/rsds_SDM_ICHEC-EC-EARTH_rcp45_r1i1p1_KNMI-RACMO22E.nc'
ds = xr.open_dataset(ncfilename)

print(ds)

for var in ds.variables.values():
    print(var)