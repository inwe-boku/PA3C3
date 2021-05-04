from netCDF4 import Dataset
import numpy as np

nc_file = '/home/cmikovits/Downloads/rsds_SDM_MOHC-HadGEM2-ES_rcp45_r1i1p1_CLMcom-CCLM4-8-17.nc'
nd = Dataset(nc_file, mode='r')

print(nd)
print(nd['time'])
print(nd['rsds'])

tsl = 924

lons = nd.variables['lon'][1]
lats = nd.variables['lat'][1]
time = nd.variables['time'][1:100]
rsds = nd.variables['rsds'][tsl, 40, 5]


print(min(lons), max(lons), min(lats), max(lats))
print(time,rsds)

#tmax_units = fh.variables['Tmax'].units
