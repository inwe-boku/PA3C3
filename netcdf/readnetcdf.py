from netCDF4 import Dataset
import numpy as np
import datetime
import suntimes

nc_file = '/home/cmikovits/Downloads/rsds_SDM_MOHC-HadGEM2-ES_rcp45_r1i1p1_CLMcom-CCLM4-8-17.nc'
nd = Dataset(nc_file, mode='r')

#print(nd)
print(nd['time'])
print(nd['rsds'])

day = datetime.datetime(2021,7,1)
place = suntimes.SunTimes(9.3, 41.6, altitude=200)
print(place.riseutc(day))

jan = 30+360*50+0.5
jul = jan + 6*30

print(nd['rsds'][jan, 200, 300])
print(nd['rsds'][jul, 200, 300])

#lons = nd.variables['lon'][1]
#lats = nd.variables['lat'][1]
#time = nd.variables['time'][1:100]
#rsds = nd.variables['rsds'][tsl, 40, 5]


#print(min(lons), max(lons), min(lats), max(lats))
#print(time,rsds)
