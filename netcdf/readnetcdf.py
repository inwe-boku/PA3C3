import xarray
import numpy as np
import datetime
import suntimes
import pyproj
import geopandas as gpd
import pandas as pd
import math
import cftime
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import pvlib

def cccanccoords(nd, point, csrs = 'epsg:4326'):
    abslat = np.abs(nd.lat-coords['geometry'].y[0])
    abslon = np.abs(nd.lon-coords['geometry'].x[0])
    c = np.maximum(abslon, abslat)

    ([yloc], [xloc]) = np.where(c == np.min(c))
    return(nd['x'][xloc].values, nd['y'][yloc].values)

day = cftime.Datetime360Day(2050, 7, 1, 12, 0, 0, 0)
print(day.strftime('%j'))

coords = pd.DataFrame(
    {'name': ['Vienna'],
     'lat': [48.210033],
     'lon': [16.363449] })

geometry = [Point(xy) for xy in zip(coords.lon, coords.lat)]
coords = coords.drop(['lon', 'lat'], axis=1)
coords = gpd.GeoDataFrame(coords, crs="epsg:4326", geometry=geometry)

print(coords)

nc_file = '/home/cmikovits/Downloads/rsds_SDM_MOHC-HadGEM2-ES_rcp45_r1i1p1_CLMcom-CCLM4-8-17.nc'
nd = xarray.open_dataset(nc_file)

nx, ny = cccanccoords(nd, coords)
res = nd.sel(x=nx, y=ny, time = day, method = 'nearest')

print(nd.title)

# print(res['rsds'].values)

place = suntimes.SunTimes(9.3, 41.6, altitude=200)
w_s = place.setutc(day)

location = pvlib.location.Location(
    coords['geometry'].y,
    coords['geometry'].x,
    'Europe/Vienna',
    250,
    'Vienna-Austria')

dtseries = pd.date_range(start='2020-07-01', end='2020-07-03', periods=49)

for dt in dtseries:
    settime = place.setutc(dt)
    solar_position = location.get_solarposition(settime)

    zenith_sunset = solar_position['apparent_zenith'].values[0]
    w_s = solar_position['azimuth'].values[0]

    solar_position = location.get_solarposition(dt)
    w_h = solar_position['azimuth'].values[0]
    #dni = pvlib.irradiance.disc(
    #    500,
    #    w_h,
    #    dt)['dni']

    rad_hourly = 300 * math.pi/24 * (math.cos(math.radians(w_h)) - math.cos(math.radians(w_s))) / ( math.sin(math.radians(w_s)) - (2 * math.pi * math.radians(w_s) / 360 * math.cos(w_s)))
    # print(math.cos(w_h))
    print(dt, w_s, w_h, rad_hourly)
    
