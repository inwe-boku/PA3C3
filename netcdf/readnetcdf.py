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

def radiation_daily2hourly(location, date, ghi_dailymean):
    return 1

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

dates = pd.date_range(start='2020-01-01', end='2020-01-03')

for date in dates:
    settime = place.setutc(date)
    solar_position = location.get_solarposition(settime)
    zenith_sunset = solar_position['apparent_zenith'].values[0]
    w_s = solar_position['azimuth'].values[0]

    datetimes = pd.date_range(start=date, end=date + + datetime.timedelta(hours=23), freq='H')
    rad = {'settime': settime,
           'w_s': w_s}
    w_harr = []
    r_h = []
    z_harr = []
    for dt in datetimes:
        solar_position = location.get_solarposition(dt)
        w_h = solar_position['azimuth'].values[0]
        z_h = solar_position['zenith'].values[0]
        #dni = pvlib.irradiance.disc(
        #    500,
        #    w_h,
        #    dt)['dni']

        r_h.append(300 * math.pi/24 * (math.cos(math.radians(w_h)) - math.cos(math.radians(w_s))) / ( math.sin(math.radians(w_s)) - (2 * math.pi * math.radians(w_s) / 360 * math.cos(w_s))))
        w_harr.append(w_h)
        z_harr.append(z_h)
    r_h = np.clip(r_h, a_min = 0, a_max=None).tolist()

    rad['w_h'] = w_harr
    rad['z_h'] = z_harr
    rad['r_h'] = r_h


    print(rad)
    
