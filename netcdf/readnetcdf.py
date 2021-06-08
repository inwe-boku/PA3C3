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
from topocalc.horizon import horizon
from topocalc.gradient import gradient_d8
from topocalc.viewf import viewf
from osgeo import gdal

from pprint import pprint

def cccanccoords(nd, point, csrs = 'epsg:4326'):
    abslat = np.abs(nd.lat-coords['geometry'].y[0])
    abslon = np.abs(nd.lon-coords['geometry'].x[0])
    c = np.maximum(abslon, abslat)

    ([yloc], [xloc]) = np.where(c == np.min(c))
    return(nd['x'][xloc].values, nd['y'][yloc].values)

def radiation_daily2hourly(location, date, ghi_dailymean):
    return 1

day = cftime.Datetime360Day(2050, 6, 20, 12, 0, 0, 0)
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

rsds_value = res['rsds'].values

place = suntimes.SunTimes(9.3, 41.6, altitude=200)
w_s = place.setutc(day)

location = pvlib.location.Location(
    coords['geometry'].y,
    coords['geometry'].x,
    'Europe/Vienna',
    250, #müa
    'Vienna-Austria')

dates = pd.date_range(start='2020-07-01', end='2020-07-03')



### sun location

for date in dates:
    settime = place.setutc(date)
    solar_position = location.get_solarposition(settime)
    zenith_sunset = solar_position['apparent_zenith'].values[0] #sunset azimuth
    w_s = solar_position['azimuth'].values[0]
    print(rsds_value)

    datetimes = pd.date_range(start=date, end=date + + datetime.timedelta(hours=23), freq='H')
    rad = {'settime': settime,
           'w_s': w_s}
    print(rad)

    data = pd.DataFrame(index = datetimes, columns = {'w_h', 'z_h', 'dni_disc', 'dni_erbs', 'dhi_erbs', 'r_h'})
    for dt in datetimes:
        solar_position = location.get_solarposition(dt)
        w_h = solar_position['azimuth'].values[0]  ### azimuth of sun
        data['w_h'].loc[dt] = w_h
        z_h = solar_position['zenith'].values[0]   ### zenith of sun
        data['z_h'].loc[dt] = z_h

        r_h = rsds_value * math.pi/24 * (math.cos(math.radians(w_h)) - math.cos(math.radians(w_s))) / ( math.sin(math.radians(w_s)) - (2 * math.pi * math.radians(w_s) / 360 * math.cos(math.radians(w_s))))
        # formula from: https://www.hindawi.com/journals/ijp/2015/968024/ #1
        data['r_h'].loc[dt] = np.clip(r_h, a_min = 0, a_max=None)

        dni_disc = pvlib.irradiance.disc(
            r_h,
            z_h,
            dt)['dni']
        data['dni_disc'].loc[dt] = dni_disc
        dni_erbs = pvlib.irradiance.erbs(r_h,z_h,dt)
        data['dni_erbs'].loc[dt] = dni_erbs['dni']
        data['dhi_erbs'].loc[dt] = dni_erbs['dhi']

    # r_h = np.clip(r_h, a_min = 0, a_max=None).tolist()

    print(data)



exit(0)
### horizon / terrain calculation

options = gdal.WarpOptions(cutlineDSName="/home/cmikovits/myshape.shp",cropToCutline=True)
outBand = gdal.Warp(srcDSOrSrcDSTab="/home/cmikovits/GEODATA/DHMAT/dhm_at_lamb_10m_2018.tif",
                        destNameOrDestDS="/tmp/cut.tif",
                        options=options)
outBand = None


ds = gdal.Open("/tmp/cut.tif")
#print(ds.info())
#gt = ds.GetGeoTransform()
dem = np.array(ds.GetRasterBand(1).ReadAsArray())
dem = dem.astype(np.double)

print(dem)

dem_spacing = 10

hrz = horizon(0, dem, dem_spacing)
#slope, aspect = gradient_d8(dem, dem_spacing, dem_spacing)
#svf, tvf = viewf(dem, spacing=dem_spacing)

print(hrz)

plt.imshow(hrz)
plt.show()

### PV System Modelling

modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
inverter = pvlib.pvsystem.retrieve_sam('cecinverter')
inverter = inverter['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
temperature_m = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']


system = pvlib.pvsystem.PVSystem(surface_tilt=20, surface_azimuth=180,
                                 module_parameters = modules,
                                 inverter_parameters = inverter,
                                 temperature_model_parameters = temperature_m)

print(system)
