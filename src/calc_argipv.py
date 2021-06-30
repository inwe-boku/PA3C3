import typer
from pathlib import Path
import yaml
import os
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

### constants

defaultpath = os.path.join(Path.home(), 'pa3c3')


def calc_ccca_xy(nd, point, csrs='epsg:4326'):
    """
    Calculates x and y values from a netcdf file for given coordinates from a
    point with lat / lon values.
    The netcdf is expected to have a format as used by the CCCA.

    Parameters
    ----------
    nd : open netcdf file (netcdf data)

    Returns
    -------
    x, y : values (integer)
    """
    abslat = np.abs(nd.lat - point['geometry'].y[0])
    abslon = np.abs(nd.lon - point['geometry'].x[0])
    c = np.maximum(abslon, abslat)

    ([yloc], [xloc]) = np.where(c == np.min(c))
    return(nd['x'][xloc].values, nd['y'][yloc].values)


def get_ccca_values(nd, x, y, date):
    """
    Calculates x and y values from a netcdf file for given coordinates from a
    point with lat / lon values.
    The netcdf is expected to have a format as used by the CCCA.

    Parameters
    ----------
    nd : open netcdf file (netcdf data)

    Returns
    -------
    x, y : values (integer)
    """
    return(nd.sel(x=nx, y=ny, time=date, method= 'nearest'))


def sunset_time(point, date):
    place = suntimes.SunTimes(
        point['geometry'].x[0],
        point['geometry'].y[0],
        altitude=200)
    return(place.setutc(date))


def sunset_azimuth(point, date, altitude=200):
    location = pvlib.location.Location(
        point['geometry'].y[0],
        point['geometry'].x[0],
        'Europe/Vienna',
        altitude,  # müa
        'Vienna-Austria')
    return(location.get_solarposition(sunset_time(point, date))['azimuth'].values[0])


def rad_d2h_liu(w_s, w):
    """
    Calculates the ratio of daily and hourly radiation values according to:
    B. Y. H. Liu and R. C. Jordan, “The interrelationship and characteristic
    distribution of direct, diffuse and total solar radiation,” Solar Energy,
    vol. 4, no. 3, pp. 1–19, 1960.

    Parameters
    ----------
    w_s : sunset azimuth (~ sunset hour angle, where 0 = north, 180 = south)
    w : vector of sun azimuth over 24 hours (=24 values) (~ sun hour angle)

    Returns
    -------
    r : vector of relation values for each of the 24 hours
    """
    w_s_adp = w_s - 180
    rad_w_s_adp = math.radians(w_s_adp)
    cos_w_s = math.cos(rad_w_s_adp)
    sin_w_s = math.sin(rad_w_s_adp)

    for w_h in w:
        w_h_adp = 180 - w_h
        cos_w_h = math.cos(math.radians(w_h_adp))
        r = (math.pi / 24 * (cos_w_h - cos_w_s)) / (sin_w_s - (rad_w_s_adp * cos_w_s))

    return(np.clip(r, a_min=0, a_max=None))



def rad_d2h_cpr(w_s, w):
    """
    calculates the ratio of daily and hourly radiation values according to:
    M. Collares-Pereira and A. Rabl, “The average distribution of solar
    radiation-correlations between diffuse and hemispherical and between
    daily and hourly insolation values,” Solar Energy, vol. 22, no. 2,
    pp. 155–164, 1979.

    Parameters
    ----------
    w_s : sunset azimuth (~ sunset hour angle, where 0 = north, 180 = south)
    w : vector of sun azimuth over 24 hours (=24 values) (~ sun hour angle)

    Returns
    -------
    r : vector of relation values for each of the 24 hours
    """
    w_s_adp = w_s - 180
    rad_w_s_adp = math.radians(w_s_adp)
    cos_w_s = math.cos(rad_w_s_adp)
    sin_w_s = math.sin(rad_w_s_adp)
    w_s_cp = w_s_adp - 60
    sin_w_s_cp = math.sin(math.radians(w_s_cp))
    a = 0.409 + (0.5016 * sin_w_s_cp)
    b = 0.6609 - (0.4767 * sin_w_s_cp)

    for w_h in w:
        w_h_adp = 180 - w_h
        cos_w_h = math.cos(math.radians(w_h_adp))
        r = (a + b * cos_w_h) * math.pi / 24 * (cos_w_h - \
             cos_w_s) / (sin_w_s - (rad_w_s_adp * cos_w_s))

    return(np.clip(r_h, a_min=0, a_max=None))


def calc_pvoutput(point, ts_rtw, tracking, capacity_kWp, tz='UTC'):
    """
    calculates ac output in Wh of a PV installation either heading
    to the ecuator and an inclination equal to the latitude or assuming
    a single-axis tracking system for a particular location. It requires
    time series of solar radiation, temperature (in °C at 2 m)and wind
    speed (in m/s at 1 m altitude) as well as the coordinates of the location
    and either 0 or 1 to define the type of tracker as input
    """
    altitude = 0
    if point['altitutde']:
        altitude = point['altitutde']
    location = pvlib.location.Location(
        latitude=point['geometry'].y[0],
        longitude=point['geometry'].x[0],
        altitude=point['altitude'],
        tz='UTC')
    temperature = ts_rtw['temp']
    wind_speed = ts_rtw['wind']
    ghi_input = ts_rtw['rad']
    timeindex = ghi_input.asfreq(freq='1H').index
    dayofyear = timeindex.dayofyear

    solarpos = pvlib.solarposition.pyephem(timeindex, location.latitude,
                                           location.longitude,
                                           temperature=np.mean(temperature))
    zenith = solarpos['zenith']

    if tracking == 0:
        slope = lats
        if lats >= 0:
            aspect = 0
        elif lats < 0:
            aspect = 180
    elif tracking == 1:

        tracker_data = pvlib.tracking.singleaxis(solarpos['apparent_zenith'],
                                                 solarpos['azimuth'],
                                                 axis_tilt=0,
                                                 axis_azimuth=0,
                                                 max_angle=90,
                                                 backtrack=True,
                                                 gcr=2.0 / 7.0)
        slope = tracker_data['surface_tilt']
        aspect = tracker_data['surface_azimuth']
    #solartime = solarpos['solar_time']
    #clearsky_irrad = location.get_clearsky(timeindex)
    # clearsky_irrad['2018-01-01'].plot()
    dni_pre = pvlib.irradiance.disc(ghi_input, Zenith, dayofyear)['dni']
    dhi_pre = ghi_input - dni_pre * cosd(Zenith)
    weather = pd.DataFrame({'ghi': ghi_input,
                            'dni': dni_pre,
                            'dhi': dhi_pre,
                            'temp_air': temperature,
                            'wind_speed': wind_speed},
                           index=timeindex)
    # weather['2017-06-01':'2017-06-08'].plot(figsize=(18,6))
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
    # the follow selection requires some sort of automatization
    sandia_module = sandia_modules['Silevo_Triex_U300_Black__2014_']
    # Tries with the stc where understimating the yearly sum. Decided to use
    # the PTC
    PTC = 280.5
    cec_inverter = cec_inverters['ABB__MICRO_0_3_I_OUTD_US_240_240V__CEC_2014_']
    # check that the Paco is at least equal to the STC
    number_of_panels_1kWp = 1000 / PTC
    area_1kWp = number_of_panels_1kWp * sandia_module['Area']
    system = PVSystem(surface_tilt=slope, surface_azimuth=aspect,
                      module_parameters=sandia_module,
                      inverter_parameters=cec_inverter)
    mc = ModelChain(system, location)
    mc.run_model(times=weather.index, weather=weather)
    pv_output = (mc.ac * number_of_panels_1kWp *
                 installed_capacity_kWp).fillna(0)
    return pv_output

def writeGEO(data, path, dataname, types = {'geojson': 0, 'shape': 0, 'gpkg': 1}):
    if 'geojson' in types:
        data.to_file(filename = os.path.join(path, 'geojson', dataname+'.geojson'), driver="GeoJSON")
    if 'shapes' in types:
        data.to_file(filename = os.path.join(path, 'shape', dataname+'.shp'), driver = 'ESRI Shapefile')
    if 'gpkg' in types:
        data.to_file(filename = os.path.join(path, 'data.gpkg'), layer = dataname, driver = 'GPKG')
    return(0)

def attic():
    mydate = cftime.Datetime360Day(2050, 6, 20, 12, 0, 0, 0)
    print(mydate.strftime('%j'))

    coords = pd.DataFrame(
        {'name': ['Vienna','BruckNeudorf'],
         'lat': [48.21003, 48.02261],
         'lon': [16.36344, 16.83951] }
                         )
    geometry = [Point(xy) for xy in zip(coords.lon, coords.lat)]
    point = coords.drop(['lon', 'lat'], axis=1)
    point = gpd.GeoDataFrame(coords, crs="epsg:4326", geometry=geometry)

    print(point)

    nc_file = '/home/cmikovits/Downloads/rsds_SDM_MOHC-HadGEM2-ES_rcp45_r1i1p1_CLMcom-CCLM4-8-17.nc'
    nd = xarray.open_dataset(nc_file)

    nx, ny = calc_ccca_xy(nd, point)
    res = get_ccca_values(nd, nx, ny, mydate)
    rsds_value = res['rsds'].values

    w_s = sunset_azimuth(point, mydate)
    print(w_s)
    exit(0)

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
        print('rsds', rsds_value)

        datetimes = pd.date_range(start=date, end=date + + datetime.timedelta(hours=23), freq='H')
        rad = {'settime': settime,
               'w_s': w_s}
        w_s_adp = w_s - 180
        rad_w_s_adp = math.radians(w_s_adp)
        cos_w_s = math.cos(rad_w_s_adp)
        sin_w_s = math.sin(rad_w_s_adp)

        ### for collares-pereira model
        w_s_cp = w_s_adp - 60
        sin_w_s_cp = math.sin(math.radians(w_s_cp))

        a = 0.409+(0.5016*sin_w_s_cp)
        b = 0.6609-(0.4767*sin_w_s_cp)

        data = pd.DataFrame(index = datetimes, columns = {'w_h', 'z_h', 'dni_disc', 'dni_erbs', 'dhi_erbs', 'r_h', 'G_h', 'r_cp'})
        for dt in datetimes:
            solar_position = location.get_solarposition(dt)
            w_h = solar_position['azimuth'].values[0]  ### azimuth of sun
            data['w_h'].loc[dt] = w_h
            z_h = solar_position['zenith'].values[0]   ### zenith of sun
            data['z_h'].loc[dt] = z_h

            w_h_adp = 180 - w_h
            cos_w_h = math.cos(math.radians(w_h_adp))
            r_h = (math.pi/24 * (cos_w_h - cos_w_s)) / ( sin_w_s - (rad_w_s_adp * cos_w_s)) # Liu Jordan formula
            r_h = np.clip(r_h, a_min = 0, a_max=None)

            r_cp = (a + b * cos_w_h) * math.pi/24 * (cos_w_h - cos_w_s) / (sin_w_s - (rad_w_s_adp * cos_w_s))

            # formulas from: https://www.hindawi.com/journals/ijp/2015/968024/ #1
            data['r_h'].loc[dt] = r_h

            dni_disc = pvlib.irradiance.disc(
                r_h,
                z_h,
                dt)['dni']
            data['dni_disc'].loc[dt] = dni_disc
            dni_erbs = pvlib.irradiance.erbs(r_h,z_h,dt)
            data['dni_erbs'].loc[dt] = dni_erbs['dni']
            data['dhi_erbs'].loc[dt] = dni_erbs['dhi']
            data['G_h'].loc[dt] = r_h * rsds_value * 24
            data['r_cp'].loc[dt] = r_cp
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

def getyml(yamlfile):
    with open(yamlfile, 'r') as stream:
        yml = yaml.safe_load(stream)
    return yml

def main(path: Path = typer.Option(defaultpath, "--path", "-p"),
         areaname: str = typer.Option("testarea", "--area", "-a"),
         configfile: Path = typer.Option("cfg/testcfg.yml", "--config", "-c"),
         dbg: bool = typer.Option(False, "--debug", "-d")):

    global config
    typer.echo(f"Using {path}, configgile: {configfile}, areaname: {areaname}, and debug is {dbg}")
    if configfile.is_file():
        config = getyml(configfile)
        if dbg: print(config)
    else:
        message = typer.style("configfile", fg=typer.colors.WHITE, bg=typer.colors.RED, bold=True) + " is no file"
        typer.echo(message)
    
    areafile = Path(os.path.join(path, areaname, 'area.shp'))
    if areafile.is_file():
        area = gpd.read_file(areafile)
    else:
        message = typer.style(str(areafile), fg=typer.colors.WHITE, bg=typer.colors.RED, bold=True) + " does not exist"
        typer.echo(message)

    dhmfile = Path(os.path.join(path, 'dhm', 'dhm_at_lamb_10m_2018.tif'))
    if not dhmfile.is_file():
        message = typer.style(str(dhmfile), fg=typer.colors.WHITE, bg=typer.colors.RED, bold=True) + " does not exist"
        typer.echo(message)
    
    
    opts = gdal.DEMProcessingOptions(scale=111120)
    slopefile = '/tmp/slope.tif'
    gdal.DEMProcessing(slopefile, str(dhmfile), 'slope') #, options=opts)
    
    area.plot()

    

if __name__ == "__main__":
    typer.run(main)
