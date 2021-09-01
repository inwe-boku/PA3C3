import typer
from pathlib import Path
import json
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
from shapely.geometry import Point, Polygon, box
import matplotlib.pyplot as plot
import pvlib
from topocalc.horizon import horizon
from topocalc.gradient import gradient_d8
from topocalc.viewf import viewf
from osgeo import gdal
#import pycrs

import rasterio.features
from rasterio.features import shapes
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask

from pprint import pprint
import tempfile
import renspatial as rs

# constants

DEFAULTPATH = os.path.join('exampledata')

# settings
pd.options.mode.chained_assignment = None  # default='warn'


def calc_ccca_xy(nd, point):
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
    abslat = np.abs(nd.lat - point.geometry.y)
    abslon = np.abs(nd.lon - point.geometry.x)
    c = np.maximum(abslon, abslat)

    ([yloc], [xloc]) = np.where(c == np.min(c))
    return(nd['x'][xloc].values, nd['y'][yloc].values)


def get_ccca_values(nd, nx, ny, date):
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
    return(nd.sel(x=nx, y=ny, method='nearest', time=date,))


def sunset_time(location, date):
    place = suntimes.SunTimes(
        location.longitude,
        location.latitude,
        altitude=location.altitude)
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
    r : vector of relation values for each of the 24 hoursu
    """
    w_s_adp = w_s - 180
    rad_w_s_adp = math.radians(w_s_adp)
    cos_w_s = math.cos(rad_w_s_adp)
    sin_w_s = math.sin(rad_w_s_adp)
    ratio = []
    for w_h in w:
        w_h_adp = 180 - w_h
        cos_w_h = math.cos(math.radians(w_h_adp))
        r = (math.pi / 24 * (cos_w_h - cos_w_s)) / \
            (sin_w_s - (rad_w_s_adp * cos_w_s))
        ratio.append(r)
    return(np.clip(ratio, a_min=0, a_max=None))


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
        r = (a + b * cos_w_h) * math.pi / 24 * (cos_w_h -
                                                cos_w_s) / (sin_w_s - (rad_w_s_adp * cos_w_s))
        print(r)
    return(np.clip(r, a_min=0, a_max=None))


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
    # solartime = solarpos['solar_time']
    # clearsky_irrad = location.get_clearsky(timeindex)
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


def writeGEO(data, path, dataname, types={'geojson': 0, 'shape': 0, 'gpkg': 1}):
    if 'geojson' in types:
        data.to_file(filename=os.path.join(path, 'geojson',
                     dataname+'.geojson'), driver="GeoJSON")
    if 'shapes' in types:
        data.to_file(filename=os.path.join(
            path, 'shape', dataname+'.shp'), driver='ESRI Shapefile')
    if 'gpkg' in types:
        data.to_file(filename=os.path.join(path, 'data.gpkg'),
                     layer=dataname, driver='GPKG')
    return(0)


def attic():
    dates = pd.date_range(start='2020-07-01', end='2020-07-03')

    # sun location

    for date in dates:
        settime = place.setutc(date)
        solar_position = location.get_solarposition(settime)
        # sunset azimuth
        zenith_sunset = solar_position['apparent_zenith'].values[0]
        w_s = solar_position['azimuth'].values[0]
        print('rsds', rsds_value)

        datetimes = pd.date_range(
            start=date, end=date + + datetime.timedelta(hours=23), freq='H')
        rad = {'settime': settime,
               'w_s': w_s}
        w_s_adp = w_s - 180
        rad_w_s_adp = math.radians(w_s_adp)
        cos_w_s = math.cos(rad_w_s_adp)
        sin_w_s = math.sin(rad_w_s_adp)

        # for collares-pereira model
        w_s_cp = w_s_adp - 60
        sin_w_s_cp = math.sin(math.radians(w_s_cp))

        a = 0.409+(0.5016*sin_w_s_cp)
        b = 0.6609-(0.4767*sin_w_s_cp)

        data = pd.DataFrame(index=datetimes, columns={
                            'w_h', 'z_h', 'dni_disc', 'dni_erbs', 'dhi_erbs', 'r_h', 'G_h', 'r_cp'})
        for dt in datetimes:
            solar_position = location.get_solarposition(dt)
            w_h = solar_position['azimuth'].values[0]  # azimuth of sun

            data['w_h'].loc[dt] = w_h
            z_h = solar_position['zenith'].values[0]  # zenith of sun
            data['z_h'].loc[dt] = z_h

            w_h_adp = 180 - w_h
            cos_w_h = math.cos(math.radians(w_h_adp))
            r_h = (math.pi/24 * (cos_w_h - cos_w_s)) / \
                (sin_w_s - (rad_w_s_adp * cos_w_s))  # Liu Jordan formula
            r_h = np.clip(r_h, a_min=0, a_max=None)

            r_cp = (a + b * cos_w_h) * math.pi/24 * (cos_w_h -
                                                     cos_w_s) / (sin_w_s - (rad_w_s_adp * cos_w_s))

            # formulas from: https://www.hindawi.com/journals/ijp/2015/968024/ #1
            data['r_h'].loc[dt] = r_h

            dni_disc = pvlib.irradiance.disc(
                r_h,
                z_h,
                dt)['dni']
            data['dni_disc'].loc[dt] = dni_disc
            dni_erbs = pvlib.irradiance.erbs(r_h, z_h, dt)
            data['dni_erbs'].loc[dt] = dni_erbs['dni']
            data['dhi_erbs'].loc[dt] = dni_erbs['dhi']
            data['G_h'].loc[dt] = r_h * rsds_value * 24
            data['r_cp'].loc[dt] = r_cp
        # r_h = np.clip(r_h, a_min = 0, a_max=None).tolist()

        print(data)

    exit(0)
    # horizon / terrain calculation

    options = gdal.WarpOptions(
        cutlineDSName="/home/cmikovits/myshape.shp", cropToCutline=True)
    outBand = gdal.Warp(srcDSOrSrcDSTab="/home/cmikovits/GEODATA/DHMAT/dhm_at_lamb_10m_2018.tif",
                        destNameOrDestDS="/tmp/cut.tif",
                        options=options)
    outBand = None

    ds = gdal.Open("/tmp/cut.tif")
    # print(ds.info())
    # gt = ds.GetGeoTransform()
    dem = np.array(ds.GetRasterBand(1).ReadAsArray())
    dem = dem.astype(np.double)

    print(dem)

    dem_spacing = 10

    hrz = horizon(0, dem, dem_spacing)
    # slope, aspect = gradient_d8(dem, dem_spacing, dem_spacing)
    # svf, tvf = viewf(dem, spacing=dem_spacing)

    print(hrz)

    plt.imshow(hrz)
    plt.show()

    # PV System Modelling

    modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    inverter = pvlib.pvsystem.retrieve_sam('cecinverter')
    inverter = inverter['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
    temperature_m = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[
        'sapm']['open_rack_glass_glass']

    system = pvlib.pvsystem.PVSystem(surface_tilt=20, surface_azimuth=180,
                                     module_parameters=modules,
                                     inverter_parameters=inverter,
                                     temperature_model_parameters=temperature_m)

    print(system)


def reproj_raster(filename, dst_crs):
    src_file = filename
    extension = os.path.splitext(filename)[1]
    descriptor, dst_file = tempfile.mkstemp(suffix=extension)

    with rasterio.open(src_file) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(dst_file, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
    return(dst_file)


def slope_from_dhm(dhmname):
    extension = os.path.splitext(dhmname)[1]
    descriptor, slope_file = tempfile.mkstemp(suffix=extension)
    opts = gdal.DEMProcessingOptions(scale=111120)
    gdal.DEMProcessing(slope_file, str(dhmname), 'slope')  # , options=opts)
    return(slope_file)


def areaselection():
    config['files']['landusefile'] = Path(
        os.path.join(path, areaname, 'landuse.tif'))
    if dbg:
        typer.echo(
            f"Reading LU file: {config['files']['landusefile']}")
    if config['files']['landusefile'].is_file():
        config['files']['landusefile'] = reproj_raster(
            config['files']['landusefile'], config['gis']['processcrs'])
    else:
        message = typer.style(str(config['files']['landusefile']), fg=typer.colors.WHITE,
                              bg=typer.colors.RED, bold=True) + " does not exist"
        typer.echo(message)
        raise typer.Exit()

    extension = os.path.splitext(config['files']['landusefile'])[1]
    descriptor, croplufile = tempfile.mkstemp(suffix=extension)
    options = gdal.WarpOptions(
        cutlineDSName=config['files']['area'], cropToCutline=True)
    outBand = gdal.Warp(srcDSOrSrcDSTab=config['files']['landusefile'],
                        destNameOrDestDS=croplufile,
                        options=options)
    outBand = None

    # calculate slope from DHM
    if dbg:
        typer.echo(
            f"Calculate slope from DHM")
    config['files']['slope'] = slope_from_dhm(config['files']['dhm'])

    # Mask is a numpy array binary mask loaded however needed
    if dbg:
        typer.echo(
            f"Landuse Raster to Polygons")

    mask = None
    with rasterio.Env():
        with rasterio.open(str(croplufile)) as src:
            rastercrs = src.crs
            image = src.read(1)  # first band
            results = (
                {'properties': {'landuse': int(v), 'lujson': json.dumps([
                    int(v)])}, 'geometry': s}
                for i, (s, v)
                in enumerate(
                    shapes(image, mask=mask, transform=src.transform)))
    geoms = list(results)
    gpd_polygonized_raster = gpd.GeoDataFrame.from_features(geoms)
    gpd_polygonized_raster = gpd_polygonized_raster.set_crs(rastercrs)
    geoms = gpd_polygonized_raster.buffer(0)
    geoms = gpd.GeoDataFrame.from_features(geoms)
    gpd_polygonized_raster = gpd.GeoDataFrame(pd.concat(
        [geoms, gpd_polygonized_raster[['landuse', 'lujson']]], axis=1), crs=gpd_polygonized_raster.crs)

    # landuse selection
    if dbg:
        typer.echo(
            f"Calculate landuse polygons")

    lupolys = rs.analysevector(gpd_polygonized_raster, infield='lujson', outfield='B_landuse', op='contains',
                               cmp='none', vals=config['landuse']['free'])
    # aggregation of features
    lupolys = lupolys.dissolve(by='landuse')
    lupolys = lupolys.explode()
    lupolys = lupolys.drop(columns=['lujson'])

    # area calculation & selection
    lupolys = lupolys.to_crs('epsg:6933')
    lupolys['area'] = lupolys['geometry'].area.astype(int)
    lupolys = lupolys.to_crs(config['gis']['processcrs'])
    lupolys['B_area'] = False
    lupolys.loc[lupolys['area'] > config['landuse']
                ['minarea'], 'B_area'] = True
    # compactness calculation & selection
    lupolys['compactness'] = lupolys.geometry.apply(rs.s_compactness)
    lupolys['B_compact'] = False
    lupolys.loc[lupolys['compactness'] > config['landuse']
                ['mincompactness'], 'B_compact'] = True

    # sample altitude & slope and filter
    if dbg:
        message = "Sampling altitude & slope"
        typer.echo(message)

    # sample altitude & slope and filter
    lupolys['PV'] = False
    lupolys.loc[(lupolys['B_landuse'] == True) & (lupolys['B_area'] == True)
                & (lupolys['B_compact'] == True), 'PV'] = True
    lupolys.reset_index(inplace=True)

    samples_per_ha = 0.25
    typer.echo(
        f"\tCreating random points")
    points = rs.randompoints(lupolys[lupolys['PV']], samples_per_ha)
    typer.echo(
        f"\taltitude ...")
    points = rs.samplerasterpoints(points, config['files']['dhm'],
                                   fieldname='altitude', samplemethod=1)
    typer.echo(
        f"\tslope ...")
    points = rs.samplerasterpoints(points, config['files']['slope'],
                                   fieldname='slope', samplemethod=1)
    points['nidx'] = points.index

    lupolys = rs.nearestgeom(lupolys, points, neighbor=1)
    lupolys = lupolys.merge(
        points[["nidx", "altitude", "slope"]], on='nidx', how='inner').drop(columns=['nidx'])

    lupolys = rs.analysevector(lupolys, infield='altitude', outfield='B_altitude', op='lt',
                               cmp='multi', vals=[2000])
    lupolys = rs.analysevector(lupolys, infield='slope', outfield='B_slope', op='lt',
                               cmp='multi', vals=[20])
    lupolys['PV'] = False
    lupolys.loc[(lupolys['B_landuse']) & (lupolys['B_area'])
                & (lupolys['B_compact']) & (lupolys['B_altitude']) & (lupolys['B_slope']), 'PV'] = True
    points = gpd.sjoin(
        points, lupolys[lupolys['PV']], how='left', op='within')
    points.drop(['altitude_left', 'nidx', 'index_right',
                'level_1', 'ndst'], axis=1, inplace=True)
    points.rename(columns={'slope_left': 'slope',
                           'altitude_right': 'altitude',
                           'slope_right': 'slope'}, inplace=True)
    points = points.dropna()
    return(lupolys, points)


def gendates(startyears, ylength):
    dates360 = {}
    dates = {}
    for y in startyears:
        startdt360 = cftime.Datetime360Day(y, 1, 1, 12, 0, 0, 0)
        dates360[y] = xarray.cftime_range(
            start=startdt360, periods=365*ylength, freq='D')
        startdt = datetime.datetime.strptime(str(y)+"-01-01", "%Y-%m-%d")
        enddt = datetime.datetime.strptime(
            str(y+ylength-1)+"-12-31", "%Y-%m-%d")
        dates[y] = [
            startdt + datetime.timedelta(days=x) for x in range(0, (enddt-startdt).days)]
    return(dates360, dates)


def cccapoints(nd, points, daterange):
    points = points.to_crs(config['ccca']['crs'])
    cccadict = {}
    for idx, row in points.iterrows():
        # logging.info('sampling %s for point %d of %d',
        #             fieldname, int(format(idx + 1)), int(len(points)))
        nx, ny = calc_ccca_xy(nd, row)
        nxnykey = str(nx)+'-'+str(ny)
        points.loc[idx, 'nxny'] = nxnykey
        # nxny.append(nxnykey)
        if not nxnykey in cccadict.keys():
            #daterange = daterange.to_datetimeindex
            res = get_ccca_values(nd, nx, ny, daterange)
            rsds_values = res['rsds'].values
            cccadict[nxnykey] = {}
            cccadict[nxnykey]['rsds'] = rsds_values  # fill numpy ndarray
            cccadict[nxnykey]['date'] = daterange
            cccadict[nxnykey]['geom'] = row.geometry
            cccadict[nxnykey]['altitude'] = row.altitude
    return(points, cccadict)


def ghid2ghih(dvalues, daterange, location):
    i = 0
    data = pd.DataFrame()
    while i < 5:  # len(daterange):
        date = daterange[i]
        dval = dvalues[i]
        # sunset azimuth
        settime = sunset_time(location, date)
        solar_position = location.get_solarposition(settime)
        w_s = solar_position['azimuth'].values[0]

        # hourly azimuth
        datetimes = pd.date_range(
            start=date, end=date + + datetime.timedelta(hours=23), freq='H')
        solar_position = location.get_solarposition(datetimes)
        w_h = solar_position['azimuth'].values  # azimuth of sun

        # daily to hourly values
        ratio = rad_d2h_liu(w_s, w_h)
        #print(dval, ratio)
        hvalues = dval*ratio*24
        hdata = pd.DataFrame(hvalues, index=datetimes, columns={'GHI'})
        data = data.append(hdata)
        i += 1
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(data)
    exit(0)

    print(dval, datetimes, hvalues)
    i += 1
    exit(0)

    dni_disc = pvlib.irradiance.disc(
        r_h,
        z_h,
        dt)['dni']
    data['dni_disc'].loc[dt] = dni_disc
    dni_erbs = pvlib.irradiance.erbs(r_h, z_h, dt)
    data['dni_erbs'].loc[dt] = dni_erbs['dni']
    data['dhi_erbs'].loc[dt] = dni_erbs['dhi']
    data['G_h'].loc[dt] = r_h * rsds_value * 24
    data['r_cp'].loc[dt] = r_cp
    print(data)
    exit(0)

    datetimes = pd.date_range(
        start=date, end=date + + datetime.timedelta(hours=23), freq='H')
    rad = {'settime': settime,
           'w_s': w_s}
    w_s_adp = w_s - 180
    rad_w_s_adp = math.radians(w_s_adp)
    cos_w_s = math.cos(rad_w_s_adp)
    sin_w_s = math.sin(rad_w_s_adp)

    # for collares-pereira model
    w_s_cp = w_s_adp - 60
    sin_w_s_cp = math.sin(math.radians(w_s_cp))

    # r_h = np.clip(r_h, a_min = 0, a_max=None).tolist()

    print(data)

    #w_h = solar_position['azimuth'].values
    w_h_adp = 180 - solar_position['azimuth'].values
    cos_w_h_adp = math.cos(math.radians(w_h_adp))
    r_h = (math.pi/24 * (cos_w_h_adp - cos_w_s)) / \
        (sin_w_s - (rad_w_s_adp * cos_w_s))  # Liu Jordan formula
    r_h = np.clip(r_h, a_min=0, a_max=None)
    return(hvalues)


def main(t_path: Path = typer.Option(DEFAULTPATH, "--path", "-p"),
         t_areaname: str = typer.Option("BruckSmall", "--area", "-a"),
         t_configfile: Path = typer.Option(
             "cfg/testcfg.yml", "--config", "-c"),
         t_cccancfile: str = typer.Option(
             "/data/Geodata/CCCA/rsds_SDM_MOHC-HadGEM2-ES_rcp45_r1i1p1_CLMcom-CCLM4-8-17.nc", "--cccanc", "-nc"),
         t_dbg: bool = typer.Option(False, "--debug", "-d")
         ):

    # define important stuff global
    global config

    global path
    global areaname
    global dbg

    path = t_path
    areaname = t_areaname
    dbg = t_dbg

    # read config

    typer.echo(
        f"Using path: {path}, configgile: {t_configfile}, areaname: {areaname}, and debug: {dbg}")
    if t_configfile.is_file():
        config = rs.getyml(t_configfile)
        if dbg:
            typer.echo(config)
    else:
        message = "configfile " + typer.style(str(t_configfile), fg=typer.colors.WHITE,
                                              bg=typer.colors.RED, bold=True) + " is no file"
        typer.echo(message)
        raise typer.Exit()

    config['files'] = {}
    config['files']['area'] = Path(os.path.join(path, areaname, 'area.shp'))
    config['files']['cccanc'] = Path(t_cccancfile)

    if dbg:
        typer.echo(
            f"Reading area: {config['files']['area']}")
    if config['files']['area'].is_file():
        area = gpd.read_file(config['files']['area'])
        area = area.to_crs(config['gis']['processcrs'])
    else:
        message = typer.style(str(config['files']['area']), fg=typer.colors.WHITE,
                              bg=typer.colors.RED, bold=True) + " does not exist"
        typer.echo(message)
        raise typer.Exit()

    if config['files']['cccanc'].is_file():
        nd = xarray.open_dataset(config['files']['cccanc'])
    else:
        message = typer.style(str(config['files']['cccanc']), fg=typer.colors.WHITE,
                              bg=typer.colors.RED, bold=True) + " does not exist"
        typer.echo(message)
        raise typer.Exit()

    config['files']['dhm'] = Path(os.path.join(path, areaname, 'dhm.tif'))
    if dbg:
        typer.echo(
            f"Reading DHM: {config['files']['dhm']}")
    if config['files']['dhm'].is_file():
        config['files']['dhm'] = reproj_raster(
            config['files']['dhm'], config['gis']['processcrs'])
    else:
        message = typer.style(str(config['files']['dhm']), fg=typer.colors.WHITE,
                              bg=typer.colors.RED, bold=True) + " does not exist"
        typer.echo(message)
        raise typer.Exit()

    if dbg:
        typer.echo(
            f"Selecting areas")

    lupolys, points = areaselection()
    # sunrise / sunset at area center
    # readNETCDF
    if dbg:
        typer.echo(
            f"Reading CCCA data")

    dates360, dates = gendates(
        config['ccca']['startyears'], config['ccca']['timeframe'])
    for year, daterange in dates360.items():
        points, cccadict = cccapoints(nd, points, daterange)

    for year, daterange in dates.items():
        for nxny in cccadict.keys():
            dvalues = cccadict[nxny]['rsds']
            geom = cccadict[nxny]['geom']
            altitude = int(json.loads(cccadict[nxny]['altitude'])[0])
            location = pvlib.location.Location(
                geom.y, geom.x,
                'UTC', altitude, nxny)

            hvalues = ghid2ghih(dvalues, daterange, location)
            print(hvalues)
            exit(0)

    # GHI_daily to GHI_hourly
    # DNI+DHI hourly

    rs.writeGEO(lupolys, path.joinpath(Path.home(), 'pa3c3out'), 'PVlupolys')
    rs.writeGEO(points, path.joinpath(Path.home(), 'pa3c3out'), 'PVpoints')

    typer.echo("finished")


if __name__ == "__main__":
    typer.run(main)
