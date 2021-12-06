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
#from topocalc.horizon import horizon
#from topocalc.gradient import gradient_d8
from topocalc.viewf import viewf
from osgeo import gdal
import calendar
# import pycrs

import rasterio.features
from rasterio.features import shapes
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask

from pprint import pprint
import tempfile
import renspatial as rs

pd.set_option("max_rows", 9999)

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
    # print(point)
    # print(nd['x'][xloc].values, nd['y'][yloc].values)
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


def get_horizon_values(nd, nx, ny):
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
    return(nd.sel(x=nx, y=ny, method='nearest'))


def sunset_time(location, date):
    """[summary]

    Parameters
    ----------
    location : [type]
        [description]
    date : [type]
        [description]
    """
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


def rad_d2h(w_s, w, method):
    """
    Calculates the ratio of daily and hourly radiation values.

    Parameters
    ----------
    w_s : sunset azimuth (~ sunset hour angle, where 0 = north, 180 = south)
    w : vector of sun azimuth over 24 hours (=24 values) (~ sun hour angle)

    Returns
    -------
    r : vector of relation values for each of the 24 hoursu
    """
    if method == 'liu':
        """
        Using the method presented in:
        B. Y. H. Liu and R. C. Jordan, “The interrelationship and characteristic
        distribution of direct, diffuse and total solar radiation,” Solar Energy,
        vol. 4, no. 3, pp. 1–19, 1960.
        """
        w_s_adp = w_s - 180
        rad_w_s_adp = math.radians(w_s_adp)
        cos_w_s = math.cos(rad_w_s_adp)
        sin_w_s = math.sin(rad_w_s_adp)
        ratio = []
        for w_h in w:
            w_h_adp = w_h - 180
            cos_w_h = math.cos(math.radians(w_h_adp))
            # r = (math.pi/24 * (cos_w_h - cos_w_s)) / \
            #    (sin_w_s - (rad_w_s_adp * cos_w_s))
            r = ((((math.pi)/24)*(cos_w_h-cos_w_s)) /
                 (sin_w_s-rad_w_s_adp*cos_w_s))
            ratio.append(r)
    elif method == 'cpr':
        """
        Using the method presented in:
        M. Collares-Pereira and A. Rabl, “The average distribution of solar
        radiation-correlations between diffuse and hemispherical and between
        daily and hourly insolation values,” Solar Energy, vol. 22, no. 2,
        pp. 155–164, 1979.
        """
        w_s_adp = w_s - 180
        rad_w_s_adp = math.radians(w_s_adp)
        cos_w_s = math.cos(rad_w_s_adp)
        sin_w_s = math.sin(rad_w_s_adp)
        w_s_cp = w_s_adp - 60
        sin_w_s_cp = math.sin(math.radians(w_s_cp))
        a = 0.409 + (0.5016 * sin_w_s_cp)
        b = 0.6609 - (0.4767 * sin_w_s_cp)

        ratio = []
        for w_h in w:
            w_h_adp = w_h - 180
            cos_w_h = math.cos(math.radians(w_h_adp))
            # r = (a + (b * cos_w_h)) * \
            #    (math.pi / 24 * (cos_w_h - cos_w_s)) / \
            #    (sin_w_s - ((math.pi * w_s_adp/180) * cos_w_s))
            r = (math.pi / 24) * (a + b * cos_w_h) * \
                (cos_w_h - cos_w_s) / \
                (sin_w_s - rad_w_s_adp * cos_w_s)
            ratio.append(r)
    elif method == 'garg':
        """
        Using the method presented in:
        H.P.Garg and S.N.Garg, “Improved correlation of daily and hourly diffuse
        radiation with global radiation for Indian stations,”
        Solar Energy, vol. 22, no. 2,
        pp. 155–164, 1979.
        """
        w_s_adp = w_s - 180
        rad_w_s_adp = math.radians(w_s_adp)
        cos_w_s = math.cos(rad_w_s_adp)
        sin_w_s = math.sin(rad_w_s_adp)
        w_s_cp = w_s_adp - 60
        sin_w_s_cp = math.sin(math.radians(w_s_cp))

        ratio = []
        for w_h in w:
            w_h_adp = w_h - 180
            rad_w_h = math.radians(w_h_adp)
            cos_w_h = math.cos(rad_w_h)
            r = (math.pi / 24) * \
                (cos_w_h - cos_w_s) / \
                (sin_w_s - rad_w_s_adp * cos_w_s) - \
                (0.008 * math.sin(3 * (rad_w_h - 0.65)))
            ratio.append(r)
    return(np.clip(ratio, a_min=0, a_max=None))


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
    descriptor, slope_file = tempfile.mkstemp(
        prefix='pa3c3slope', suffix=extension)
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
    # print(croplufile)
    with rasterio.Env():
        with rasterio.open(str(croplufile)) as src:
            rastercrs = src.crs
            image = src.read(1)  # first band
            # print(image)
            results = (
                {'properties': {'landuse': int(v), 'lujson': json.dumps([
                    int(v)])}, 'geometry': s}
                for i, (s, v) in enumerate(
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

    typer.echo(
        f"\tCreating random points ...")
    points = rs.randompoints(
        lupolys[lupolys['PV']], config['landuse']['points_per_ha'])
    typer.echo(
        f"finished")
    typer.echo(
        f"\taltitude ...")
    points = rs.samplerasterpoints(points, config['files']['dhm'],
                                   fieldname='altitude', samplemethod=1)
    typer.echo(
        f"finished")
    typer.echo(
        f"\tslope ...")
    points = rs.samplerasterpoints(points, config['files']['slope'],
                                   fieldname='slope', samplemethod=1)
    typer.echo(
        f"finished")
    typer.echo(
        f"\tjoining polygons and points ...")
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
    typer.echo(
        f"finished")
    return(lupolys, points)


def gendates360(startyears, ylength):
    dates = {}
    for y in startyears:
        startdt = cftime.Datetime360Day(y, 1, 1, 12, 0, 0, 0)
        enddt = cftime.Datetime360Day(y+ylength-1, 12, 30, 12, 0, 0, 0)
        dates[y] = xarray.cftime_range(
            start=startdt, end=enddt, freq='D', calendar='360_day')
    return(dates)


def gendates365(startyears, ylength):
    dates = {}
    for y in startyears:
        dr = pd.date_range(start=str(y)+"-01-01",
                           end=str(y+ylength-1)+"-12-31", freq='1d')
        dates[y] = dr[(dr.day != 29) | (dr.month != 2)]
    return(dates)


def cccapoints(nd, points, daterange, daterange365):
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
            res = get_ccca_values(nd, nx, ny, daterange)
            # rsds_values = res['rsds'].values
            # values = values_day360_day365(rsds_values)
            values = res.rsds.values
            cccadict[nxnykey] = {}
            cccadict[nxnykey]['drsds'] = values  # fill numpy ndarray
            cccadict[nxnykey]['ddate'] = daterange365
            #cccadict[nxnykey]['geom'] = row.geometry
            #cccadict[nxnykey]['altitude'] = row.altitude
            #cccadict[nxnykey]['horizon'] = row.horizon

    return(points, cccadict)


def values_day360_day365(values):
    np.set_printoptions(threshold=np.Inf)
    valchunks = np.array_split(values, 3)
    # years = np.unique(daterange.year)
    # indleap = [31, 91, 151, 211, 271, 331]
    indnoleap = [91, 151, 211, 271, 331]
    for i in range(0, len(valchunks)):
        # if calendar.isleap(years[i]):
        #    valchunks[i] = np.insert(valchunks[i], indleap, np.nan)
        # else:
        valchunks[i] = np.insert(valchunks[i], indnoleap, np.nan)
    values = np.concatenate(valchunks)
    nans, x = nan_helper(values)
    values[nans] = np.interp(x(nans), x(~nans), values[~nans])
    return(values)


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return(np.isnan(y), lambda z: z.nonzero()[0])


def ghid2ghih(ddata, daterange, location):
    i = 0
    data = pd.DataFrame()
    while i < len(ddata):
        # raw data is in W/m2 -> this is meteorological mean data per hour and day, have to multiply by 24
        dval = ddata[i] * 24
        # print(ddata[i], dval)
        date = daterange[i]
        # sunset azimuth
        settime = sunset_time(location, date)
        solar_position = location.get_solarposition(settime)
        w_s = solar_position['azimuth'].values[0]

        # hourly azimuth
        datetimes = pd.date_range(
            start=date + datetime.timedelta(hours=0.5), end=date + datetime.timedelta(hours=23.5), freq='H', tz=config['ccca']['tz'])
        solar_position = location.get_solarposition(datetimes)
        # azimuth of sun
        w_h = np.around(solar_position['azimuth'].values, decimals=2)
        # zenith of sun
        z_h = np.around(solar_position['zenith'].values, decimals=2)
        z_h_a = np.around(solar_position['apparent_zenith'].values, decimals=2)
        cos_z_h = np.cos(np.deg2rad(z_h))
        # cos_z_h = np.where(cos_z_h > 0.08, cos_z_h, 1)

        # daily to hourly values
        ratio = rad_d2h(w_s, w_h, config['ccca']['downscale'])
        # normalize
        ratio = ratio*1/(sum(ratio))
        # ratio = np.roll(ratio, 1)
        hvalues = np.around(dval*ratio, decimals=2)
        tempdata = np.stack([w_h, z_h, z_h_a, cos_z_h, ratio, hvalues], axis=1)
        hdata = pd.DataFrame(data=tempdata, index=datetimes,
                             columns=['w_h', 'z_h', 'z_h_a', 'cos_z_h', 'ratio', 'ghi'])
        # print(hdata)
        data = data.append(hdata)
        i += 1
    return(data)


def ghi2dni(data, model='disc'):
    if model == 'disc':
        dnidata = pvlib.irradiance.disc(
            ghi=data['ghi'], solar_zenith=data['z_h'], datetime_or_doy=data.index)
        data = pd.concat([data, dnidata], axis=1)
        data.dni = data.dni.round(decimals=2)
        data['dhi'] = data.ghi - (data.dni * data.cos_z_h)
        data.kt = data.kt.round(decimals=5)

    elif model == 'erbs':
        dnidata = pvlib.irradiance.erbs(
            ghi=data['ghi'], zenith=data['z_h'], datetime_or_doy=data.index, min_cos_zenith=0.065, max_zenith=85)
        data = pd.concat([data, dnidata], axis=1)
        data.dni = data.dni.round(decimals=2)
        data.dhi = data.dhi.round(decimals=2)
        data.kt = data.kt.round(decimals=5)
    return(data)


def relative_diffuse_ratio(distance, height, tilt):
    # returns fraction of diffuse irradiance loss row-to-row diffuse shading [0-1]
    # print(distance, height)
    gcr = height/distance  # 1/k
    psi = pvlib.shading.masking_angle_passias(tilt, gcr)
    shading_loss = pvlib.shading.sky_diffuse_passias(psi)

    transposition_ratio = pvlib.irradiance.isotropic(tilt, dhi=1.0)
    total_ratio = transposition_ratio * (1-shading_loss)
    if dbg:
        typer.echo(
            f"Masking angle: {psi}, Shading loss: {shading_loss}, Transposition ratio: {transposition_ratio}, total: {total_ratio}")
    return(total_ratio, shading_loss, transposition_ratio)


def skyviewfactor(hangles):
    # calculates the sky view factor from give horizon zenith angles (as degrees and numpy vector)
    # angles: 0 = top zenith; 90 = perfect horizontal horizon
    # print(hangles)
    sin2h = np.sin(np.radians(hangles))**2
    svf = np.sum(sin2h)/len(hangles)
    return(svf)


def PVmoduleinfo():
    cecmodules = pvlib.pvsystem.retrieve_sam('CECMod')
    for name in cecmodules.columns:
        pmodule = cecmodules[name]
        if ("LG" in name) and (pmodule.Bifacial == 1):
            print(pmodule)


def pvmodeltest():
    location = pvlib.location.Location(48, 16.5)
    inverter_parameters = {'pdc0': 10000, 'eta_inv_nom': 0.96}
    module_parameters = {'pdc0': 250, 'gamma_pdc': -0.004}
    temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[
        'sapm']['open_rack_glass_glass']

    array_one = pvlib.pvsystem.Array(mount=pvlib.pvsystem.FixedMount(surface_tilt=20, surface_azimuth=200),
                                     module_parameters=module_parameters,
                                     temperature_model_parameters=temperature_model_parameters,
                                     modules_per_string=10, strings=2)
    systemarray = pvlib.pvsystem.PVSystem(arrays=[array_one],  # , array_two],
                                          inverter_parameters={'pdc0': 8000})
    # print(systemarray)
    weather = pvlib.iotools.get_pvgis_tmy(location.latitude, location.longitude,
                                          map_variables=True)[0]
    mc = pvlib.modelchain.ModelChain(systemarray, location, aoi_model='no_loss',
                                     spectral_model='no_loss')
    mc.run_model(weather)
    # , 'display.max_columns', None):
    with pd.option_context('display.max_rows', None):
        print(mc.results.ac)


def pvsystemtest():
    location = pvlib.location.Location(48, 16.5)
    cecmodules = pvlib.pvsystem.retrieve_sam('CECMod')
    cecinverters = pvlib.pvsystem.retrieve_sam('CECInverter')
    pmodule = cecmodules["LG_Electronics_Inc__LG415N2T_L5"]
    pinverter = cecinverters["LG_Electronics_Inc___D007KEEN261__240V_"]
    temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[
        'sapm']['open_rack_glass_glass']

    array_one = pvlib.pvsystem.Array(mount=pvlib.pvsystem.FixedMount(surface_tilt=20, surface_azimuth=200),
                                     module_parameters=pmodule,
                                     temperature_model_parameters=temperature_model_parameters,
                                     modules_per_string=10, strings=2)
    systemarray = pvlib.pvsystem.PVSystem(arrays=[array_one],  # , array_two],
                                          inverter_parameters=pinverter)
    print(systemarray)
    weather = pvlib.iotools.get_pvgis_tmy(location.latitude, location.longitude,
                                          map_variables=True)[0]
    mc = pvlib.modelchain.ModelChain(systemarray, location, aoi_model='no_loss',
                                     spectral_model='no_loss')
    mc.run_model(weather)
    return(mc)


def pvsystem(pvsys, location):
    cecmodules = pvlib.pvsystem.retrieve_sam('CECMod')
    cecinverters = pvlib.pvsystem.retrieve_sam('CECInverter')
    module_parameters = cecmodules[config['pvsystem'][pvsys]['module']]
    inverter_parameters = cecinverters[config['pvsystem'][pvsys]['inverter']]
    temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[
        'sapm']['open_rack_glass_glass']
    # temp_strings = 2
    # temp_modules_per_string = 5
    # mult = config['pvsystem'][pvsys]['modules_per_string'] * \
    #    config['pvsystem'][pvsys]['strings'] / \
    #    temp_strings/temp_modules_per_string
    parray = dict(
        module_parameters=module_parameters,
        temperature_model_parameters=temperature_model_parameters,
        modules_per_string=config['pvsystem'][pvsys]['modules_per_string'],
        strings=config['pvsystem'][pvsys]['strings']
        # strings_per_inverter=config['pvsystem'][pvsys]['strings']
    )
    parrays = []
    for i in range(len(config['pvsystem'][pvsys]['azimuth'])):
        tilt = config['pvsystem'][pvsys]['tilt'][i]
        azim = config['pvsystem'][pvsys]['azimuth'][i]
        parrays.append(pvlib.pvsystem.Array(pvlib.pvsystem.FixedMount(tilt, azim),
                                            name='agripv',
                                            **parray))
    system = pvlib.pvsystem.PVSystem(
        arrays=parrays, inverter_parameters=inverter_parameters)
    mc = pvlib.modelchain.ModelChain(system, location, aoi_model='physical',
                                     spectral_model='no_loss')
    return(mc)


def gethorizon(points):
    # gets horizon for landuse points
    if config['files']['hornc']:
        hornc = xarray.open_dataset(config['files']['hornc'])
    for idx, row in points.iterrows():
        if dbg:
            typer.echo(
                f"\tProcessing Points")
        if config['files']['hornc']:
            horres = get_horizon_values(hornc, row.geometry.x, row.geometry.y)
            res = horres['horizon'].values
            res = np.subtract(90, res).tolist()
        else:
            with rasterio.open(config['files']['dhm'], 'r') as ds:
                crop_dem, crop_tf = rasterio.mask.mask(
                    ds, areabuf.geometry, crop=True)

                # read all raster values
                crop_dem = crop_dem.astype(np.double)[0]
                psx, psy = ds.res
                horangles = {}
                if dbg:
                    typer.echo(
                        f"\tProcessing DEM")
                with typer.progressbar(angles) as progressangles:
                    for a in progressangles:
                        horangles[a] = horizon(a, crop_dem, psx)
                # print(horangles)
                res = []
                for a in angles:
                    py, px = ds.index(row.geometry.x, row.geometry.y)
                    res.append(
                        round(math.degrees(math.acos(horangles[a][(py, px)])), 2))
        res = json.dumps(res)
        # print(res)
        points.at[idx, 'horizon'] = [res]
    return(points)


def getcccadata(nd, points):
    dates360 = gendates360(
        config['ccca']['startyears'], config['ccca']['timeframe'])
    dates365 = gendates365(
        config['ccca']['startyears'], config['ccca']['timeframe'])
    # for other files we do not need dates360

    dates360 = dates365

    for year, daterange in dates360.items():
        daterange365 = dates365[year]
        points, cccadict = cccapoints(nd, points, daterange, daterange365)
    print(points)
    print(cccadict)
    exit(0)

    for year, daterange in dates365.items():
        for nxny in cccadict.keys():
            ddata = cccadict[nxny]['rsds']
            geom = cccadict[nxny]['geom']
            altitude = int(json.loads(cccadict[nxny]['altitude'])[0])
            location = pvlib.location.Location(
                geom.y, geom.x,
                'UTC', altitude, nxny)
            # GHI daily to GHI hourly
            df = pd.DataFrame(data=ddata)
            # df.to_csv('rad_dailyRAW.csv')
            hdata = ghid2ghih(ddata, daterange, location)
            # hdata.to_csv('hourlyraw.csv')
            hdata['location'] = geom
            # DNI+DHI hourly
            hdata = ghi2dni(hdata, config['pvmod']['hmodel'])
            #print(hdata.head(144))
    return(hdata, points)


def dhidni_horizonadaption(hdata, prow):
    phors = json.loads(prow['horizon'])
    numangles = np.arange(-180, 180, config['pvmod']['numangles'])
    for hidx, hrow in hdata.iterrows():
        # set the direct normal radiation to zero if sun is behind/below obstacle
        if (hrow['dni'] > 0 and prow['geometry'] == hrow['location']):
            # find the angle closest to the hourly azimuth of the sun (w_h)
            #print("below sun")
            idx = (np.abs(numangles - hrow['w_h'])).argmin()
            if phors[idx] < hrow['z_h']:
                hdata.at[hidx, 'dni_orig'] = hdata.at[hidx, 'dni']
                hdata.at[hidx, 'dni'] = 0
    #print(hdata.head(144))
    hangles = np.asarray(phors, dtype=float)
    svf = skyviewfactor(hangles)
    #print(svf)
    hdata.at[hidx, 'dhi_orig'] = hdata.at[hidx, 'dhi']
    return(hdata)


def simpvsystems(hdata):
    result = {}
    for pvsys in config['pvsystem'].keys():

        print(pvsys)
        [rDHI, rShade, rTransp] = relative_diffuse_ratio(config['pvsystem'][pvsys]['distance'],
                                                         config['pvsystem'][pvsys]['height'],
                                                         config['pvsystem'][pvsys]['tilt'])
        mcsys = pvsystem(pvsys, pvlib.location.Location(
            prow['geometry'].y, prow['geometry'].x, altitude=json.loads(prow['altitude'])[0]))
        mcsim = mcsys.run_model(hdata)
        # with pd.option_context('display.max_rows', None):
        res = mcsim.results.ac
        res[res < 0] = 0
        result[pvsys] = res
    return(result)

def main(t_path: Path = typer.Option(DEFAULTPATH, "--path", "-p"),
         t_areaname: str = typer.Option("BruckSmall", "--area", "-a"),
         t_configfile: Path = typer.Option(
             "cfg/testcfg.yml", "--config", "-c"),
         t_cccancfile: str = typer.Option(
             "/data/projects/PA3C3/Input/rsds_SDM_ICHEC-EC-EARTH_rcp85_r1i1p1_KNMI-RACMO22E.nc", "--cccanc", "-nc"),
         t_dhmfile: str = typer.Option(
             "/data/projects/PA3C3/Input/dhm_at_lamb_10m_2018.tif", "--dhm", "-dhm"),
         t_horfile: str = typer.Option(
             "/data/projects/PA3C3/Input/horizon_austria.nc", "--hornc", "-hor"),
         t_dhmhor: bool = typer.Option(False, "--dhmhor", "-dh"),
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

    if not config['ccca']['downscale']:
        config['ccca']['downscale'] = 'cpr'

    config['files'] = {}
    config['files']['area'] = Path(os.path.join(path, areaname, 'area.shp'))
    config['files']['cccanc'] = Path(t_cccancfile)
    config['files']['hornc'] = Path(t_horfile)
    # calculate horizon on the fly
    if t_dhmhor:
        config['files']['hornc'] = False

    if dbg:
        typer.echo(
            f"Reading area: {config['files']['area']}")
    if config['files']['area'].is_file():
        area = gpd.read_file(config['files']['area'])
        area = area.to_crs(config['gis']['processcrs'])
        # dissolve
        area['tmpdis'] = 1
        area = area.dissolve(by='tmpdis')
        #area = area.drop(columns=['tmpdis'])
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

    t_dhmfile = Path(t_dhmfile)
    if t_dhmfile.is_file():
        config['files']['dhm'] = t_dhmfile
    else:
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

    if (config['files']['hornc'] and config['files']['hornc'].is_file()):
        print("x")
    elif config['files']['hornc']:
        message = typer.style(str(config['files']['hornc']), fg=typer.colors.WHITE,
                              bg=typer.colors.RED, bold=True) + " does not exist"
        typer.echo(message)
        raise typer.Exit()

    if dbg:
        typer.echo(
            f"Selecting areas")

    lupolys, points = areaselection()
    # points = points[1:5]

    # horizon calculation for each point (angle as COS)
    if dbg:
        typer.echo(
            f"Calculation of angles for each point")
    # ds = gdal.Open(config['files']['dhm'])
    # dem = np.array(ds.GetRasterBand(1).ReadAsArray()).astype(np.double)

    areabuf = area
    areabuf['cat'] = 1
    areabuf = areabuf.dissolve(
        by='cat').geometry.convex_hull.buffer(50000)
    rs.writeGEO(areabuf, path.joinpath(
        Path.home(), 'pa3c3out'), 'area')
    
    points = gethorizon(points)
    #print(points)
    # sunrise / sunset at area center
    # readNETCDF
    if dbg:
        typer.echo(
            f"Reading CCCA data")

    hdata, points = getcccadata(nd, points)
    # print(hdata.head(48))

    # with pd.option_context('display.max_rows', None): #, 'display.max_columns', None):
    #    print(hdata)
    # exit(0)
    # idx = datetime
    if dbg:
        print('Number of points:', len(points))
    store = pd.HDFStore(path.joinpath(Path.home(),
                                      'pa3c3out',
                                      'hourly.hdf'))
    
    # iterate over all points in the area
    for pidx, prow in points.iterrows():
        hdata = dhidni_horizonadaption(hdata, prow)
        result = simpvsystems(hdata, prow)
    
        #store[str(prow['geometry'].y) + "-" +
        #      str(prow['geometry'].x)] = res
        #res.to_csv(path.joinpath(Path.home(), 'pa3c3out', str(prow['geometry'].y) + "-" +
        #                                      str(prow['geometry'].x) + '.csv'))

    store.close()

    # print(mcsim.results)
    # statistics for hdata
    print(result)
    exit(0)
    res = res * 10
    res = res.reset_index(name='kWh')
    # print(res.head(144))
    res = res.set_index('index')
    # print(res.head(144))
    res.to_csv('hdata.csv')
    # res = res.rename('kWh')
    daily = res.resample('D').sum()/1000
    # print(daily.head(365))
    daily.to_csv('ddata.csv')
    monthly = res.resample('M').sum()/1000

    monthly.to_csv('mdata.csv')

    # print(hdata.head(144))

    ahourly = hdata.groupby((hdata.index.dayofyear - 1) *
                            24 + hdata.index.hour).ghi.mean()
    # print(ahourly)
    ahourly.to_csv('ahourly.csv')

    rad_monthly = hdata.groupby(hdata.index.month).ghi.mean()
    rad_monthly.to_csv('rad_monthly.csv')

    amonthly = monthly.groupby(monthly.index.month).kWh.mean()
    amonthly.to_csv('amdata.csv')
    print(amonthly.head(12))
    # print(hourly)
    # print(daily.head(365))

    # output for EPIC

    # hdata.to_csv(path.joinpath(Path.home(), 'pa3c3out', 'hdata.csv'))
    # ddata = pd.DataFrame(data=ddata)
    # ddata.to_csv(path.joinpath(Path.home(), 'pa3c3out', 'ddata.csv'))
    rs.writeGEO(lupolys, path.joinpath(Path.home(), 'pa3c3out'), 'PVlupolys')
    rs.writeGEO(points, path.joinpath(Path.home(), 'pa3c3out'), 'PVpoints')

    typer.echo("finished")


if __name__ == "__main__":
    typer.run(main)
