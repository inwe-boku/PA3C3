import pandas as pd
import geopandas as gpd
import numpy as np
import scipy as sp
import pyproj
import shapely
from osgeo import osr
from osgeo import gdal
import math
import statistics
import itertools
import re
import os
import matplotlib.pyplot as plot


import yaml
import pathlib
import urllib.parse
import requests
from io import StringIO

import json

import struct
import osmnx as ox

import inspect
import operator

import logging

def nearestpolygons(matchpolygons, basepolygons, neighbor=1):
    """
    Check which polygon from the first argument is nearest to
    one from the second based on the centroids.

    Parameters
    ----------
    matchpolygons : geopandas geoseries of polygons
    basepolygons : geopandas geoseries of polygons
    neighbor : int, optional
        returns the x-th nearest neighbor

    Returns
    -------
    matchpolygons : geomandas geoseries of polygons
        with new columns 'nidx' and 'ndst', the index and distance (m)
    """
    matchpoints = nearestpoints(matchpolygons['geometry'].centroid,
                                basepolygons['geometry'].centroid,
                                neighbor)
    matchpolygons['nidx'] = matchpoints['nidx']
    matchpolygons['ndst'] = matchpoints['ndst']
    return(matchpolygons)


def nearestpoints(matchpoints, basepoints, neighbor=1):
    """
    Check which polygon from the first argument is nearest to
    one from the second based on the centroids.

    Parameters
    ----------
    matchpoints : geopandas geoseries of points
    basepoints : geopandas geoseries of points
    neighbor : int, optional
        returns the n-th nearest neighbor

    Returns
    -------
    matchpoints : geomandas geoseries of points
        with new columns 'nidx' and 'ndst', the index and distance (m)
    """
    nA = np.array([matchpoints['geometry'].x,
                   matchpoints['geometry'].y]).T
    nB = np.array([basepoints['geometry'].x,
                   basepoints['geometry'].y]).T
    btree = sp.spatial.cKDTree(nB)
    dst, idx = btree.query(nA, k=[neighbor])
    matchpoints['nidx'] = idx
    matchpoints['ndst'] = dst
    return(matchpoints)


def haversine_distance(pointa, pointb):
    """
    Get the distance of two points (lat/lon as in WGS)
    using the the haversine formula in meter.

    Parameters
    ----------
    pointa : shapely point
    pointb : shapely point

    Returns
    -------
    distance : int
        distance between the two points (m)
    """
    R = 6371000  # average earth radius in meter
    dlat = math.radians(pointb.y - pointa.y)
    dlon = math.radians(pointb.x - pointa.x)
    a = (math.sin(dlat / 2)
         * math.sin(dlat / 2)
         + math.cos(math.radians(pointa.y))
         * math.cos(math.radians(pointb.y))
         * math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return(R * c)


def metre_per_degree(point):
    """
    Return the number of meters lat (y), lon (x) per degree
    lat (y) and lon (x) at a certain point on earth. If the
    points crs is aready in meters (1, 1) is returned.

    Parameters
    ----------
    point : shapely point

    Returns
    -------
    (mlat, mlon) : int tuple
        (metre lat per degree lat, metre lon per degree lon)
    """
    crs = pyproj.CRS(point.crs)
    if ('m' in crs.axis_info[0].unit_name):
        return(1, 1)
    mlat = haversine_distance((point.x, point.y - 0.5),
                              (point.x, point.y + 0.5))
    mlon = haversine_distance((point.x - 0.5, point.y),
                              (point.x + 0.5, point.y))
    return(mlat, mlon)

def pp_compactness(geom): # Polsby-Popper
    p = geom.length
    return (4*math.pi*geom.area)/(p*p)
    
def s_compactness(geom): # Schwartzberg
    p = geom.length
    a = geom.area    
    return 1/(geom.length/(2*math.pi*math.sqrt(geom.area/math.pi)))


def nlatlon_gridpoints(polygon, resolution=100):
    """
    Get the number regular grid points neccessary at a given resolution (m)
    for a polygon as lat (y), lon (x)

    Parameters
    ----------
    polygon : geopandas polygon
    resolution : integer, optional
        resolution of grid in meters, default is 100

    Returns
    -------
    (numlat, numlon) : int tuple
        (number of points lat, number of points lon)
    """
    mlat, mlon = metre_per_degree(polygon.centroid)
    bbox = polygon.bounds
    dlat = int((bbox.maxy - bbox.miny) * mlat)
    dlon = int((bbox.maxx - bbox.minx) * mlon)
    return(math.ceil(dlat / resolution),
           math.ceil(dlon / resolution))

def pointraster(polygons, resolution=100, sareaID=[], crsl='epsg:4087'):
    """
    Returns a geodataframe with regular gridpoints for a (one) polygon at a
    given resolution (m). If more than one polygon exists in the input data it
    will be dissolved temporarily to prepare the grid.

    Parameters
    ----------
    polygon: geopandas geoseries polygons
    resolution : integer
        resolution of grid in meters, default: 100
    sareaID : list of columns to be preserved, default: []
    crsl : proj4 CRS string
        epsg to do the length calculations, default: 'epsg:4087'

    Returns
    -------
    points: geopandas geoseries points
    """
    polygons['tempid'] = 1
    polycrs = polygons.crs
    aggpoly = polygons.dissolve(by='tempid')
    aggpoly = aggpoly.to_crs(crsl)
    nlat, nlon = nlatlon_gridpoints(aggpoly, resolution=resolution)
    lats = list(np.linspace(float(aggpoly.bounds.miny),
                            float(aggpoly.bounds.maxy),
                            nlat))
    lons = list(np.linspace(float(aggpoly.bounds.minx),
                            float(aggpoly.bounds.maxx),
                            nlon))
    logging.info('resolution: {:.2f} m;\n\
                \tgridsize: {:d}x{:d}\n\
                \tpoints: {:d}'.format(resolution, nlat, nlon, nlat * nlon))

    lats = np.repeat(lats, nlon)
    lons = np.tile(lons, nlat)
    df = pd.DataFrame({'Latitude': lats, 'Longitude': lons})
    points = gpd.GeoDataFrame(df,
                            geometry=gpd.points_from_xy(df.Longitude,
                                                        df.Latitude),
                            crs=crsl)
    points = points.to_crs(crs=polycrs)
    points = points[['geometry']]
    aggpoly = aggpoly.to_crs(crs=polycrs)
    points = gpd.sjoin(points, polygons, how='left', op='within')
    points = points.dropna()
    if points.empty:
        df = pd.DataFrame({'Latitude': aggpoly.centroid.y, 'Longitude': aggpoly.centroid.x})
        points = gpd.GeoDataFrame(df,
                              geometry=gpd.points_from_xy(df.Longitude,
                                                          df.Latitude),
                              crs=polycrs)
    points = points.reset_index()
    points = points[['geometry'] + sareaID]
    logging.info('Sampling points in area: {:d}'.format(len(points)))
    return(points)


def samplerasterpoints(points, rasterfile,
                       fieldname='alt', samplemethod=1, crsl='epsg:4087'):
    """
    Takes geodataframe points and samples values from a rasterfile,
    a new column is created with and filled with values.

    Parameters
    ----------
    points : geopandas geoseries points
    rasterfiles : list of strings
        list of filenames to the rasterfiles
    fieldname : string
        columnname for sampleresults
    samplemethod : int
        1: sample one point (mid,mid)
        2: sample  5 points (1 + 1/2 north, east, south, west)
        3: sample 13 points (1 + 1/3 N, E, S, W +
                             2/3 N NE, E, SE, S, SW, W, NW)
    crsl: epsg for distance measurement (default: 4087)
        fieldname: column name for the sampling result

    Returns
    -------
    points: geopandas geoseries points
        with new column fieldname and sampleresults as string (json list)
    """
    dist = 10
    if len(points) > 1:
        pts = points.to_crs(crsl)
        dist = int(nearestpoints(pts, pts, neighbor=2)['ndst'].mean() / 2)
    #vrt = gdal.BuildVRT("/vsimem/temp.vrt", rasterfiles) # build virtual raster from several files
    #gdal.Translate('/vsimem/myvirt.vrt', vrt, format='VRT')
    #vrt = None # close raster again, necessary step
    ds = gdal.Open(str(rasterfile)) # open as raster
    gt = ds.GetGeoTransform()
    rb = ds.GetRasterBand(1)
    proj = osr.SpatialReference(wkt=ds.GetProjection())
    #if dbg:
    print('rastercrs: ', proj.GetAttrValue('AUTHORITY',1))
    bbox = shapely.geometry.polygon.Polygon([
        (points.bounds.miny.min(), points.bounds.minx.min()),
        (points.bounds.miny.min(), points.bounds.maxx.max()),
        (points.bounds.maxy.max(), points.bounds.minx.min()),
        (points.bounds.maxy.max(), points.bounds.maxx.max())])
    poly = gpd.GeoDataFrame(geometry=[bbox])
    poly = poly.set_crs(points.crs)
    mlat, mlon = metre_per_degree(poly)
    tups = [(0, 0)]
    if (samplemethod == 2):
        values = [1 / 2, -1 / 2]
        tups.extend(list(itertools.product(values, values)))
    elif (samplemethod == 3):
        values = [1 / 3, -1 / 3]
        tups.extend(list(itertools.product(values, values)))
        values = [0, 2 / 3, -2 / 3]
        tups.extend(list(itertools.product(values, values))[1:])
    points[fieldname] = np.nan
    for idx, row in points.iterrows():
        logging.info('sampling %s for point %d of %d',
                    fieldname, int(format(idx + 1)), int(len(points)))
        intlist = []
        for tup in tups:
            py = int((row.geometry.y +
                      tup[0] * mlat * dist -
                      gt[3]) / gt[5])
            px = int((row.geometry.x +
                      tup[1] * mlon * dist -
                      gt[0]) / gt[1])
            intval = int(struct.unpack(
                  'h', rb.ReadRaster(px, py, 1, 1,
                                     buf_type=gdal.GDT_UInt16))[0])
            intlist.append(intval)
        intlist = json.dumps(intlist)
        points[fieldname] = points[fieldname].astype(str)
        points.at[idx, fieldname] = intlist
        print(points)
    return(points)


def analysevector(gdf, infield='alt', outfield='res_alt', op='lt',
                  cmp='multi', vals=[2000], perc=0.6):
    """
    Takes a geodataframe and cross-checks values from the fieldname
    with a given operator and values

    Parameters
    ----------
    gdf : geopandas geoseries
    fieldname : string
        columnname with values (json list as string)
    op : string
        operator as 'lt', 'le', 'eq', 'ne', 'ge', 'gt',
        'contains', 'countOf'
    cmp : string
        either 'multi' for multi comparisons and a percentage calculation
        or one of the following: 'min', 'max', 'mean', 'median', 'sum'
    vals : list
    perc : integer

    Returns
    -------
    gdf: geopandas geoseries
        with a new column
    """
    #logger.info("point analysis for fieldname: %s", infield)
    funs = {'lt': operator.lt,
            'le': operator.le,
            'eq': operator.eq,
            'ne': operator.ne,
            'ge': operator.ge,
            'gt': operator.gt,
            'contains': operator.contains,
            'countOf': operator.countOf}
    cmps = {'min': min, 'max': max, 'sum': sum,
            'mean': statistics.mean, 'median': statistics.median}
    cmpslist = list(cmps.keys())
    singleops = ['lt', 'le', 'eq', 'ne', 'ge', 'gt']
    for idx, row in gdf.iterrows():
        result = False
        if op == 'countOf':
            result = []
        count = 0
        dats = json.loads(row[infield])
        if cmp in cmpslist:
            dats = [cmps[cmp](dats)]
        for dat in dats:
            if op == 'countOf':
                result.append(funs[op](vals, dat))
            if op == 'contains':
                if funs[op](vals, dat):
                    count += 1
            if op in singleops:
                if funs[op](dat, vals[0]):
                    count += 1
        if op != 'countOf':
            if (count/len(dats) >= perc):
                result = True
        else:
            result = json.dumps(result)
        #logger.info("comparison: %s %s %s, result: %s of %s >= %s is %s",
        #            dats, op, vals, count, len(dats), perc, result)
        gdf.at[idx, outfield] = result
    return(gdf)


def fetchrendata(gdf, rencfg, fieldname='res_alt', fieldvals=(1, True)):
    """
    fetch data of fieldname == True / 1
    """
    name = rencfg['rencfg']['name']
    parameter = rencfg['rencfg']['parameter']
    outputformat = rencfg['rencfg']['outputformat']

    gdft = gdf.to_crs(rencfg['rencfg']['crs'])
    crs = pyproj.CRS(gdft.crs)

    dist = nearestpoints(gdft, gdft, neighbor=2)['ndst'].mean() / 2
    if ('m' not in crs.axis_info[0].unit_name):
        mlat, mlon = metre_per_degree(gdf.centroid)
        dist = 2 * dist/(mlat + mlon)
    for idx, row in gdft.iterrows():
        if (row[fieldname] in fieldvals):
            latlon = {rencfg['rencfg']['latlon'][0]:
                      '{:.4f}'.format(row['geometry'].y),
                      rencfg['rencfg']['latlon'][1]:
                      '{:.4f}'.format(row['geometry'].x)}
            params = {**parameter, **latlon}
            r = requests.get(rencfg['rencfg']['urlbase'], params=params)
            if r.status_code != 200:  # exit if http response is other than 200
                break
            if 'no valid' in r.text:
                latlon['lat'] = float(latlon['lat']) + 0.0005
                latlon['lon'] = float(latlon['lon']) + 0.0005
            if outputformat == 'json':
                print(r.json()[rencfg['rencfg']['outputloc']])
            elif outputformat == 'csv':
                print(r.text)

def getyml(filename):
    """
    loads a yaml file and returns it as an array containing lists, dicts, etc.

    Parameters
    ----------
    filename: string
        columnname with values (json list as string)

    Returns
    -------
    config: array
    """
    with open(filename, 'r') as stream:
        try:
            return(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

def getgufnames(polygon, crsl = 'epsg:4326'):
    """
    returns the filename for rasterfiles following a naming scheme of GUF by DLR
    """
    def lonsub(lon):
        if lon >= 0:
            return('e'+str(lon).zfill(3))
        else:
            return('w'+str(abs(lon)).zfill(3))

    def latsub(lat):
        if lat >= 0:
            return('n'+str(lat).zfill(2))
        else:
            return('s'+str(abs(lat)).zfill(2))

    polygon = polygon.to_crs(crsl)
    lon1 = float(polygon.bounds.minx)
    lon2 = float(polygon.bounds.maxx)
    lat1 = float(polygon.bounds.miny)
    lat2 = float(polygon.bounds.maxy)

    rlon = list(range(5*math.floor(lon1/5), 5*math.ceil(lon2/5)+5, 5))
    rlat = sorted(list(range(5*math.floor(lat1/5), 5*math.ceil(lat2/5)+5, 5)), reverse = True)
    rlon = list(map(lonsub, rlon))
    rlat = list(map(latsub, rlat))

    i = 0
    names = []
    while i < len(rlon)-1:
        j = 0
        while j < len(rlat)-1:
            names.append('GUF04_DLR_v02_' + rlon[i] + '_' + rlat[j] + '_' +
                     rlon[i+1] + '_' + rlat[j+1] +
                     '_OGR04.tif')
            j += 1
        i += 1

    return(names)

def getghsnames(polygons, ghspolys):
    """
    returns the filename for rasterfiles following a naming scheme of GHS/ESM by JRC
    """
    namepolys = gpd.sjoin(ghspolys, polygons) #, how='left', op='intersects')
    return(namepolys[['location', 'index_right']])

def writeGEO(data, path, dataname):
    #data.to_file(filename = os.path.join(path, 'geojson', dataname+'.geojson'), driver="GeoJSON")
    #data.to_file(filename=os.path.join(path, dataname + '.shp'),
    #             driver='ESRI Shapefile')
    data.to_file(filename = os.path.join(path, 'data.gpkg'), layer = dataname, driver = 'GPKG')
    return(0)

def getosmbuildings(area, fbuildings):
    buildings = gpd.GeoDataFrame()
    bdsample = gpd.GeoDataFrame(columns=['geometry', 'KG_NR', 'fparea'])
    for idx, row in area.iterrows():  # get building area per municipality
        print('\tOSM fetch {:d} of {:d}'.format(idx + 1, len(area)), end='\r')
        if fbuildings.empty == False:
            rgdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(row[['geometry']]))
            rgdf.crs = area.crs
            bd = gpd.sjoin(fbuildings, rgdf, how="inner", op='intersects')
            bd = bd.dropna()
            bd = bd.reset_index(drop=True)
            bd.crs = rgdf.crs
        else:
            bd = ox.footprints_from_polygon(row.geometry)
        bd = bd[['geometry']]
        bd = bd.to_crs({'init': 'epsg:6933'})
        bd['fparea'] = bd['geometry'].area.astype(int)
        bd = bd.to_crs({'init': 'epsg:4326'})
        bd['KG_NR'] = row['KG_NR']
        ssize = round(2.5 * len(bd)**0.50)  # int(len(bd)/math.sqrt(len(bd))
        ssize = min(len(bd), ssize)
        ssize = min(len(bd), 10)
        bds = bd.sample(ssize)
        bd = nearestbld(bds, bd)
        bdj = bd.groupby([bd.nearest]).sum()
        bds['index'] = bds.reset_index().index
        bds = bds.set_index('index').join(bdj, rsuffix='r')
        bds = bds[['geometry', 'KG_NR', 'fparear']]
        bds.rename(columns={'fparear': 'fparea'}, inplace=True)
        bdsample = pd.concat([bdsample, bds])
        buildings = pd.concat([buildings, bd])

    buildings['KG_NR'] = pd.to_numeric(buildings.KG_NR)
    buildings['fparea'] = pd.to_numeric(buildings.fparea)
    bdsample['KG_NR'] = pd.to_numeric(bdsample.KG_NR)
    bdsample['fparea'] = pd.to_numeric(bdsample.fparea)
    bdsample.crs = {'init': 'epsg:4326'}
    bdsample = bdsample.reset_index(drop=True)
    return(buildings, bdsample)


def calcfreepv(pointspv, urloptions, urlbase, seasons, store, timeres):
    count = 0
    ln = len(pointspv[pointspv['type'] == 'free'])

    for idx, row in pointspv.iterrows():
        fails = 0
        if row.type != 'free':
            continue
        count += 1
        print('\tPV data {:d} of {:d}'.format(count, ln), end='\r')
        if dbg:
            print('\n')
        lonlat = {'lat': '{:.4f}'.format(row['geometry'].y),
                  'lon': '{:.4f}'.format(row['geometry'].x)}
        if dbg:
            print(lonlat)
        urloptions['mountingplace'] = 'free'
        dataname = 'C' + 'x'.join("{!s}".format(str(v).replace('.', '_')) for (k, v) in lonlat.items())
        if dbg:
            print(dataname)
        try:
            pvdata = store[dataname]
            if dbg:
                print('existing, loaded from file')
        except BaseException:
            if dbg:
                print('not here, downloading from JRC')
            pvdata, fails = getPVJRC(urlbase, urloptions, lonlat, timeres)
            store[dataname] = pvdata

        if pvdata.empty:
            pointspv.at[idx, 'type'] = 'nodata'
            pointspv.at[idx, 'fail'] = fails
        else:
            pvdata['year'] = pvdata['Date'].dt.year
            pvdata['month'] = pvdata['Date'].dt.month
            pvdata['day'] = pvdata['Date'].dt.day

            df = pvdata.groupby([pvdata.month]).sum()
            nryears = len(pvdata['Date'].dt.year.unique())
            for s in seasons:
                if timeres == 'daily':
                    kWhpkWp = int(
                        df.loc[seasons[s]].Wh.mean() / 1000 * 12 / nryears)
                elif timeres == 'monthly':
                    kWhpkWp = int(df.loc[seasons[s]].Wh.mean() * 12)
                if dbg:
                    print(s, ': ', kWhpkWp, 'kWh/a per kWp')
                fname1 = 'kWhp_' + s
                fname2 = 'kWhm2_' + s
                if s == 'all':
                    fname1 = 'kWhp'
                    fname2 = 'kWhm2'
                pointspv.at[idx, fname1] = kWhpkWp
                pointspv.at[idx, fname2] = int(
                    kWhpkWp / cfg['landuse']['m2kWp'])
                pointspv.at[idx, 'fail'] = fails
    return(pointspv)


def calcbldpv(bdsample, area, urloptions, urlbase, seasons, store, timeres):
    count = 0
    ln = len(bdsample)

    for idx, row in bdsample.iterrows():
        count += 1
        print('\tPV data {:d} of {:d}'.format(count, ln), end='\r')
        if dbg:
            print()
            print()
        lonlat = {'lat': '{:.4f}'.format(row['geometry'].centroid.y),
                  'lon': '{:.4f}'.format(row['geometry'].centroid.x)}
        if dbg:
            print(lonlat)
        urloptions['mountingplace'] = 'building'
        url = urlbase + \
            urllib.parse.urlencode({**lonlat, **urloptions}).strip("'")

        dataname = 'C' + 'x'.join("{!s}".format(str(v).replace('.', '_')) for (k, v) in lonlat.items())
        if dbg:
            print(dataname)
        try:
            pvdata = store[dataname]
            if dbg:
                print('existing, loaded from file')
        except BaseException:
            if dbg:
                print('not here, downloading from JRC')
            pvdata = getPVJRC(url, timeres)
            store[dataname] = pvdata

        pvdata['year'] = pvdata['Date'].dt.year
        pvdata['month'] = pvdata['Date'].dt.month
        pvdata['day'] = pvdata['Date'].dt.day
        df = pvdata.groupby([pvdata.month]).sum()
        nryears = len(pvdata['Date'].dt.year.unique())
        for s in seasons:
            if timeres == 'daily':
                kWhpkWp = int(
                    df.loc[seasons[s]].Wh.mean() / 1000 * 12 / nryears)
            elif timeres == 'monthly':
                kWhpkWp = int(df.loc[seasons[s]].Wh.mean() * 12)
            if dbg:
                print(s, ': ', kWhpkWp, 'kWh/a per kWp')
            fname1 = 'kWhp_' + s
            fname2 = 'kWhm2_' + s
            fname3 = 'kWh_' + s
            if s == 'all':
                fname1 = 'kWhp'
                fname2 = 'kWhm2'
                fname3 = 'kWh'
            bdsample.at[idx, fname1] = kWhpkWp
            bdsample.at[idx, fname2] = int(kWhpkWp / cfg['buildings']['m2kWp'])
            bdsample.at[idx, fname3] = int(
                kWhpkWp / cfg['buildings']['m2kWp'] * row['fparea'] * cfg['buildings']['rPV'])
    bds = bdsample.filter(regex='KG_NR|fparea|kWh$|kWh_')
    dfs = bds.groupby([bds.KG_NR]).sum()
    bds = bdsample.filter(regex='KG_NR|kWhp|kWhm2')
    dfm = bds.groupby([bds.KG_NR]).mean()
    areabuildingspv = area
    areabuildingspv['KG_NR'] = pd.to_numeric(areabuildingspv.KG_NR)
    areabuildingspv = areabuildingspv.set_index(
        'KG_NR').join(dfs, rsuffix='rs')
    areabuildingspv = areabuildingspv.join(dfm, rsuffix='rm')
    return(bdsample, areabuildingspv)

# @click.command()
# @click.option('-resolution', '-r', help = 'output resolution in km x km', default = 0.25, type=float)
# @click.option('-areaname', '-a', help = 'specify the name of the area', default = 'nockberge', type = str)
# @click.option('-inpdir', '-i', help = 'input directory', default = 'Data/input', type = str)
# @click.option('-outdir', '-o', help = 'output directory', default = 'Data/output', type = str)
# @click.option('-pvddir', '-p', help = 'raw pv data directory', default = 'Data/pvdata', type = str)
# @click.option('-landusefile', '-l', help = 'land use file (optional)', default = '', type = str)
# @click.option('-elevationfile', '-e', help = 'elevation file (optional)', default = '', type = str)
# @click.option('-osmbld', '-b', help = 'buildings from OSM (optional)', default = False, type = bool)
# @click.option('-configfile', '-c', help = 'configuration file', default = 'retour.yml', type = str)
# @click.option('-jrcconfig', '-j', help = 'jrc download config file', default = 'pvjrc-monthly.yml', type = str)
# @click.option('-debug', '-d', help = 'debug output', default = False, type=bool)


def main(resolution, areaname, inpdir, outdir, pvddir, landusefile,
         elevationfile, osmbld, configfile, jrcconfig, debug):
    # pd.options.mode.chained_assignment = None
    # config = getyml(
    #     os.path.join(
    #         os.path.dirname(
    #         os.path.realpath(__file__)),
    #         configfile))
    # jrccfg = getyml(
    #     os.path.join(
    #         os.path.dirname(
    #         os.path.realpath(__file__)),
    #         jrcconfig))
    # areafile = os.path.join(inpdir, areaname, 'area.shp')
    # area = gpd.read_file(areafile)

    # global dbg
    # global cfg
    # global jrc
    # dbg = debug
    # cfg = config
    # jrc = jrccfg
    # timeres = str(jrc['timeres'])
    # outdirarea = os.path.join(outdir, areaname)

    # if dbg:
    #     print(areafile)
    # pathlib.Path(
    #     os.path.join(
    #         outdir,
    #         areaname,
    #         'geojson')).mkdir(
    #         parents=True,
    #     exist_ok=True)
    # pathlib.Path(
    #     os.path.join(
    #         outdir,
    #         areaname,
    #         'shape')).mkdir(
    #         parents=True,
    #     exist_ok=True)
    # pathlib.Path(os.path.join(pvddir)).mkdir(parents=True, exist_ok=True)
    # writeGEO(area, outdirarea, 'dbg_area')

    # if dbg:
    #     print(jrc['options'])

    # points = pointraster(resolution, area)
    #writeGEO(points, outdirarea, 'dbg_points')

    # if landusefile:
    #     if elevationfile:

    #         try:
    #             maxalt = int(cfg['landuse']['maxalt'])
    #         except BaseException:
    #             maxalt = 0
    #         if (maxalt > 0):
    #             print('')
    #             print('Processing Points Altitude: <', maxalt, 'masl')
    #             points = checkrasterint(
    #                 points, resolution, elevationfile, maxalt, 'altitude')

    #         try:
    #             maxslope = int(cfg['landuse']['maxslope'])
    #         except BaseException:
    #             maxslope = 0
    #         if (maxslope > 0):
    #             print('')
    #             print('Processing Points Slope: <', maxslope, ' degrees')
    #             opts = gdal.DEMProcessingOptions(scale=111120)
    #             slopefile = '/tmp/slope.tif'
    #             gdal.DEMProcessing(
    #                 slopefile, elevationfile, 'slope', options=opts)
    #             points = checkrasterint(
    #                 points, resolution, slopefile, maxslope, 'slope')

    #         writeGEO(points, outdirarea, 'dbg_pointsaltslp')
    #     print('')
    #     print('Processing Points LU')
    #     pointslu = getfreelandpoints(
    #         points,
    #         resolution,
    #         landusefile,
    #         cfg['landuse']['free'])
    #     writeGEO(pointslu, outdirarea, 'dbg_pointslu')
    #     print('')
    #     store = pd.HDFStore(os.path.join(pvddir, 'free_' +
    #                                      str(jrc['options']['raddatabase']) + '_' +
    #                                      str(jrc['options']['startyear']) + '_' +
    #                                      str(jrc['options']['endyear']) + '_' +
    #                                      timeres + '.hdf5'))
    #     freepvpts = calcfreepv(
    #         pointslu,
    #         jrc['options'],
    #         jrc['url'],
    #         cfg['seasons'],
    #         store,
    #         timeres)
    #     store.close()
    #     writeGEO(freepvpts, outdirarea, 'freepvpts')

    # if osmbld:
    #     print('')
    #     print('Processing buildings')
    #     bldfile = os.path.join(inpdir, areaname, 'buildings.shp')
    #     try:
    #         buildings = gpd.read_file(bldfile)
    #         fromfile = True
    #     except BaseException:
    #         buildings = gpd.GeoDataFrame()
    #         fromfile = False
    #     buildings, bdsample = getosmbuildings(area, buildings)
    #     print('')
    #     writeGEO(buildings, outdirarea, 'dbg_buildings')
    #     store = pd.HDFStore(os.path.join(pvddir, 'building_' +
    #                                      str(jrc['options']['raddatabase']) + '_' +
    #                                      str(jrc['options']['startyear']) + '_' +
    #                                      str(jrc['options']['endyear']) + '_' +
    #                                      timeres + '.hdf5'))
    #     bdsample, areabuildingspv = calcbldpv(
    #         bdsample, area, jrc['options'], jrc['url'], cfg['seasons'], store, timeres)
    #     store.close()
    #     writeGEO(bdsample, outdirarea, 'dbg_bdsample')
    #     writeGEO(areabuildingspv, outdirarea, 'areabuildingspv')
    #     print('')
    print('done')


#if __name__ == "__main__":
#    main()

