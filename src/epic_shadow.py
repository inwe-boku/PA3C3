#!/usr/bin/env python3

import typer
from pathlib import Path
import json
import yaml
import os
import datetime
import renspatial as rs
import pandas as pd
import dask
import dask.dataframe as ddf
import numpy as np
import pvlib
import suntimes
import math
from calendar import isleap
import timeit

__author__ = "Christian Mikovits"

GRIDFILE = '/data/projects/PA3C3/EPICOKS15/A_Infos/Grid info/OKS15_AT_geodata_new.txt'
DLYDIR = '/data/projects/PA3C3/EPICOKS15/ICHEC45_new'
DLYDIRMOD = '/data/projects/PA3C3/EPICOKS15/ICHEC45_new_FH'
#BBOX = [48.18, 13.78, 48.36, 14.11]
BBOX = [0, 0, 100, 100]  # [48.18, 13.78, 48.36, 14.11]
WORKERS = 16

pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)


def PVmoduleinfo(modulename):
    cecmodules = pvlib.pvsystem.retrieve_sam('CECMod')
    for name in cecmodules.columns:
        pmodule = cecmodules[name]
        if (name == modulename):
            return(pmodule)


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


def ghid2ghih(ddata, location):
    years = ddata[0].unique()
    hvals = hourly_solarangles(location, leap=False)
    lhvals = hourly_solarangles(location, leap=True)

    hdata = pd.DataFrame()
    cl = 0
    for y in years:
        if isleap(y):
            lhvals.index = lhvals.index.map(lambda x: x.replace(year=y))
            hdata = pd.concat([hdata, lhvals])
        else:
            hvals.index = hvals.index.map(lambda x: x.replace(year=y))
            hdata = pd.concat([hdata, hvals])

    dvals = ddata[3].values * 1000 / 3.6
    hvals = np.repeat(dvals, 24)

    hdata['ghi'] = hvals * hdata.ratio
    et = datetime.time()
    return(hdata)


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

# read daily value from ASCII and transform to hourly

# reduce hourly by shadow and transform to daily

# write ASCII


def hourly_solarangles(location, leap=False):
    data = pd.DataFrame()
    year = 1999
    if leap:
        year = 2000

    sd = datetime.datetime(year, 1, 1)
    ed = datetime.datetime(year, 12, 31)
    dates = pd.date_range(
        start=sd, end=ed, freq='D', tz=config['ccca']['tz'])
    ddata = pd.DataFrame(dates, columns=['date'])
    for idx, row in ddata.iterrows():
        settime = sunset_time(location, row['date'])
        solar_position = location.get_solarposition(settime)
        w_s = solar_position['azimuth'].values[0]
        # hourly azimuth
        datetimes = pd.date_range(
            start=row['date'] + datetime.timedelta(hours=0.5), end=row['date'] + datetime.timedelta(hours=23.5), freq='H', tz=config['ccca']['tz'])
        solar_position = location.get_solarposition(datetimes)
        # azimuth of sun
        w_h = np.around(solar_position['azimuth'].values, decimals=2)
        # zenith of sun
        z_h = np.around(solar_position['zenith'].values, decimals=2)
        z_h_a = np.around(solar_position['apparent_zenith'].values, decimals=2)
        cos_z_h = np.cos(np.deg2rad(z_h))
        cos_z_h = np.where(cos_z_h > 0.08, cos_z_h, 1)

        # daily to hourly values
        # print(config['ccca'])
        ratio = rad_d2h(w_s, w_h, config['ccca']['downscale'])
        # normalize
        ratio = ratio*1/(sum(ratio))
        ratio = np.roll(ratio, 1)
        # hvalues = np.around(dval*ratio, decimals=2)
        tempdata = np.stack([w_h, z_h, z_h_a, cos_z_h, ratio], axis=1)
        hdata = pd.DataFrame(data=tempdata, index=datetimes,
                             columns=['w_h', 'z_h', 'z_h_a', 'cos_z_h', 'ratio'])
        data = pd.concat([data, hdata])
        # i += 1
    return(data)


def itergrid(csvr):
    #    idx, row = arg
    starttime = timeit.default_timer()

    #print('processing', csvr)

    location = pvlib.location.Location(
        csvr['longitude'], csvr['latitude'],
        'UTC', csvr['Elev'], csvr['Identity'])
    dlyfn = os.path.join(DLYDIR, csvr['Identity'] + '.dly')
    dlycsv = gridcsv = pd.read_csv(dlyfn, sep='\s+', header=None)
    dlycsv['date'] = pd.to_datetime(
        dlycsv[0]*10000+dlycsv[1]*100+dlycsv[2], format='%Y%m%d')

    # print(dlycsv.head())

    hdata = ghid2ghih(dlycsv, location)
    hdata = ghi2dni(hdata, config['pvmod']['hmodel'])
    pvsystem = config['pvsystem'][0]
    #print(pvsystem)

    if pvsystem['lengthwidth'] == 'width':
        module_lw = PVmoduleinfo(pvsystem['module'])['Width']
    else:
        module_lw = PVmoduleinfo(pvsystem['module'])['Length']

    footprint = round(
        (module_lw * math.cos(math.radians(pvsystem['tilt'][0]))), 2)

    # zenith shadow calc
    #
    # the 'footprint' of the module on the ground
    # is independent of the zenith and always the same

    # the shade of the module height varies with the zenith
    z_b = round(pvsystem['height'] / np.tan(np.radians(90-(hdata['z_h']))), 2)

    # shadefree length: z_c = pvsystem['moduledist'] - hdata['z_b']

    # zenith angle shade relation; 1 = full shade, 0 = no shade
    hdata['z_rel'] = round((footprint + abs(z_b)) /
                           abs(pvsystem['moduledist'] - z_b), 2)
    hdata.loc[hdata.z_rel > 1, 'z_rel'] = 1

    # azimuth shadow calc
    # relation shading the sun; 1 = full shade, 0 = no shade
    hdata['w_rel'] = round(
        abs(np.cos(np.radians(pvsystem['azimuth'][0] - hdata['w_h']))), 2)

    # combination of zenith and azimuth shading
    hdata['s_rel'] = round(hdata.z_rel*hdata.w_rel, 2)

    # reduction of dni
    hdata['dni_red'] = hdata.dni * (1 - hdata.s_rel)

    # reduction of dhi by the skyviewfactor
    hdata['dhi_red'] = hdata.dhi * pvsystem['svf']

    # combine dni and dhi to ghi

    hdata['ghi_red'] = hdata['dni_red']*hdata['cos_z_h'] + hdata['dhi_red']
    hdata['MJpm2'] = hdata['ghi_red'] / 1000 * 3.6
    #print("The time difference is :", timeit.default_timer() - starttime)
    # print(hdata.head(n=72))

    # print(len(hdata))
    ddata = hdata.resample('D').sum()
    # print(len(ddata))

    if 1 == 0:
        tempdata = np.stack(
            [dlycsv[3].to_numpy(), ddata['MJpm2'].to_numpy()], axis=1)
        ddata = pd.DataFrame(data=tempdata,
                             columns=['MJpm2', 'MJpm2_red'])
        ddata['perc'] = ddata['MJpm2_red']/ddata['MJpm2']
        print(ddata)
        if 1 == 1:
            for m in range(1, 12, 1):
                print(hdata[hdata.index.month == m].head(n=24))
            print(pvsystem['svf'])
    if 1 == 0:
        with open('hdata.csv', 'w') as fo:
            fo.write(hdata.__repr__())
    # red = ddata['MJpm2'].to_numpy().transpose()
    # print(ddata['MJpm2'].values)
    # print(len(dlycsv[3]))
    # exit(0)
    dlycsv[3] = np.round(ddata['MJpm2'].values, 1)
    dlycsv = dlycsv.drop(columns=['date'])
    dlycsv = dlycsv.replace(np.nan, '', regex=True)
    # drop index col
    #dlycsv = dlycsv.set_index(0)
    # write the modifications to a new directory
    dlyfnmod = os.path.join(DLYDIRMOD, csvr['Identity'] + '.dly')
    # dlycsv.to_csv(dlyfnmod, sep='\t', header=None, index=False)
    
    lastvals = dlycsv[9].values
    #dlycsv[8] = dlycsv[8].astype(str) + ' ' + dlycsv[9].astype(str)
    dlycsv = dlycsv.drop(columns=[9])
    #print(dlycsv.head(n=24))
    
    colspaces = [6,3,3,5,4,4,4,5,5]
    with open(dlyfnmod, 'w') as fo:
        dfAsString = dlycsv.to_string(header=False, index=False, col_space=colspaces)
        fo.write(dfAsString)
        
    # open the file
    with open(dlyfnmod, 'r') as original:
    # get all file content into a variable
        allLines = original.readlines()

    # open the the same file in write mode
    modifyFile = open(dlyfnmod, 'w')
    
    # modify first line of the data
    for i in range(0, len(lastvals)):
        if lastvals[i]:
            #print(i)
            allLines[i] = allLines[i].rstrip() + " " + str(lastvals[i]) +"\n"
        modifyFile.write(allLines[i])


    # Iterate all lines and write into the file
    #for i in allLines:
    #    modifyFile.write(i)
    modifyFile.close()
    #with open(dlyfnmod, 'w') as fo:
    #    fo.write(dlycsv.__repr__())
    #with open(dlyfnmod, 'r') as fi:
    #    data = fi.read().splitlines(True)
    #with open(dlyfnmod, 'w') as fo:
    #    fo.writelines(data[2:])
    print('written file', dlyfnmod)
    #print("The time difference is :", timeit.default_timer() - starttime)


def main(t_configfile: Path = typer.Option(
    "cfg/testcfg.yml", "--config", "-c"),
):

    global config
    # read PV config

    config = rs.getyml(t_configfile)
    config['ccca']['downscale'] = 'liu'

    # open gridinfo and get coordinates, elevation and filename

    # tilt = pvsystem['tilt'][0] # tilt of modules, 0 = horizontal, 90 = wall
    # azimuth = pvsystem['azimuth'][0]

    gridcsv = pd.read_csv(GRIDFILE, sep='\s+')
    print(config['pvsystem'][0])
    print(BBOX)
    gridcsv = gridcsv[gridcsv['latitude'] > BBOX[1]]
    gridcsv = gridcsv[gridcsv['latitude'] < BBOX[3]]
    gridcsv = gridcsv[gridcsv['longitude'] > BBOX[0]]
    gridcsv = gridcsv[gridcsv['longitude'] < BBOX[2]]

    df_dask = ddf.from_pandas(gridcsv, npartitions=1024)
    #print(df_dask.shape)
    with dask.config.set(num_workers=WORKERS):
        # for csvi, csvr in gridcsv.iterrows():
        df_dask.apply(lambda x: itergrid(x), meta=('str'),
                      axis=1).compute(scheduler='processes')


if __name__ == "__main__":
    typer.run(main)
