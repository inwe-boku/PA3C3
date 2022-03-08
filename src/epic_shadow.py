#!/usr/bin/env python3

import typer
from pathlib import Path
import json
import yaml
import os
import datetime
import renspatial as rs
import pandas as pd
import numpy as np
import pvlib
import suntimes
import math

__author__ = "Christian Mikovits"

GRIDFILE = '/data/projects/PA3C3/EPICOKS15/A_Infos/Grid info/OKS15_AT_geodata_new.txt'
DLYDIR = '/data/projects/PA3C3/EPICOKS15/SZEN'

pd.set_option('display.max_rows', 999)

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
    i = 0
    data = pd.DataFrame()
    for idx, row in ddata.iterrows():
        #print(ddata)
        # raw data is in W/m2 -> this is meteorological mean data per hour and day, have to multiply by 24
        ###
        ### conversion MJ/m2 in W/m2
        ###
        dval = row[3] * 1000 / 3.6
        # print(ddata[i], dval)
        date = row['date']
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
        #print(config['ccca'])
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

### read daily value from ASCII and transform to hourly

### reduce hourly by shadow and transform to daily

### write ASCII

def main(t_configfile: Path = typer.Option(
             "cfg/testcfg.yml", "--config", "-c"),
         ):
    
    global config
    ### read PV config
    
    config = rs.getyml(t_configfile)
    config['ccca']['downscale'] = 'cpr'
        
    pvsystem = config['pvsystem'][0]
    print(pvsystem)
    ### open gridinfo and get coordinates, elevation and filename
    
    #tilt = pvsystem['tilt'][0] # tilt of modules, 0 = horizontal, 90 = wall
    #azimuth = pvsystem['azimuth'][0]
    
    if pvsystem['lengthwidth'] == 'width':
        module_lw = PVmoduleinfo(pvsystem['module'])['Width']
    else:
        module_lw = PVmoduleinfo(pvsystem['module'])['Length']
    
    footprint = round((module_lw * math.cos(math.radians(pvsystem['tilt'][0]))),2)
    height = pvsystem['height']
    
        
    gridcsv = gridcsv = pd.read_csv(GRIDFILE, sep='\s+')
    i = 0
    for csvi, csvr in gridcsv.iterrows():
        i = i+1
        if i == 2: break
        location = pvlib.location.Location(
                csvr['longitude'], csvr['latitude'],
                'UTC', csvr['Elev'], csvr['Identity'])
        dlyfn = os.path.join(DLYDIR, csvr['Identity'] + '.dly')
        dlycsv = gridcsv = pd.read_csv(dlyfn, sep='\s+', header=None)
        dlycsv['date'] = pd.to_datetime(dlycsv[0]*10000+dlycsv[1]*100+dlycsv[2], format='%Y%m%d')
        dlycsv = dlycsv[dlycsv[0] < 1982]
        
        #print(dlycsv.head())
                
        hdata = ghid2ghih(dlycsv, location)
        hdata = ghi2dni(hdata, config['pvmod']['hmodel'])
        
        # zenith shadow calc
        #
        # the 'footprint' of the module on the ground
        # is independent of the zenith and always the same
        
        # the shade of the module height varies with the zenith
        z_b = round(height / np.tan(np.radians(90-(hdata['z_h']))),2)
        
        # shadefree length: z_c = pvsystem['moduledist'] - hdata['z_b']
        
        # zenith angle shade relation; 1 = full shade, 0 = no shade
        hdata['z_rel'] = round((footprint + abs(z_b))/abs(pvsystem['moduledist'] - z_b),2)
        hdata.loc[hdata.z_rel > 1 , 'z_rel'] = 1
        
        # azimuth shadow calc
        # relation shading the sun; 1 = full shade, 0 = no shade
        hdata['w_rel'] = round(abs(np.cos(np.radians(pvsystem['azimuth'][0] - hdata['w_h']))),2)

        # combination of zenith and azimuth shading
        hdata['s_rel'] = round(hdata.z_rel*hdata.w_rel,2)
        
        # reduction of dni
        hdata['dni_red'] = hdata.dni * (1-hdata.s_rel)
        
        print(hdata[hdata.index.month == 1].head(n=24))
        print(hdata[hdata.index.month == 6].head(n=72))
        print(footprint)
        print(height)
        print(pvsystem['moduledist'])        
        print(pvsystem['azimuth'][0])
        exit(0)
        
        
        for dlyi, dlyr in gridcsv.iterrows():
            date = dlyr['date']
            srad = dlyr[3] # MJ/m2 and day
            
            ###
            ### conversion MJ/m2 in W/m2
            srad = srad * 10^3 / 3.6
            ###
            
            
            
            ### get sunrise and set for coordinates 
            sstime = rs.sunset_time(location, date)
            times = pd.date_range(date, date+pd.Timedelta('24h'), freq='1h')
            solpos = pvlib.solarposition.get_solarposition(times, csvr['latitude'], csvr['longitude'])
            solpos = solpos.loc[solpos['apparent_elevation'] > 0, :]
            #print(location)
            solpos['r_azimuth'] = np.sin(np.radians(90 - (solpos['azimuth'] - pvsystem['azimuth'])))
            solpos['r_zenith'] = np.cos(np.radians((solpos['zenith']- pvsystem['tilt'][0])))
            print(solpos)
            
            # calculate hourly values for shadowing reduction
            
            print(sstime)
        
    

if __name__ == "__main__":
    typer.run(main)
