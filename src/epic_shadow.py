#!/usr/bin/env python3

import typer
from pathlib import Path
import json
import yaml
import os
import datetime
import suntimes
import renspatial as rs
import pandas as pd
import pvlib

__author__ = "Christian Mikovits"

GRIDFILE = '/data/projects/PA3C3/EPICOKS15/A_Infos/Grid info/OKS15_AT_geodata_new.txt'
DLYDIR = '/data/projects/PA3C3/EPICOKS15/SZEN'

### open gridinfo and get coordinates, elevation and filename

### get sunrise and set for coordinates

### read PV config and calculate hourly values for shadowing reduction

### read daily value from ASCII and transform to hourly

### reduce hourly by shadow and transform to daily

### write ASCII

def main():
    gridcsv = gridcsv = pd.read_csv(GRIDFILE, sep='\s+')
    i = 0
    for csvi, csvr in gridcsv.iterrows():
        i = i+1
        if i == 5: break
        location = pvlib.location.Location(
                csvr['longitude'], csvr['latitude'],
                'UTC', csvr['Elev'], csvr['Identity'])
        dlyfn = os.path.join(DLYDIR, csvr['Identity'] + '.dly')
        dlycsv = gridcsv = pd.read_csv(dlyfn, sep='\s+', header=None)
        dlycsv['date'] = pd.to_datetime(dlycsv[0]*10000+dlycsv[1]*100+dlycsv[2], format='%Y%m%d')
        for dlyi, dlyr in gridcsv.iterrows():
            date = dlyr['date']
            srad = dlyr[3] # MJ/m2 and day    
            sstime = rs.sunset_time(location, date)
            print(sstime)
        
    

if __name__ == "__main__":
    typer.run(main)
