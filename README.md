# PA3C3

## Workflow

1) read configuration (yml)
2) read areafile (polygon shape)
3) read DHM (raster)
4) read LU (raster)
5) calculate slope from DHM
6) create regular points on area
7) landuse calculations
  a) transform LU raster to LU polygons
  b) dissolve polygons
  c) explode polygons (multipart to singlepart)
  d) calculate area & filter
  e) calculate compactness (Schwartzberg algorithm) & filter
  


## Description

### Downscaling

https://www.hindawi.com/journals/ijp/2015/968024/

### Sunrise/Sunset

several python libraries exists, simple one: suntimes

```
import suntimes
place = SunTimes(lon, lat, alt=200)

```

### NetCDF

The NC Files from CCCA have the following structure (solar radiation example):

```
<class 'netCDF4._netCDF4.Dataset'>
root group (NETCDF4 data model, file format HDF5):
    comment: Bias corrected (scaled distribution mapping) data of the EURO-CORDEX model MOHC-HadGEM2-ES_rcp45_r1i1p1_CLMcom-CCLM4-8-17
using observational data from Global radiation dataset (ZAMG).
Historical and future projection under the RCP4.5 scenario.
Reference period: 1981-2005
    contact: Armin Leuprecht <armin.leuprecht@uni-graz.at>
    institution: Wegener Center for Climate and Global Change, University of Graz, Austria
    project: OEKS 15
    title: Statistically downscaled Global radiation for Austria until 2100 under the RCP4.5 scenario
    Conventions: CF-1.5
    history: Wed Oct  5 14:23:40 2016: ncks -d time,0,44999 rsds_SDM_MOHC-HadGEM2-ES_rcp45_r1i1p1_CLMcom-CCLM4-8-17_all.nc /work/eau00/eau006/oeks15/euro-cordex-sdm/rcp45/rsds_SDM_MOHC-HadGEM2-ES_rcp45_r1i1p1_CLMcom-CCLM4-8-17_1971-2075.nc
    NCO: "4.5.5"
    references: Matthew B. Switanek et al., Scaled distribution mapping: a bias correction method that preserves raw climate model projected changes, Hydrology and Earth System Sciences Discussions, 2016, doi:10.5194/hess-2016-435
    dimensions(sizes): y(297), x(575), time(53610), bnds(2)
    variables(dimensions): int32 lambert_conformal(), float64 lat(y, x), float64 lon(y, x), float32 rsds(time, y, x), float64 time(time), float64 time_bnds(time, bnds), int32 x(x), int32 y(y)
    groups:
```

we have t approximate the location to get to the metered values on the conical CRS.
- written a function for that (thanks peter for the hints)

### Time

- variable: time
- days since 1949-12-01 00:00:00
- 360 day calendar (12x30)

### Radiation

- function (time, y, x)
- variable: rsds
- unit: W/m²
- mode: mean

### Process of DATA

#### GHI daily mean -> GHI hourly

##### Lui Jordan
- INPUT: angle of sunset (pvlib), current sun angle (pvlib)

##### Collares-Pereira Rabel
- INPUT: angle of sunset, current angle

##### horizon calculation from DEM/DHM

https://pypi.org/project/topocalc/#horizon-angles


