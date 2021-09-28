import typer
import rasterio
from rasterio.features import shapes
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from topocalc.horizon import horizon
import numpy as np
import netCDF4 as nc
from fiona.crs import from_epsg
import os
import tempfile
import xarray


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
    

def main(t_dhm: str = typer.Option("exampledata/Felberthal/dhm.tif", "--dhm", "-d"),
         t_ncf: str = typer.Option("/home/cmikovits/horizon.nc", "--netcdf", "-nc"),
         t_angles: int = typer.Option(36, "--angles", "-a")):
    typer.echo(
        f"Reading {t_dhm} to calculate horizons")
    nc_filename = "/home/cmikovits/netcdftest.nc"
    ncfile = nc.Dataset(nc_filename, 'w', format='NETCDF4')
    angles = np.arange(-180, 180, t_angles)

    with rasterio.open(t_dhm, 'r') as ds:
        psx, psy = ds.res
        width = ds.width
        height = ds.height
        typer.echo(
            f"\tDimensions: {height}x{width}\n\tResolution: {psx}"
        )
        dhmcrs = ds.crs
        dhmbounds = ds.bounds
        dem = ds.read()[0].astype(np.double) #.swapaxes(0,1) # read all raster values
        
        ncfile.title = 'Horizon file'
        ncfile.subtitle="My model data subtitle"
        y_dim = ncfile.createDimension('y', height)     # latitude - y axis
        x_dim = ncfile.createDimension('x', width)    # longitude - x axis
        angle_dim = ncfile.createDimension('angle', len(angles)) # angles 180/-180 = N; 0 = S
        
        y = ncfile.createVariable('y', np.int32, ('y',))
        y.units = 'meter'
        y.long_name = 'y (lat)'
        x = ncfile.createVariable('x', np.int32, ('x',))
        x.units = 'meter'
        x.long_name = 'x (lon)'
        angle = ncfile.createVariable('angle', np.int32, ('angle',))
        angle.units = 'angle from 180 to -180 (where 0 = S, minus towards East, plus towards West)'
        angle.long_name = 'angle'
        hor = ncfile.createVariable('horizon',np.float32,('angle','y','x')) # note: unlimited dimension is leftmost
        hor.units = 'degree' # degrees Kelvin
        hor.standard_name = 'horizon angle in degrees' # this is a CF standard name
        ny = len(y_dim); nx = len(x_dim); nangle = len(angles)
        
        y[:] = dhmbounds.bottom + (dhmbounds.top/ny)*np.arange(ny)
        x[:] = dhmbounds.left + (dhmbounds.left/nx)*np.arange(nx)
        angle[:] = angles
        #angle[:] = angles
        typer.echo(
            f"\tProcessing DEM with angles: {angles}")
        with typer.progressbar(angles) as progressangles:
            for idx, a in enumerate(progressangles):
                result = horizon(a, dem, psx)
                hor[idx,:,:] = result
    
    ncfile.close()
    typer.echo(
        f"Horizon calculation finished")

if __name__ == "__main__":
    typer.run(main)
