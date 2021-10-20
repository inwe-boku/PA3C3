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
import math
from pathlib import Path
import glob
from pyproj import CRS
# import gdal


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


def create_npy(t_dhm, angles, npyout):
    typer.echo(
        f"Reading {t_dhm} to calculate horizons")

    with rasterio.open(t_dhm, 'r') as ds:
        psx, psy = ds.res
        width = ds.width
        height = ds.height
        typer.echo(
            f"\tDimensions: {height}x{width}\n\tResolution: {psx}"
        )
        dhmcrs = ds.crs
        dhmbounds = ds.bounds
        # .swapaxes(0,1) # read all raster values
        dem = ds.read()[0].astype(np.double)

        # angle[:] = angles
        typer.echo(
            f"\tProcessing DEM with angles: {angles}")

        with typer.progressbar(angles) as progressangles:
            for idx, a in enumerate(progressangles):
                result = horizon(a, dem, psx)
                filename = os.path.join(npyout, str(a))
                np.save(filename, result)
                del(result)


def stich_npy(nc_filename, angles, outfolder, t_npy):
    ncfile = nc.Dataset(nc_filename, 'r+', format='NETCDF4')
    hor = ncfile['horizon']
    print(hor)
    for idx, a in enumerate(angles):
        angle = ncfile['angle'][idx]
        filename = os.path.join(outfolder, str(angles[idx]))
        print("reading: ", filename, 'at index:', idx)
        if t_npy:
            filename = str(filename) + '.npy'
            result = np.load(filename)
        else:
            filename = os.path.join(outfolder, str(angles[idx]))
            filename = str(filename) + '.tif'
            with rasterio.open(filename, 'r') as ds:
                # print(ds.bounds)
                # psx, psy = ds.res
                # width = ds.width
                # height = ds.height
                # dhmcrs = ds.crs
                # print(ds.crs)
                # dhmbounds = ds.bounds
                # .swapaxes(0,1) # read all raster values
                result = ds.read(1)

                # print(test.shape)
                # result = ds.read()[0].astype(np.double)
                # print(result)
                # tif result = radians -> conversion to degrees
                result = np.degrees(result)
            # print(np.median(result))
        # result = result.clip(min=0)
        print(np.ndarray.min(result), np.ndarray.mean(
            result), np.ndarray.max(result))
        hor[idx, :, :] = result
    ncfile.close()


def main(t_dhm: str = typer.Option("exampledata/Felberthal/dhm.tif", "--dhm", "-d"),
         t_out: str = typer.Option("out", "--out", "-o"),
         t_angles: int = typer.Option(10, "--angles", "-a"),
         t_resume: bool = typer.Option(False, "--resume", "-r"),
         t_stitch: bool = typer.Option(False, "--stitch", "-s"),
         t_npy: bool = typer.Option(True, "--npy", "-n")):

    angles = np.arange(-180, 180, t_angles)
    nc_filename = os.path.join(Path.home(), t_out, 'horizons.nc')

    outfolder = t_out
    if t_npy:
        npyfolder = os.path.join(Path.home(), t_out, 'npy')

        if not os.path.exists(npyfolder):
            os.makedirs(npyfolder, exist_ok=True)

    if t_stitch:
        if t_npy:
            outfolder = npyfolder
        # filename = os.path.join(outfolder, 'dimensions')
        # print(filename)
        with rasterio.open(t_dhm, 'r') as ds:
            psx, psy = ds.res
            width = ds.width
            height = ds.height
            typer.echo(
                f"\tDimensions: {height}x{width}\n\tResolution: {psx}"
            )
            dhmcrs = ds.crs
            dhmbounds = ds.bounds
        if t_resume:
            typer.echo(
                f"STITCHING to EXISTING file: {nc_filename}")
        else:
            typer.echo(
                f"STITCHING to NEW file: {nc_filename}")
            filename = os.path.join(outfolder, str(angles[0]))
            if t_npy:
                filename = str(filename) + '.npy'
                result = np.load(filename)
            else:
                filename = os.path.join(outfolder, str(angles[0]))
                filename = str(filename) + '.tif'
                with rasterio.open(filename, 'r') as ds:
                    result = ds.read()[0].astype(np.double)
            height = result.shape[0]
            width = result.shape[1]
            ncfile = nc.Dataset(nc_filename, 'w', format='NETCDF4')
            ncfile.title = 'Horizon file'
            ncfile.subtitle = "Horizon calculations from DHM each 10 degrees"

            y_dim = ncfile.createDimension('y', height)     # latitude - y axis
            y = ncfile.createVariable('y', np.uint32, ('y',))
            y.standard_name = 'y coordinate (lat)'
            y.units = 'meter'
            y.axis = "Y"
            y.long_name = 'y (lat)'

            x_dim = ncfile.createDimension('x', width)    # longitude - x axis
            x = ncfile.createVariable('x', np.uint32, ('x',))
            x.units = 'meter'
            x.standard_name = 'x coordinate (lon)'
            x.axis = "X"
            x.long_name = 'x (lon)'

            angle_dim = ncfile.createDimension(
                'angle', len(angles))  # angles 180/-180 = N; 0 = S
            angle = ncfile.createVariable('angle', np.int16, ('angle',))
            angle.units = 'angle from 180 to -180 (where 0 = S, minus towards East, plus towards West)'
            angle.long_name = 'angle'
            # note: unlimited dimension is leftmost
            hor = ncfile.createVariable(
                'horizon', np.float32, ('angle', 'y', 'x'),
                zlib=True,
                complevel=4,
                least_significant_digit=3,
                fill_value=0)
            hor.units = 'degree'  # degrees
            hor.setncattr('grid_mapping', 'spatial_ref')
            # this is a CF standard name
            hor.standard_name = 'horizon angle, 0 = flat, 90 = maximum zenith'
            ny = len(y_dim)
            nx = len(x_dim)
            nangle = len(angles)

            nccrs = ncfile.createVariable('spatial_ref', 'i4')
            mycrs = CRS.from_epsg(31287)
            # print(mycrs.to_wkt())
            nccrs.spatial_ref = mycrs.to_wkt()

            y[:] = dhmbounds.top - ((np.arange(ny)+1)*10)
            x[:] = dhmbounds.left + ((np.arange(nx)+1)*10)
            # print(dhmbounds)
            # print(np.arange(ny))
            #print(len(dhmbounds.top - ((np.arange(ny)+1)*10)))
            # exit(0)
            angle[:] = angles+180
            ncfile.close()
            stich_npy(nc_filename, angles, outfolder, t_npy)

    else:
        if t_resume:
            typer.echo(
                f"RESUMING horizons from: {t_dhm}")
            npyfiles = glob.glob(os.path.join(npyfolder, "*"))
            for npyf in npyfiles:
                npyf = os.path.basename(npyf)
                npyf = os.path.splitext(npyf)[0]
                # print(np.where(angles == int(npyf)))
                angles = np.delete(angles, np.where(angles == int(npyf)))
            create_npy(t_dhm, angles, npyfolder)
        else:
            typer.echo(
                f"CREATING horizons from: {t_dhm}")
            create_npy(t_dhm, angles, npyfolder)

    typer.echo(
        f"Horizon calculation finished")


if __name__ == "__main__":
    typer.run(main)
