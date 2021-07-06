import typer
from pathlib import Path
import xarray
import numpy as np


def calc_ccca_xy(nd, nx, ny, csrs='epsg:4326'):
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
    abslat = np.abs(nd.lat - ny)
    abslon = np.abs(nd.lon - nx)
    c = np.maximum(abslon, abslat)

    ([yloc], [xloc]) = np.where(c == np.min(c))
    return(xloc, yloc)

def main(ncfile: Path = typer.Option("/home/cmikovits/Downloads/rsds_SDM_MOHC-HadGEM2-ES_rcp45_r1i1p1_CLMcom-CCLM4-8-17.nc", "--file", "-f"),
         bbox: str = typer.Option("12.4321,12.5798,47.1177,47.2827", "--bbox", "-b")
         ):

    typer.echo(f"Netcdf File: {ncfile}, bounds: {bbox}")
    nd = xarray.open_dataset(ncfile)

    [minx,maxx,miny,maxy] = [float(x) for x in bbox.split(',')]

    [minx,miny] = calc_ccca_xy(nd, minx, miny)
    [maxx,maxy] = calc_ccca_xy(nd, maxx, maxy)

    print(f"x,{minx},{maxx} y,{miny},{maxy}")


if __name__ == "__main__":
    typer.run(main)
