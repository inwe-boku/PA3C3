import typer
import rasterio
from topocalc.horizon import horizon
import numpy as np


def main(t_dhm: str = typer.Option("/home/cmikovits/GEODATA/DHMAT/dhm_at_lamb_10m_2018.tif", "--dhm", "-dhm"),
         t_angles: int = typer.Option(36, "--angles", "-a")):
    typer.echo(
        f"Reading {t_dhm} to calculate horizons")
    angles = np.arange(-180, 180, t_angles)

    with rasterio.open(t_dhm, 'r') as ds:
        psx, psy = ds.res
        width = ds.width
        height = ds.height
        typer.echo(
            f"\tDimensions: {width}x{height}\n\tResolution: {psx}"
        )
        dem = ds.read()[0].astype(np.double)  # read all raster values
        typer.echo(
            f"\tProcessing DEM with angles: {angles}")
        with typer.progressbar(angles) as progressangles:
            print('test')
            for a in progressangles:
                result = horizon(a, dem, psx)
                print(result.width)


if __name__ == "__main__":
    typer.run(main)
