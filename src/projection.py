from pyproj import CRS

crs = CRS.from_epsg(31287)
print(crs.to_wkt())