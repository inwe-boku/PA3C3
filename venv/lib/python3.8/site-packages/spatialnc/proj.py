import os
from urllib.request import urlopen

from netCDF4 import Dataset

from .utilities import strip_chars


def gather_utm_meta(epsg_str):
    """
    Use if the EPSG data is associated to UTM
    Gathers the data and returns a dictionary of data and attributes that need
    to be added to the netcdf based on
    https://www.unidata.ucar.edu/software/thredds/current/netcdf-java/reference/StandardCoordinateTransforms.html

    Args:
        epsg_str: String received from the epsg request.

    Returns:
        map_meta: dictionary of UTM projection data to be added to the netcdf
    """
    meta = epsg_str.lower()

    map_meta = {
        "grid_mapping_name": "universal_transverse_mercator",
        "utm_zone_number": None,
        "semi_major_axis": None,
        "inverse_flattening": None,
        'spatial_ref': epsg_str,
        "_CoordinateTransformType": "Projection",
        "_CoordinateAxisTypes": "GeoX GeoY"}

    # Assign the zone number
    zone_str = meta.split('zone')[1]
    map_meta['utm_zone_number'] = float(
        (strip_chars(zone_str.split(',')[0])).strip()[-2:])

    # Assign the semi_major_axis
    axis_string = meta.split('spheroid')[1]
    map_meta['semi_major_axis'] = float(axis_string.split(',')[1])

    # Assing the flattening
    map_meta["inverse_flattening"] = float(
        strip_chars(axis_string.split(',')[2]))

    return map_meta


def add_proj(nc_obj, epsg=None, nc_to_copy=None, map_meta=None):
    """
    Adds the projection using two different methods. One from an internet
    source using the epsg value and the other from an existing file containing
    it.
    Args:
        nc_obj: netCDF4 dataset object needing the projection information
        nc_to_copy: netcdf obj or path that has desired projection information
        espg: Look up an epsg value on the web if not none.
        map_meta: Pass in a dictionary of the map meta data directly
    Returns:
        nc_obj: Original nc_bj plus the projection information
    """
    if epsg is not None:
        map_meta = add_proj_from_web(epsg)
    elif nc_to_copy is not None:
        map_meta = add_proj_from_file(nc_to_copy)
    elif map_meta is not None:
        pass
    else:
        raise IOError("A netcdf with projection information, or an EPSG code,"
                      " or a dictionary of projection information must be"
                      " passed.")

    # Create a variable called projection
    nc_obj.createVariable("projection", "S1")
    nc_obj["projection"].setncatts(map_meta)

    # Adding coordinate system info to
    for name, var in nc_obj.variables.items():

        # Assume all 2D+ vars are the same projection
        if 'x' in var.dimensions and 'y' in var.dimensions:
            nc_obj[name].setncatts({"grid_mapping": "projection"})

        elif name.lower() in ['x', 'y']:
            # Set a standard name, which is required for recognizing
            # projections
            nc_obj[name].setncatts({"standard_name": "projection_{}_coordinate"
                                    "".format(name.lower())})

            # Set the units
            nc_obj[name].setncatts({"units": "meters"})

    return nc_obj


def add_proj_from_file(nc_to_copy):
    """
    Use a netcdf file converted from a tif using gdal to retrieve the
    projection information

    Args:
        nc_to_copy: netcdf obj or path that has desired projection information

    Returns:
        map_meta: dictionary of attributes to add for spatial reference
    """
    # open a file otherwise assume its an object
    if os.path.isfile(nc_to_copy):
        ds = Dataset(nc_to_copy)
        path_passed = True
    else:
        nc_to_copy = ds
        path_passed = False

    # Sometimes the projection is called transverse_mercator
    if 'transverse_mercator' in ds.variables.keys():
        map_meta = parse_wkt(
            ds['transverse_mercator'].getncattr('spatial_ref'))

    elif 'projection' in ds.variables.keys():
        map_meta = parse_wkt(ds['projection'].getncattr('spatial_ref'))

    else:
        raise Exception('No projection information found in the nc_to_copy')

    # If a path was passed close it
    if path_passed:
        ds.close()

    return map_meta


def add_proj_from_web(epsg):
    """
    Adds the appropriate attributes to the netcdf for managing projection info

    Args:
        epsg:   projection information to be added

    Returns:
        map_meta: dictionary of attributes to add for spatial reference
    """
    # Retrieving projection information...
    # function to generate .prj file information using spatialreference.org
    # access projection information
    try:
        wkt = urlopen(
            "http://spatialreference.org/ref/epsg/{0}/prettywkt/".format(epsg))
    except BaseException:
        wkt = urlopen(
            "http://spatialreference.org/ref/sr-org/{0}/prettywkt/".format(epsg))  # noqa

    # remove spaces between charachters
    remove_spaces = ((wkt.read()).decode('utf-8')).replace(" ", "")

    # Add in the variable for holding coordinate system info
    map_meta = parse_wkt(remove_spaces)

    return map_meta


def parse_wkt(epsg_string):
    """
    Processes the epsg string returned from the URL request.

    Args:
        epsg_str: String received from the epsg request.

    Returns:
        map_meta: dictionary of projection data to be added to the netcdf
    """
    map_meta = {}
    # wkt_data = (epsg_string.lower()).split(',')

    # Add more projection parsers here
    if 'utm' in epsg_string.lower():
        map_meta = gather_utm_meta(epsg_string)

    # Projection information to be added
    # for k,v in map_meta.items():
    #     out.respond("{}: {}".format(k,repr(v)))
    return map_meta
