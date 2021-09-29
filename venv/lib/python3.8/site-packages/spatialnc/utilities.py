import logging
import os

import numpy as np
from netCDF4 import Dataset


def get_logger(name, level='debug', log_file='log.txt'):
    """
    Retrieve the logger for SWIFLow with coloredlogs installed in the
    right format
    """
    # Setup logging
    log_level = level.upper()
    level = logging.getLevelName(log_level)

    # Add a custom format for logging
    fmt = "%(levelname)s: %(msg)s"

    # Always write log file to <output>/log.txt
    log = logging.getLogger(name)

    # Log to file, no screen output.
    logging.basicConfig(filename=log_file, filemode='w+',
                        level=log_level,
                        format=fmt)
    return log


def strip_chars(edit_str, bad_chars='[(){}<>,"_]=\nns'):
    """
    Written to strip out unwanted chars from the proj strings received
    back from the URL for the EPSG requests.

    Args:
        edit_str: String containing unwanted chars
        bad_chars: String of chars to be removed

    Returns:
        result: The edit_str without any of the chars in bad_chars
    """

    result = ''.join([s for s in edit_str if s not in bad_chars])
    return result


def copy_nc(infile, outfile, exclude=None):
    """
    Copies a netcdf from one to another exactly.

    Args:
        infile: filename or netCDF4 dataset object you want to copy
        outfile: output filename
        exclude: variables to exclude

    Returns the output netcdf dataset object for modifying
    """

    if exclude is not None:
        if not isinstance(exclude, list):
            exclude = [exclude]
    else:
        exclude = []

    dst = Dataset(outfile, "w")

    # Allow for either object or filename to be passed
    if isinstance(infile, str):
        src = Dataset(infile)
    else:
        src = infile

    # copy global attributes all at once via dictionary
    dst.setncatts(src.__dict__)
    dst.set_fill_on()

    if exclude is not None:
        for vname in exclude:
            if vname not in src.variables:
                raise ValueError("Attempting to exclude variable '{}' "
                                 "which is not in {}.\n Available variables "
                                 "are: {}".format(vname, src.filepath(),
                                                  ", ".join(src.variables)))

    # copy dimensions
    for name, dimension in src.dimensions.items():
        dst.createDimension(
            name, (len(dimension) if not dimension.isunlimited() else None))

    # copy all file data except for the excluded
    for name, variable in src.variables.items():

        if name not in exclude:
            dst.createVariable(name, variable.datatype, variable.dimensions)
            # fill_value=variable._grp[variable._varid])#.varid._FillValue)

            if name != 'projection':
                dst[name][:] = src[name][:]

            # copy variable attributes via dictionary and avoid overwriting any
            incoming_dict = {}

            for k, v in src[name].__dict__.items():

                if k not in dst[name].__dict__.keys():
                    if k not in ["_FillValue", "fill_value"]:
                        incoming_dict[k] = v

            dst[name].setncatts(incoming_dict)

    return dst


def mask_nc(unmasked_file, mask_file, output=None, exclude=[]):
    """
    Masks a all the variables in a netcdf exlcuding, time, projection, x, y.
    If output = none it will make the file named after the original filename

    Args:
        unmasked_file: Path to a netcdf dataset to that is to be masked
        mask_file: Path to a netcdf dataset containing a variable named mask
        output: filename to output the data
        exlcude: variables to exclude

    Returns:
        dst: dataset object that the new masked dataset was written
    """
    # Isolate the name of the input file and use it for netcdf
    out_fname = 'masked_{}.nc'.format(
        (os.path.split(unmasked_file)[-1]).split('.')[0])

    # Parse the option name
    if output is not None:
        out_fname = os.path.join(output, out_fname)

    unmasked = Dataset(unmasked_file)
    mask_ds = Dataset(mask_file)
    mask = mask_ds.variables['mask'][:]

    mask = mask.astype(float)
    mask[mask == 0] = np.nan

    # Make a copy
    dst = copy_nc(unmasked_file, out_fname)

    for name, variable in unmasked.variables.items():
        dims = variable.dimensions

        if 'x' in dims and 'y' in dims:
            if 'time' in dims:
                # Mask all data in the time series
                for i, t in enumerate(unmasked.variables['time'][:]):
                    dst.variables[name][i, :] = unmasked.variables[name][i, :] * mask  # noqa
            else:
                dst.variables[name][:] = unmasked.variables[name][:] * mask
            # Set the
            np.ma.set_fill_value(dst.variables[name][:], np.nan)

    # Close out
    unmasked.close()
    mask_ds.close()

    return dst


# def ipw2nc(ipw_f, nc_f):
#     """
#     Converts ipw to netcdf
#     """
#     ipw_ds = ipw.IPW(ipw_f)
#     for b, var in enumerate(m['name']):
#         em.variables[var][j, :] = i_em.bands[b].data
