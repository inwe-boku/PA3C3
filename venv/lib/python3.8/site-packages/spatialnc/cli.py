#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
from os.path import abspath, expanduser

from netCDF4 import Dataset

from . import __version__
from .analysis import get_stats
from .proj import add_proj


def nc_stats():

    parser = argparse.ArgumentParser(
        description='Calculate Statistics on NetCDF Files.')

    parser.add_argument(
        'filename', metavar='f', type=str,
        help='Path of a netcdf file (File Extension = .nc).')

    parser.add_argument(
        '-v', '--variables', dest='variables', type=str, nargs='+',
        help='Name of the variable(s) in the netcdf file to process.')

    parser.add_argument(
        '-p', '--precision', dest='precision', type=int,
        default=4,
        help='Number of decimals to show in the results.')

    parser.add_argument(
        '-idt',
        dest='index_time',
        type=int,
        nargs='+',
        help="Specify a time index from file")

    parser.add_argument(
        '-idy',
        dest='index_y',
        type=int,
        nargs='+',
        help="Specify an y index to run statistics on ")

    parser.add_argument(
        '-idx',
        dest='index_x',
        type=int,
        nargs='+',
        help="Specify an x index to run statistics on ")

    parser.add_argument('-s', dest='statistics', type=str, nargs='+',
                        choices=['max', 'min', 'mean', 'std'],
                        default=['max', 'min', 'mean', 'std'],
                        help="Specify an x index to run statistics on ")

    args = parser.parse_args()

    filename = abspath(expanduser(args.filename))

    # Open data set
    ds = Dataset(filename, 'r')

    variables = args.variables
    if args.variables is None:
        variables = [
            name for name,
            v in ds.variables.items() if name not in [
                'time',
                'x',
                'y',
                'projection']]

    # track how long this takes.
    start = time.time()

    print("\n=========== NC_STATS v{} ==========".format(__version__))
    print("\nProcessing netCDF statistics...")
    print("Filename: {0}".format(filename))

    # Check user inputs are valid
    dimensions = []

    # Is the user requesting a slice?
    user_slicing = False

    for dim in ['time', 'y', 'x']:
        dim_input = getattr(args, 'index_{}'.format(dim))
        dim_range = []

        # Is it a valid dimension name?
        if dim not in ds.variables and dim_input is not None:
            print('\nError: {} is not a valid dimension in this dataset! '
                  'Available dimensions are: {}\n'
                  ''.format(dim, ', '.join(ds.dimensions.keys())))
            sys.exit()

        if dim in ds.variables.keys():
            # Was not provided
            dimension_shape = ds.variables[dim][:].shape[0]

            if dim_input is None:
                dim_range = [0, dimension_shape]

            # Was a dimension index provided?
            else:
                user_slicing = True
                # Check the values actually provided
                for v in dim_input:
                    # is it in the index of the dataset
                    if v in range(0, dimension_shape):
                        dim_range.append(v)
                    else:
                        print('Error: Index of {} out of range for {}'
                              'dimension!\n'.format(dim, v))
                        sys.exit()

                # Is the user asking for a single point?
                if len(dim_range) == 1:
                    dim_range.append(dim_range[0] + 1)

            # Append the dimensions to our list and make them a slice
            dimensions.append(slice(*dim_range))

    # Check the dimensionality and put them into a tuple
    ND = len(dimensions)

    if ND == 3:
        slices = dimensions[0], dimensions[1], dimensions[2]

    elif ND == 2:
        slices = dimensions[0], dimensions[1]

    elif ND == 1:
        slices = (dimensions[0])
    else:
        print('ERROR: nc_stats is unable to handle {} dimensional data.\n'.format(ND))  # noqa
        sys.exit()

    # Loop through all variables and print out results
    for v in variables:
        print('')
        if v not in ds.variables.keys():
            print('{} is not in the dataset, skipping...'.format(v))

        else:
            msg_str = " " * 3 + "{} statistics".format(v) + " " * 3
            print("=" * len(msg_str))
            print(msg_str)
            print("=" * len(msg_str))
            data = ds.variables[v][slices]

            print("  Data Dimensions: {0}"
                  "".format(" X ".join(
                      [str(s) for s in ds.variables[v].shape])
                  ))

            if user_slicing:
                print("  Filtered Dimensions: {0}"
                      "".format(
                          " X ".join([str(s) for s in data.shape])
                      ))
                print('  Using indicies [{}]'.format(
                    ", ".join(
                        ["{}:{}".format(s.start, s.stop) for s in slices])
                ))

            print('')

            # Get the statistics
            stats = get_stats(data, np_stats=args.statistics)

            # Output to screen
            interior = "1:0.{}f".format(args.precision)
            msg = '   {0} = ' + '{' + interior + '}'

            for stat, value in stats.items():
                print(msg.format(stat.upper(), value))

    ds.close()
    print('\nComplete! Elapsed {:d}s'.format(int(time.time() - start)))


def make_projected_nc():
    '''
    CLI entrypoint for adding projections information to a netcdf that
    doesnt have it already

    Add projection information to netcdf file. This will add it to existing
    file old_nc
    '''

    # Parse arguments
    p = argparse.ArgumentParser(
        description='Add projection info to NetCDF file')

    p.add_argument('old_nc', metavar='o', type=str,
                   help='Path to original netcdf with no projection')

    p.add_argument('-pnc', '--projected_nc', required=True,
                   help="Path to netcdf that has projection information")

    args = p.parse_args()

    old_nc = os.path.abspath(args.old_nc)
    projected_nc = os.path.abspath(args.projected_nc)

    s = Dataset(old_nc, 'a')

    # if we have projection info, do not need to add it
    if 'projection' in s.variables.keys():
        raise IOError('{} already has projection information'.format(old_nc))

    # give the user a choice
    else:
        y_n = 'a'  # set a funny value to y_n
        # while it is not y or n (for yes or no)
        while y_n not in ['y', 'n']:
            y_n = input(
                'The script make_projected_nc will modify the file '
                '{} by adding projection information. Would you like to '
                'proceed? (y n): '.format(old_nc))

            # if they say yes, add it
            if y_n.lower() == 'y':
                s = add_proj(s, None, projected_nc)
                s.sync()
            elif y_n.lower() == 'n':
                print('Exiting')
            else:
                raise IOError('Did not answer "y" or "n"')

    # close nc file
    s.close()
