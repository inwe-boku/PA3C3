from collections import OrderedDict


def get_stats(data, np_stats=['max', 'min', 'mean', 'std']):
    '''
    Given a numpy array, calculate all the numpy statistics and return them
    in a dictionary

    Args:
        data: Numpy array of any size/dimension
        np_stats: Statistic name to use, must be an function of a numpy array
    Returns:
        dictionary: dict of the stats names as keys and values associated
            with np_stats
    '''

    # put together the operations to use
    operations = np_stats

    # Build the output
    out = OrderedDict()

    for op in operations:
        stat_fn = getattr(data, op)
        out[op] = stat_fn()

    return out
