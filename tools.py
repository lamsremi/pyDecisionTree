"""This module includes the common decorators used for development.
"""

import time
import sys


def debug_more(function):
    """Debug with output and arguments."""
    def wrapper(*args):
        """Wrapper definition."""
        result = function(*args)
        print(('{0}{1} :\n {2}'.format(function.__name__, args, result)))
        return result
    return wrapper


def debug(function):
    """Debug only with output."""
    def wrapper(*args, **kw):
        """Wrapper definition."""
        result = function(*args, **kw)
        if isinstance(result, (dict, list)):
            print(('{0}() :\n'.format(function.__name__)))
            print_well(result)
        else:
            print(('{0}() :\n {1}'.format(function.__name__, result)))
        return result
    return wrapper


def timeit(method):
    """Time a function."""
    def timed(*args, **kw):
        """Wrapper time."""
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((end_time - start_time) * 1000)
        else:
            print(('%r  %2.1f ms' % (method.__name__, (end_time - start_time) * 1000)))
        return result
    return timed


def print_well(obj, nested_level=0, output=sys.stdout):
    """Print a dictionnary properly."""
    spacing = '   '
    if isinstance(obj, dict):
        print('%s{' % ((nested_level) * spacing), file=output)
        for key, value in list(obj.items()):
            if hasattr(value, '__iter__'):
                print('%s%s:' % ((nested_level + 1) * spacing, key), file=output)
                print_well(value, nested_level + 1, output)
            else:
                print('%s%s: %s' % ((nested_level + 1) * spacing, key, value), file=output)
        print('%s}' % (nested_level * spacing), file=output)
    elif isinstance(obj, list):
        print('%s[' % ((nested_level) * spacing), file=output)
        for value in obj:
            if hasattr(value, '__iter__'):
                print_well(value, nested_level + 1, output)
            else:
                print('%s%s' % ((nested_level + 1) * spacing, value), file=output)
        print('%s]' % ((nested_level) * spacing), file=output)
    else:
        print('%s%s' % (nested_level * spacing, obj), file=output)
