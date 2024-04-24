import time
import warnings

def timeit(func):

    def inner(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        print('[T] Function {} finished in {:2.4f} s.'.format(func.__name__, te-ts))
        return result

    return inner


def ignore_warnings(func):

    def inner(*args, **kwargs):
        warnings.simplefilter("ignore")
        result = func(*args, **kwargs)
        warnings.resetwarnings()
        return result

    return inner