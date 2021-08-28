import argparse

import sys
import re
import numpy as np
import pylab
import json
import requests

try: # Python 3.x
    from urllib.parse import quote as urlencode
    from urllib.request import urlretrieve
except ImportError:  # Python 2.x
    from urllib import pathname2url as urlencode
    from urllib import urlretrieve

try: # Python 3.x
    import http.client as httplib 
except ImportError:  # Python 2.x
    import httplib 
    
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from concurrent.futures import ThreadPoolExecutor
import asyncio
import asyncio
import codecs
#import aiohttp

import sfdmap
import astropy
#astropy is a requirement for sfdmap

from photoz_helper import easy_photoz_objid

if __name__ == '__main__':
    '''
    Given a single objID, computes the Z and Zerr and objid in nice prints
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('objids',nargs='+',help='PanStarrs objids of objects to classify.')
    parser.add_argument('-f',type=str,default=None,help='Filename to write results to, if not given, wont write. Results saved as a numpy obj array')
    parser.add_argument('-q',type=bool,default=False,help='quiet, if true reduces amount printed to command line') 
    args = parser.parse_args()
 
    if args.f:
        try:
            open(args.f, 'w')
        except OSError:
            raise OSError('given filename path doesnt exist, or is otherwise unwritable')
    
    posterior, point_estimate, err, returned_objids = easy_photoz_objid(args.objids)
    
    if not(args.q):
        for j in range(len(returned_objids)):
            print('objid: {0}, point_estimate: {1:.5f}, err: {2:.5f}'.format(returned_objids[j],point_estimate[j],err[j]))
    
    if args.f:
        save_dict={'posterior':posterior,'point_estimate':point_estimate,'err':err,'returned_objids':returned_objids}
        np.save(args.f,save_dict)

        