from __future__ import division
import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')
import numpy as np
import scipy as sp
import scipy.linalg
import time
import random
import tensorflow as tf
from termcolor import colored
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    print ("Values are: \n%s" % (x))
import pickle

from mdgps import mdGPS
## main
myGPS = mdGPS('hi',True)

myLogging = myGPS.update()

with open('True_set.p', 'wb') as file :
    pickle.dump(myLogging,file)