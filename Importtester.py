# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 19:36:34 2020

@author: David
"""

lib = 'CNN6_FC2'
package = 'model'

import importlib

def importModelArchitecture(package, module):
    try:
        cnn = importlib.import_module(package+'.'+module)
        #cnn = importlib.import_module(architecture, 'model')
        #from model import module as cnn # import the desired module
        print('Successfully imported {} from package {}.'.format(
            module, package))
        return cnn
    except ImportError:
        print('Importing failed.')

np = importModelArchitecture(package, lib)




