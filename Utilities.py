# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 19:52:49 2020

@author: David
"""
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