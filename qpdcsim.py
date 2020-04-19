#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:16:37 2020

Simulates (quasi)-periodic drive in frequency lattice description
Rational approximations for quasi-periodic

@author: acpotter
"""
import numpy as np # generic math functions
#import numpy.linalg as la
#import scipy.linalg as linalg
import matplotlib.pyplot as plt

PHI = (np.sqrt(5)-1)/np.sqrt(2) # golden-ratio

#%% dev
# setup extended Hilbert space
D = 2 # number of drives
Ts = [1, PHI] # periods (should be incommensurate if more than one)

# define the drive Hamiltonian

# simulate the quasi-energy spectrum
