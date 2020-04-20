#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:16:37 2020

Simulates (quasi)-periodic drive in frequency lattice description
Rational approximations for quasi-periodic

@author: acpotter
"""
import numpy as np # generic math functions
import numpy.linalg as la
#import scipy.linalg as linalg
import matplotlib.pyplot as plt

#%% Global variables
PHI = (np.sqrt(5)+1)/2 # golden-ratio
# pauli matrices
PAULI = {'X':np.array([[0,1],[1,0]]),
         'Y':np.array([[0,-1j],[1j,0]]),
         'Z':np.array([[1,0],[0,-1]])
         }

#%% Fibonacci drive functions
def Fibonacci(N):
    """
    computes Nth Fibonacci number recursively

    Parameters
    ----------
    N : int (positive)
        DESCRIPTION.

    Returns
    -------
    Nth Fibonacci number.

    """
    # exception handling
    if type(N)!= int:
        raise TypeError('N must be integer')
    if N<0:
        raise ValueError('N must be >= 0')
        
    # recursion
    if N==0:
        return 1
    elif N==1:
        return 1
    else:
        return Fibonacci(N-1)+Fibonacci(N-2)

def Fold(x,W):
    """
    Folds a generic point in ZxZ frequency lattice back into the reduced zone

    Parameters
    ----------
    x : np.array of length 2, and type int
        original lattice position
    W : GL(2,Z) matrix as 2x2 integer np.array 
        Rotates tilted lattice into original one 

    Returns
    -------
    x_folded = x folded back into reduced zone

    """
    # shift before folding so that reduced zone includes (0,0), 
    # and goes through upper right and lower left quadrants
    shift = W[1,:]-np.array([1,0]) 
    y = la.solve(W,x+shift) # y = W^{-1}x, coordinate in tilted lattice
    y[1] = np.mod(y[1],1) # fold back into zone
    
    # rotate back to original lattice (return as int, and undo shift)
    return np.around(W@y-shift).astype(int)
    
#%% Pulses
def GaussianFT(width,weight,n):
    """
    Returns nth fundamental harmonic component of Gaussian pulse train with period 1

    Parameters
    ----------
    width : float (>0) 
        pulse width (sigma)
    weight : float (>0)
        integrated weight of pulse
    n : int 
        harmonic to compute

    Returns
    -------
    amplitude: float
        amplitude of nth Harmonic

    """
    return weight/np.sqrt(2*np.pi/width**2)*np.exp(-width**2*n**2/2)

#%% classes
class Interaction():
    """
    """
    
    def __init__(self,attributes):
        """
        

        Parameters
        ----------
        attributes : dictionary
            entries must include:   'type':string-indicating type, 
                                    'operator':numpy array operator
                                    'parameters':dictionary of parameters for that interaction type

        Returns
        -------
        None.

        """
        self.attributes = attributes # dictionary of attributes
            

class FiboSim():
    """
    Simulator object
    """
    
    def __init__(self,d,N,L):
        """
        initialize object

        Parameters
        ----------
        d : int (>0)
            (physical) Hilbert-space dimension
        N : int (>0)
            Level of Fibonacci approximation to Golden ratio
        L : int (>0)
            truncation length (sites truncated between -L<=n1<=L 

        Returns
        -------
        None.

        """
        self.N=N
        self.L=L
        self.d=d
        FN = Fibonacci(N) 
        FNp1 = Fibonacci(N+1)
        self.omega = np.array([1,FN/FNp1]) # frequency vector
        
        # rotation matrix
        self.W = np.array([[FNp1,FN],
                      [FN,-FNp1]])
        
        self.site_dict = {} # dictionary of sites, keys = int index, entries = tuples of site-vectors
        ctr = 0 # counter
        
        """
        Construct the reduced zone of the frequency lattice:
        idea: it's hard to specify exactly which points on the frequency lattice 
        lie within the first zone since the width of the strip has some quasiperiodic modulation
        so, just pad to be safe, and only record the points that fall within the zone
        """
        vert = FN+FNp1-1 # vertical width of reduced zone strip
        for n1 in range(-L,L+1):
            # define approximate strip boundaryies for each n1
            n2min = np.int(np.ceil(n1*FN/FNp1))-1 # increases with average F(N)/F(N+1)
            n2max = n2min+vert
            for n2 in range(n2min,n2max+1):
                x = np.array([n1,n2])
                x_fold = Fold(x,self.W)
                if np.sum(np.abs(x_fold-x) == 0): # site is in reduced zone
                    self.site_dict[ctr]=tuple(x) # record the site in the dictionary
                    ctr+=1 # increment the total number of sites
        
        self.size = np.copy(ctr) # record total number of sites
        self.inv_site_dict = {v: k for k, v in self.site_dict.items()} # reversed dictionary (site vector -> index)

        # setup Hilbert space
        self.state_dict = {} # keys = index (integer); values = tuple (physical state, freq_1, freq_index_2)
        ctr=0
        for k in self.site_dict.keys():
            for j in range(d):
                self.state_dict[ctr] = (j,self.site_dict[k][0],self.site_dict[k][1])
                ctr+=1
        self.Hsize = np.copy(ctr) # Hilbert space size
        self.inv_state_dict = {v: k for k,v in self.state_dict.items()} # inverse state dictionary
        
        
        # initialize Hamiltonian
        self.interactions = [] # list of interaction pulses
        self.K = np.zeros([self.Hsize,self.Hsize],dtype=complex) # quasi-Floquet-zone effective Hamiltonian, K 

    def setup_K(self,interactions=[]):
        """
        

        Parameters
        ----------
        pulses : list of Interactions, optional
            DESCRIPTION. The default is [].

        Returns
        -------
        None.

        """
        
        # drive-indpendent diagonal terms
        for s in self.state_dict.keys(): # loop over extended Hilbert space
            v = self.state_dict[s]
            n_s = np.array([v[1],v[2]]) # frequency lattice point for s^th state
            self.K[s,s] += np.dot(self.omega,n_s) # add diagonal part of K
            
        # other terms
        for v in self.interactions:
            if v.attributes['type']=='Gaussian':
                self.add_gaussian_pulse(v)
            else:
                raise NotImplementedError('Interaction type %s not implemented')
    
    def add_gaussian_pulse(self,pulse):
        """
        adds a Gaussian pulse to K

        Parameters
        ----------
        pulse : Interaction object
            must be gaussian pulse.

        Returns
        -------
        None.

        """
        operator = pulse.attributes['operator'] # operator associated with pulse
        freq = pulse.attributes['parameters']['freq'] # 0 or 1, which frequency
        width = pulse.attributes['parameters']['width'] # pulse width as fraction of period, defined as sigma
        weight = pulse.attributes['parameters']['weight'] # integrated weight of pulse
        cutoff = pulse.attributes['parameters']['cutoff'] # maximum change in frequency index (e.g. pulse effects: n-> n-cutoff,...n+cutoff)

        # setup pulse
        pulse_amplitudes = [GaussianFT(width,weight,m) for m in range(-cutoff,cutoff+1)]
        for site in self.site_dict.values():
            n = np.array(site)
            for m in range(-cutoff,cutoff+1): # loop over harmonics of pulse
                # find the frequency connected by the mth harmonic
                n2 = n.copy()
                n2[freq]+=m
                
                # fold that back into the reduced zone
                n2_folded = Fold(n2,self.W).astype(int)
                
                # check if that regular and folded final state are both part of the truncated Hilbert space
                if (np.abs(n2_folded[0])<=self.L) and (np.abs(n2[0])<=self.L): 
                    # if so, add the operator for the component
                    state_index = self.inv_state_dict[(0,n2_folded[0],n2_folded[1])]
                    self.K[state_index:state_index+self.d,state_index:state_index+self.d] += pulse_amplitudes[m]*operator
                
    def plot_freq_lattice(self):
        """
        Scatter plot of kept frequencies

        Returns
        -------
        None.

        """
        plt.figure()
        for n1 in range(-L,L+1):
            for n2 in range(-L*Fibonacci(self.N),L*Fibonacci(self.N)+1):
                n = np.array([n1,n2])
                n_folded = Fold(n,self.W)
                if np.sum(np.abs(n_folded-n))==0:
                    plt.scatter(n1,n2,s=20,c='blue')
                else: 
                    plt.scatter(n1,n2,s=20,c='gray')
                    
                if (n1,n2) in self.inv_site_dict.keys():
                    plt.scatter(n1,n2,s=20,c='red',marker='x')
                #else:
                
#%% dev
d=2
N=3
L=5
sim = FiboSim(d,N,L)

# define the drive Hamiltonian
# define gaussian pulse                  
pulse_parameters = {'freq':0,
                    'width':0.5,
                    'weight':np.pi,
                    'cutoff':5}
pulse = Interaction({'type':'Gaussian',
                     'operator':PAULI['X'],
                     'parameters': pulse_parameters})
        
sim.interactions += [pulse]
sim.setup_K()
#sim.plot_freq_lattice()

# simulate the quasi-energy spectrum
