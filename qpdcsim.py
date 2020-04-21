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

def Fold(x,W,give_folding_vec=False):
    """
    Folds a generic point in ZxZ frequency lattice back into the reduced zone

    Parameters
    ----------
    x : np.array of length 2, and type int
        original lattice position
    W : GL(2,Z) matrix as 2x2 integer np.array 
        Rotates tilted lattice into original one 
    give_folding_vec: bool (optional)
        if true => returns number of translations used to fold back into zone
        
    Returns:
    -------
    x_folded = x folded back into reduced zone
    
    """
    # shift before folding so that reduced zone includes (0,0), 
    # and goes through upper right and lower left quadrants
    shift = W[1,:]-np.array([1,0]) 
    y = la.solve(W,x+shift) # y = W^{-1}x, coordinate in tilted lattice
    
    y_folded = [y[0],np.mod(y[1],1)] # fold back into zone
    
    # rotate back to original lattice (return as int, and undo shift)
    if give_folding_vec:
        return np.around(W@y_folded-shift).astype(int), (y[1]-y_folded[1]).astype(int)
    else:        
        return np.around(W@y_folded-shift).astype(int)
    
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

    def setup_K(self,flux=0):
        """
        

        Parameters
        ----------
        pulses : list of Interactions, optional
            DESCRIPTION. The default is [].
        
        flux : float in [0,2pi)
            flux through compactified direction of strip
            
        Returns
        -------
        None.

        """
        self.K = np.zeros([self.Hsize,self.Hsize],dtype=complex)
        
        # drive-indpendent diagonal terms
        for s in self.state_dict.keys(): # loop over extended Hilbert space
            v = self.state_dict[s]
            n_s = np.array([v[1],v[2]]) # frequency lattice point for s^th state
            self.K[s,s] += np.dot(self.omega,n_s) # add diagonal part of K
            
        # other terms
        for v in self.interactions:
            if v.attributes['type']=='Gaussian':
                self.add_gaussian_pulse(v,flux)
            elif v.attributes['type']=='cos':
                self.add_cos_drive(v,flux)
            elif v.attributes['type']=='constant':
                self.add_constant_term(v)
            else:
                raise NotImplementedError('Interaction type %s not implemented')
    
    def add_gaussian_pulse(self,pulse,flux=0):
        """
        adds a Gaussian pulse to K

        Parameters
        ----------
        pulse : Interaction object
            pulse.['attributes']['parameters'] must contain:
                'freq': int = 0,1 which frequency axis the pulse is on (e.g. if 0 then train of gaussian pulses with period 1, if 1, then period 1/PHI)
                'width': float, >0, gaussian width (std-dev)
                'weight': float, integrated weight of gaussian
                'cutoff': int >0, cut off frequency shifts by larger magnitude than cutoff
        
        flux : float between 0 and 2pi
            flux through compactified direction of reduced zone strip
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
                # keep track of the folding vector so that we can apply an appropriate phase
                n2_folded,shift = Fold(n2,self.W, give_folding_vec=True)
                phase = np.exp(1j*flux*shift) # pick up flux each time you go around the compact direction 

                key = (0,n2_folded[0],n2_folded[1])
                # check if that regular and folded final state are both part of the truncated Hilbert space
                if key in self.inv_state_dict.keys():
                    # if so, add the operator for the component
                    state_index = self.inv_state_dict[key]
                    self.K[state_index:state_index+self.d,state_index:state_index+self.d] += phase * pulse_amplitudes[m]*operator
    
                
    def add_constant_term(self,interaction_term):
        """
        adds a constant, time-independent term (uniform on frequency lattice)

        Parameters
        ----------
        interaction_term : interaction

        Returns
        -------
        None.

        """
        for site in self.site_dict.keys():
            n = self.site_dict[site]
            state_index = self.inv_state_dict[(0,n[0],n[1])]
            self.K[state_index:state_index+self.d,state_index:state_index+self.d] += interaction_term.attributes['operator']
    
    def add_cos_drive(self,pulse,flux=0):
        """
        adds A*cos(wt+phi) type-drive

        Parameters
        ----------
        pulse : Interaction object
            pulse.['attributes']['parameters'] must contain:
                'freq': int = 0,1 which frequency axis the pulse is on (e.g. if 0 then train of gaussian pulses with period 1, if 1, then period 1/PHI)
                'amplitude': float, >0, amplitude
                'phase': float, phase offset
                'cutoff': int >0, cut off frequency shifts by larger magnitude than cutoff
        
        flux : float between 0 and 2pi
            flux through compactified direction of reduced zone strip
        Returns
        -------
        None.
        """
        operator = pulse.attributes['operator'] # operator associated with pulse
        freq = pulse.attributes['parameters']['freq'] # 0 or 1, which frequency
        amplitude = pulse.attributes['parameters']['amplitude'] # pulse width as fraction of period, defined as sigma
        phase_shift = pulse.attributes['parameters']['phase'] # integrated weight of
        
        for site in self.site_dict.values():
            n = np.array(site)
            for m in [-1,1]: # loop over harmonics of pulse
                # find the frequency connected by the mth harmonic
                n2 = n.copy()
                n2[freq]+=m
                
                # fold that back into the reduced zone
                # keep track of the folding vector so that we can apply an appropriate phase
                n2_folded,shift = Fold(n2,self.W, give_folding_vec=True)
                overall_phase = np.exp(1j*(flux*shift+m*phase_shift)) # pick up flux each time you go around the compact direction 

                key = (0,n2_folded[0],n2_folded[1]) # index of site that's connected by this harmonic
                # check that new-site is in truncated strip
                if key in self.inv_state_dict.keys():
                    # if so, add the operator for the component
                    state_index = self.inv_state_dict[key]
                    self.K[state_index:state_index+self.d,state_index:state_index+self.d] += overall_phase * amplitude/2*operator
    
            
    
    def compute_spectrum(self,flux=0):
        """
        

        Parameters
        ----------
        flux : float between 0 and 2pi
            flux through compactified direction of reduced zone strip

        Returns
        -------
        eigenvalues, eigenvectors in extended zone

        """
        self.setup_K(flux)
        return la.eigh(self.K)
    
    def plot_freq_lattice(self):
        """
        Scatter plot of kept frequencies

        Returns
        -------
        None.

        """
        plt.figure()
        for n1 in range(-self.L,self.L+1):
            for n2 in range(-self.L*Fibonacci(self.N),self.L*Fibonacci(self.N)+1):
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
# d=2
# N=3
# L=5
# sim = FiboSim(d,N,L)



# # define the drive Hamiltonian
# # define gaussian pulse                  
# pulse1 = Interaction({'type':'Gaussian',
#                      'operator':PAULI['Z'],
#                      'parameters': {'freq':0,
#                                     'width':0.5,
#                                     'weight':np.pi/2,
#                                     'cutoff':1}
#                      }
#                     )
# pulse2 = Interaction({'type':'Gaussian',
#                      'operator':PAULI['Z'],
#                      'parameters': {'freq':1,
#                                     'width':0.5,
#                                     'weight':np.pi/2,
#                                     'cutoff':5}
#                      }
#                     )

        
# sim.interactions = [pulse1,pulse2]
# #sim.setup_K()

# N_sweep = 11
# fluxes = np.linspace(-np.pi,np.pi,N_sweep)
# Es =np.zeros([N_sweep,sim.Hsize])
# for j in range(N_sweep):    
#     Es[j,:],psi=sim.compute_spectrum(flux=fluxes[j])
# #sim.plot_freq_lattice()

# # simulate the quasi-energy spectrum
# plt.figure()
# ylim = 2*np.pi/Fibonacci(N+1)
# plt.plot(fluxes,Es)
# plt.ylim(-ylim,ylim)
# plt.title('Width = %s' % Fibonacci(N) + ', L=%s'%L)