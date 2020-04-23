"""
Stroboscopic Floquet drive, Spin-chain simulator
"""
#%% imports
import numpy as np # generic math functions
import numpy.linalg as la
import scipy.linalg as linalg
import matplotlib.pyplot as plt

#%% functions
def array_index(index, L):
    """
    

    Parameters
    ----------
    index : int
        index of state, between 0 and 2**(L-1)-1.
    L : int
        length of spin-chain (must be >1)
    sector: boolean
        symmetry sector (even=False, odd=True)

    Returns
    -------
    np array of binary variables representing spin-configuration for that index
    with 1 for up 0 for down (in z-basis).
    
    Note: spins are ordered by sites 0 to L 
    """
    
    bin_string = np.binary_repr(index, width=L) # convert state index to binary string
    bin_array = np.array([np.int(s) for s in bin_string[::-1]]) # change to numpy array
    return bin_array

def find_index(state_array,L):
    """
    returns 

    Parameters
    ----------
    state_array : binary np.array of length L
        numpy array representing spin-configuration (e.g. output of array_index).
    L : int, >1
        length of spin-chain.

    Returns
    -------
    index, 0 <= int < 2**L.

    """  
    #parity = np.mod(np.sum(state_array),2)
    two_powers = np.array([2**j for j in range(L)]) # np.array of powers of 2 to convert binary to 
    return np.dot(two_powers,state_array)#

#%% Ising Hamiltonian class
class Hamiltonian(object):
    """
    Hamiltonian (in X-basis)
    """
    
    def __init__(self,L,sector=False):
        """
        initialize blank Hamiltonian

        Parameters
        ----------
        L : int (>1)
            length of spin-chain.
        sector : boolean, optional
            symmetry-sector . The default is False = even.

        Returns
        -------
        None.

        """
        self.L=L
        self.D = 2**(L) # Hilbert space dimension
        self.H = np.zeros([self.D,self.D],dtype=complex) # Hamiltonian (even sector)
        
    def add_coupling(self,coupling_type, coupling_constants):
        """
        

        Parameters
        ----------
        coupling_type : string
            type of interactions/couplings (e.g. 'ZZ', 'X', 'XX',etc...).
            must preserve Ising symmetry
        coupling_constants : numpy array of length L
            coupling constants (note, for two-spin interactions, 
                                last index is for periodic boundary conditions).

        Returns
        -------
        None.

        """
        if coupling_type == 'Z':
            for j in range(self.D): #loop over states
                self.H[j,j]+= np.dot(coupling_constants,2*array_index(j,self.L)-1)
        
        if coupling_type == 'ZZ':
            for j in range(self.D): #loop over states
                Zs = 2*array_index(j,self.L)-1 # +/- valued array of spin states
                Zs_shifted = np.roll(Zs,-1)
                ZZs = Zs*Zs_shifted
                self.H[j,j]+= np.dot(coupling_constants,ZZs)
                
        if coupling_type == 'XX':
            for j in range(self.D): #loop over states
                psi_i = array_index(j,self.L) # initial state
                for x in range(L): # loop over sites
                    # construct state that is psi_i but with two flipped spins on sites x, x+1
                    psi_f = psi_i.copy()
                    psi_f[x] = 1-psi_f[x] 
                    y=np.mod(x+1,self.L)
                    psi_f[y]=1-psi_f[y]
                    
                    # find the index of that state
                    j_prime = find_index(psi_f,self.L)
                    
                    # update the appropriate entries of H
                    self.H[j,j_prime]+= coupling_constants[x]
                    
    def unitary(self,t=1.0):
        """
        

        Parameters
        ----------
        t : float
            t interval that H is applied for (defaults to 1).

        Returns
        -------
        e^{-iHt} for specified time-interval.

        """
        return linalg.expm(-1j*t*self.H)

#%% Stroboscopic Floquet dynamics simulation
class Sim_Object(object):
    """
    Simulates stroboscopic evolution of Ising chain
    """
    
    def __init__(self,L,sector=0):
        """
        

        Parameters
        ----------
        L : positive int (>1)
            length of spin-chain.

        Returns
        -------
        None.

        """
        self.L = L # chain length
        self.D = 2**(L) # Hilbert space dimension
        self.Hs = [] # list of Hamiltonians
        self.F = np.eye(self.D,dtype=complex)
        self.evecs = np.eye(self.D,dtype=complex)
        self.evals = np.eye(self.D,dtype=complex)
        
    def compute_F(self):
        """
        

        Returns
        -------
        Nothing, but updates Floquet operator and diagonalizes it

        """
        self.F = np.eye(self.D,dtype=complex) # reset F
        
        # multiply out unitaries for each stroboscopic drive step
        for H in self.Hs:
            self.F = np.dot(H.unitary(),self.F)
            
        # diagonalize F
        self.evals, self.evecs = la.eig(self.F)
        
    def autocorrelations_timesweep(self,ts,
                                   basis='X',
                                   inital_state=-1):
        """
        

        Parameters
        ----------
        ts : np.array of floats
            times at which to measure correlators.
        basis : str, optional
            'X','Y', or 'Z' basis. The default is 'X'.
        inital_state : int, optional
            index of initial state. Default = -1 gives average over spectrum

        Returns
        -------
        None.

        """
        if basis != 'X':
            raise NotImplementedError('Only X basis measurements implemented')
        
        # initialize results dictionary
        results = {} # dictionary of results
        results['basis']=basis
        results['times']=ts
        results['correlators'] = {}
        results['correlators']['X'] = np.zeros([self.L,len(ts)])
        results['correlators']['Xalt'] = np.zeros([self.L,len(ts)])
        results['correlators']['Z'] = np.zeros([self.L,len(ts)])
        
        # construct observable operators
        Xops = [np.zeros([self.D,self.D],dtype=complex) for j in range(self.L)] # X operator for every site
        Zops = [np.zeros([self.D,self.D],dtype=complex) for j in range(self.L)] # Z operator for every site
        for j in range(self.L): # loop over sites
            for s in range(self.D): # loop over states
                # Z operator  for each site  
                Zops[j][s,s] = 2*(array_index(s,self.L)[j])-1 # fill in diagonal entires of X
                
                # X operator for each site
                s2_array = array_index(s,self.L)
                s2_array[j]=1-s2_array[j] # flip jth spin
                s2 = find_index(s2_array, self.L)
                
                Xops[j][s,s2] = 1
         
        # rotate observables into Floquet eigenbasis
        Xops_eig = Xops.copy()
        Zops_eig = Zops.copy()
        for j in range(self.L): 
            Xops_eig[j] = self.evecs.conj().T @ Xops[j] @ self.evecs
            Zops_eig[j] = self.evecs.conj().T @ Zops[j] @ self.evecs
                
        
        # compute correlation function (time sweep)
        for i_t in range(len(ts)): # loop over times
            t = ts[i_t]
            Ft_eig = np.diag(self.evals**t)  # F^t (in Floquet-eigenbasis)
            
            for j in range(self.L): # correlation function for each site
                #results['correlators']['X'][j,i_t] = np.trace(Ft.conj().T @ Xops[j] @ Ft @ Xops[j])/self.D
                results['correlators']['X'][j,i_t] = np.trace(Ft_eig.conj().T @ Xops_eig[j] @ Ft_eig @ Xops_eig[j])/self.D
                
                
                results['correlators']['Z'][j,i_t] =  np.trace(Ft_eig.conj().T @ Zops_eig[j] @ Ft_eig @ Zops_eig[j])/self.D
                #np.trace(Ft_eig.conj().T @ Zops_eig[j] @ Ft_eig @ Zops_eig[j])
    
  
        return results
        
#%% testing and debug
L = 6
ts = [t for t in range(0,100,2)]

# XX-pulse
Jx = 0.95
Jxs = (Jx)*np.pi/2*np.append(np.ones(L-1),0)
H1 = Hamiltonian(L)
H1.add_coupling('XX',Jxs)

# Z-pulse
hzs = 2*np.pi*np.random.rand(L)
H2 = Hamiltonian(L)
H2.add_coupling('Z',hzs)

# ZZ-pulse
Jz = 0.15
Jzs = (Jz)*np.pi/2*np.append(np.ones(L-1),0)
H3 = Hamiltonian(L)
H3.add_coupling('ZZ',Jzs)

# setup simulation
sim = Sim_Object(L)
sim.Hs = [H1,H2,H3]
sim.compute_F()
results = sim.autocorrelations_timesweep(ts)

#%% plots
colors = plt.cm.gnuplot(np.linspace(0,0.8,L))

plt.figure()
fig = plt.gcf()
plt.subplot(1,2,1)
for j in range(L):
    plt.plot(ts, results['correlators']['X'][j,:], color = colors[j])
plt.legend([j for j in range(L)])
plt.title('X')
plt.subplot(1,2,2)
for j in range(L):
    plt.plot(ts,  results['correlators']['Z'][j,:], color=colors[j])
plt.legend([j for j in range(L)])
plt.title('Z')
fig.suptitle('(L,Jx,Jz) = %s (couplings implemented *pi/2), symmetry-axis=Z' %np.array([L,Jx,Jz]))
