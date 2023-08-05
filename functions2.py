# +
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

from scipy.integrate import quad, simps
from scipy.linalg import solve_banded
import cmath
import scipy.optimize as opt


# -

def func(x, tau): # Week 5
    """
    Exponential function with parameter tau.
    """
    return np.exp(-x/tau)

def autocorrelation(A, h):
    """
    Function that takes as argument an array A of the measurements of a physical observable in all timesteps
    (what we call instantaneous measurements) and computes and plots its autocorrelation function.
    It also returns the mean value of all the measurements <A>,
    its correlation time τ and its standard deviation sigmaA. !!! Note that the array A must only include measurements
    of the physical quantity AFTER the rescaling has been already performed and equilibrium has been achieved.
    """
    lpn = len(A) - 1
    chi = np.zeros((lpn,))
    xdata = np.linspace(0,h*(lpn-1),lpn)
    
    for m in range(lpn): # We correlate all times -m (referring to the time of the autocorrelation function)
                         # times -n (timesteps in the array A)
        
        B = np.sum(A[: lpn+1 - m] * A[m:])
        C = A[: lpn+1 - m].sum() * A[m:].sum()
        D = (A[:lpn+1 - m]**2).sum()
        E = (A[:lpn+1 - m].sum())**2
        F = (A[m:]**2).sum()
        G = (A[m:].sum())**2
        
        chi[m] = ((lpn+1 - m)*B - C) / (np.sqrt((lpn+1 - m)*D - E) * np.sqrt((lpn+1 - m)*F - G))
        
    tau, pcov = opt.curve_fit(func, xdata, chi) # chi = ydata
        
    plt.plot(xdata[:100], chi[:100], 'b-', label='measured χ(t)') # We plot only the first 100 elements
                                                                  # assuming tau not greater than 100*h
    plt.plot(xdata[:100], func(xdata[:100], tau), 'r-', label=f'fitted χ(t), τ = {tau[0]}')
    plt.title('Autocorrelation function as a function of time')
    plt.xlabel('t (t*)')
    plt.ylabel('autocorrelation function χ')
    plt.legend()
    plt.show()
    
    Amean = np.mean(A)
    A2mean = np.mean(A**2)
    
    sigmaA = np.sqrt(2 * (tau/h) * (A2mean - Amean**2)/(lpn+1)) # Note that we divide tau by the timestep h because
                                                                # in our definition of tau it has units

    return Amean, tau, sigmaA[0]

def ham_1D_well(V, dx):
    
    """
    Function that provided a potential along the grid, can return the hamiltonian matrix that corresponds to this potential, including the kinetic term.
    Likewise, it applies the boundary conditions of the 1D infinite well, psi(0) = psi(L) = 0. The potential inside the well wwill initially be zero,
    but by defining it enables as to vary it later.
    """
    # Potential be a N+1 long array
    a = np.size(V)
    
    pot = np.diag(V, 0) # potential term of the hamiltonian matrix
    
    A = -2 * np.ones(a)
    B = np.ones(a-1)
    
    A1 = np.diag(A, 0)
    B1 = np.diag(B, 1)
    B2 = np.diag(B, -1)
    
    kin = - (0.5/(dx)**2) * (A1 + B1 + B2) # kinetic term of the hamiltonian matrix
    
    ham = kin + pot
    
    # Applying the Dirichlet boundary conditions of the infinite 1D well
    
    ham[0,0] = 1
    ham[0,1] = 0
    ham[-1,-1] = 1
    ham[-1,-2] = 0
    
    return ham


def normal_simps(psi, dx):
    
    """
    Function which, provided a wavefunction, normalizes it using Simpson's rule for integration
    """
    
    a = len(psi)
    
    # Creating the array that will help create the sequence we sum in Simpson's rule
    b = np.zeros((a,))
    c = np.zeros((a,))
    b[::2] = 1
    c[1::2] = 1
    b = 2*b
    c = 4*c
    d = b + c
    d[0] = 1
    d[a-1] = 1
    
    psi_magn = (np.absolute(psi))**2
    integral = dx * (d @ psi_magn)/3 #Turning the normalization integral into a sum according to Simpson's rule
    #print(integral)
    
    psi = psi/np.sqrt(integral)
    
    return psi


def crank_nicolson1(psi, V, N, dx, iterations):
    
    """
    Function which, provided with an initial configuration psi, gives the evolution in time of the wavefunction for 1D infinite well.
    """
    
    Dt = 2/(iterations) # Adjusted to the value that leads to the best results
    psi_tracker = np.zeros((N+1, iterations+1), dtype = complex)
    psi_tracker[:,0] = normal_simps(psi, dx)
    
    H = ham_1D_well(V, dx)
    
    for i in range(iterations):
    
        psi_tracker[:, i+1] = np.dot(np.linalg.inv(1*np.eye(N+1) + 1j*Dt*H), np.dot((1*np.eye(N+1) - 1j*Dt*H), psi_tracker[:,i]))
        psi_tracker[:, i+1] = normal_simps(psi_tracker[:, i+1], dx) # Normalize after each computation
        
    return psi_tracker


def crank_nicolson3(psi, V, N, dx, iterations, b):
    
    """
    Function which, provided with an initial configuration of psi and V, gives the evolution in time of the wavefunction for 1D infinite SPLITTING well. Likewise, it returns the potential at each time t, the stationary eigenenergies of this potential and the actual computed energies of the system (& error) at time t.
    """
    # b = Height the wall obtains per iteration (typically b=1 for adiabatic, b=30 for non- adiabatic
    
    Dt = 8/(iterations) # Adjusted to the value that leads to the best results
    
    psi_tracker = np.zeros((N+1, iterations+1), dtype = complex)
    V_tracker = np.zeros((N+1, iterations+1))
    eigen_tracker = np.zeros((iterations+1,))
    energy_tracker = np.zeros((iterations+1,))
    error_tracker = np.zeros((iterations+1,))
    
    psi_tracker[:,0] = normal_simps(psi, dx)
    
    H = ham_1D_well(V, dx)
    w, v = np.linalg.eigh(H)
    eigen_tracker[0] = w[2] # Energy eigenvalues of the given potential if it were to be stationary (ground state!)
    
    energy_tracker[0] = np.dot(np.transpose(np.conjugate(psi_tracker[:, 0])), np.dot(H, psi_tracker[:, 0])) * dx # Computed actual energy at time t of the time dependent WF
    error_tracker[0] = energy_error(psi_tracker[:,0], H, N, dx)
    
    epsilon = N//20 # Region around the center of the grid where the potential is time dependent --> The potential wall is being erected
    delta = N//200 # Asymmetry against the right side --> The wall extends further towards the right section of the well
    
    for i in range(iterations):
    
        V[(N//2 - epsilon) : (N//2 + epsilon + delta)] += b
        V_tracker[:,i+1] =  V
        
        H = ham_1D_well(V, dx)
        w, v = np.linalg.eigh(H)
        eigen_tracker[i+1] = w[2] # Energy eigenvalues of the given potential if it were to be stationary (ground state!)
        
        psi_tracker[:, i+1] = np.dot(np.linalg.inv(1*np.eye(N+1) + 1j*Dt*H), np.dot((1*np.eye(N+1) - 1j*Dt*H), psi_tracker[:,i]))
        psi_tracker[:, i+1] = normal_simps(psi_tracker[:, i+1], dx) # Normalize after each computation
        
        energy_tracker[i+1] = np.dot(np.transpose(np.conjugate(psi_tracker[:, i+1])), np.dot(H, psi_tracker[:, i+1]))/(N+1) # Computed actual energy at time t of the time dependent WF
        error_tracker[i+1] = energy_error(psi_tracker[:,i+1], H, N, dx)
        
    return psi_tracker, V_tracker, eigen_tracker, energy_tracker, error_tracker


def energy_error(psi, H, N, dx):
    
    der_psi = np.zeros((N+1,))
    
    for i in range(N):
        der_psi[i] = (psi[i+1] - psi[i])/dx
    
    der_psi[N] = der_psi[N-1]
    
    A = np.dot(H, psi)
    dE = 2 * dx**2 * np.sqrt(der_psi**2 @ A**2)
    
    return dE
