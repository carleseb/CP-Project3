# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

from scipy.integrate import quad, simps
from scipy.linalg import solve_banded
import cmath
from matplotlib import animation
from matplotlib.animation import PillowWriter
from functions2 import *

# +
###########################################################################################################################################################################
#                                                                            SYSTEM 1                                                                                     #
###########################################################################################################################################################################

# Discretize domain
N = 200
L = 1
dx = L/N
x = np.linspace(0,L,N+1)


# -

# Define shape of potential, now we can also choose the turning point of the theta (x_t variable)
def V_h(x,a,x_t):
    return a*np.heaviside(x-x_t,0.5) # this last one is the value exactly at the turning point x_t


# We print it to check it looks correct
a = 10000
x_t = 0.5
plt.plot(x, V_h(x,a,x_t), label= "x_t=0.5")
plt.plot(x, V_h(x,a,0.55), label= "x_t=0.55")
plt.plot(x, V_h(x,a,0.6), label= "x_t=0.60")
plt.legend()

# We solve for the stationary problem
diag = V_h(x, a, x_t) + 1/dx**2
off_diag = ((-1/2)/dx**2)*np.ones(N-2)
eigenenergies, eigenvector_matrix = eigh_tridiagonal(diag[1:-1], off_diag)

# +
# We pad our eigenvector and normalize it
v_0 = np.pad(eigenvector_matrix.T[0], 1)
II = simps(np.abs(v_0)**2, x, dx)
v_0_norm = (1/np.sqrt(II))*v_0
plt.plot(x, v_0_norm)

print(eigenenergies[0])
# -

# Check normalization
I = simps(np.abs(v_0_norm)**2, x, dx)
print(I)

# +
# Now we want the potential to move to the positive x axis and the wavefunction to converve with it (if slow enough)
# The speed at which the finite potnetial barrier moves is v. We observe:
# For v = 0.01 or lower the system stays adiabatic
# For v = 100 or higher the system stays non-adiabatic
# In the values in between there is the turning point adiabatic - non-adiabatic

iterations = 1000 #original 1000
psi_tracker = np.zeros((N+1, iterations+1), dtype = complex)
psi_tracker[:,0] = v_0_norm
v = 0.001

for t in range(iterations):
    
    # Step 1, compute b = B*v_0_norm
    dt = 0.001
    r = dt*1j/(2*dx*dx)
    Pot = V_h(x[1:-1],a, x_t+v*t)
    B = np.diag(np.ones(N-2)*r,-1)+ np.diag(np.ones(N-1)*(1-2*r))+ np.diag(-0.5*Pot*1j*dt) +np.diag(np.ones(N-2)*r,+1)
    b = np.matmul(B,v_0_norm[1:-1])

    # Step 2, compute coefficients alpha and beta
    alpha = np.zeros(N, dtype = complex) # from 0 to N-1
    beta = np.zeros(N, dtype = complex)
    A0 = 1+(2*r)+Pot*1j*dt
    Aplus = -r
    Aminus = -r
    for k in np.flip(np.arange(N-1)): # k from N-2 (1998) to 0
        denominator = A0[k]+(Aplus*alpha[k+1])
        alpha[k] = (-Aminus)/denominator
        beta[k] = (b[k]-(Aplus*beta[k+1]))/denominator

    # Step 3, compute the wavefunction values: v_fin
    v_fin = np.zeros(N+1, dtype = complex)
    for k in range(N-1): # k from 0 to N-2
        v_fin[k+1] = alpha[k]*v_fin[k]+beta[k]

    # Step 4, normalize
    integral = simps(np.abs(v_fin)**2, x, dx)
    v_fin_norm = (1/np.sqrt(integral))*v_fin
    
    # Step 5, rename wavevector as old to iterate again
    v_0_norm = v_fin_norm
    psi_tracker[:,t+1] = v_0_norm

# +
# This code creates an animation of all the wavevectors stored in psi_tracker
# Parameters have to be modified if we change the number of iterations or any other relevant input for the figure

fig, ax = plt.subplots(1, 1, figsize = (8,4))
ln1, = plt.plot([],[], label= '$|\psi(\~x, \~t)|^2$') # empty line 1, nothing shows
ln2, = plt.plot([],[], label= '$V(\~x)$')
time_text = ax.text(0.65, 0.95, '', fontsize = 16, transform = ax.transAxes,
                    bbox = dict(facecolor = 'white', edgecolor = 'black'))
ax.set_xlim(0, 1)
ax.set_ylim(0, 10)
ax.set_ylabel('$|\psi(\~x, \~t)|^2$')
ax.set_xlabel('$\~x$')
ax.legend()

def animate(i): # need to update every frame (i is framenumber)
    ln1.set_data(x, np.abs(psi_tracker[:,i])**2)
    ln2.set_data((x_t+v*i, x_t+v*i),(0, 10))
    time_text.set_text('t={:.3f}'.format(i*dt))
    

ani = animation.FuncAnimation(fig, animate, frames = 1000, interval=50)
ani.save('time_evo.gif', writer = 'pillow',
         fps = 20, dpi = 100)  # we save it in our computer (specify fps) 20, 100

# +
# Now we compute the wavefunction of the infinite well without any potential, which would have to coninde with the final state
# after the adiabatic evolution.
# Accordingly, energy of this ground state should be different from the initial energy, specifically smaller.

diag = (1/dx**2)*np.ones(N+1)
off_diag = ((-1/2)/dx**2)*np.ones(N-2)
eigenenergies_f, eigenvector_matrix_f = eigh_tridiagonal(diag[1:-1], off_diag)
print(eigenenergies_f[0])

# We also print the final wavefunction if we want (see it is complex)
# print(psi_tracker[:,1000])
# -

# We normalize
final_gs= np.pad(eigenvector_matrix_f.T[0], 1)
ii = simps(np.abs(final_gs)**2, x, dx)
final_gs = (1/np.sqrt(ii))*final_gs

# +
# We can compare the three square amplitudes of the wavefunctions (ground states) in order to see they are the same
# (or similar):
# 1) The wavefunction coming out of the adiabatic evolution
# 2) The wavefunction coming out of solving the eigenproblem
# 3) The analytical wavefunction for this problem

plt.plot(x, np.abs(psi_tracker[:,1000]**2), label = 'final $|\psi(\~x, \~t)|^2$')
plt.plot(x, np.abs(final_gs)**2, label = 'numerical final solved eigenfunction')
plt.plot(x, 2*np.abs(np.sin(np.pi*x))**2, ':', label = 'numerical final analytical eigenfunction')
plt.ylabel('$|\psi(\~x, \~t)|^2$')
plt.xlabel('$\~x$')
plt.legend()

# +
# Now we would wish to estimate the real energy of the wavefunction coming out of our evolution. In the adiabatic case,
# the final energy should be:
# - Similar to the previous one computed for the eigenproblem (within errorbars)
# In the non-adiabatic case energy should be:
# - Similar to the energy we started with (conserved)
# Since this last case does NOT belong to a single eigenstate, the energy will be calculated by means of an expectation value
# We will use the discretization of the derivative in order to perform the integral (trapezoids)

# We have the wavefunctiona nd we pad it with two zeros more
wf = np.pad(psi_tracker[:,100], 1)
ene1 = 0
integrand = np.zeros(N+1)

for i in range(N+1):
    integrand[i] = np.real((-1/2)*np.conjugate(wf[i])*(wf[i+1]-2*wf[i]+wf[i-1])/(dx**2)) # we make the expression real
    #print(integrand[i])
    
    ene1 += integrand[i]*dx # Since we do not need a lot of precision we can do with simple trapezoids
    
print(ene1)

# +
# We can also plot the energy evolution (iterating for different k for psi_tracker[:,k])
ene1 = np.zeros(iterations+1)

for k in range(iterations):
    
    wf = np.pad(psi_tracker[:,k], 1)
    integrand = np.zeros(N+1)

    for i in range(N+1):
        integrand[i] = np.real((-1/2)*np.conjugate(wf[i])*(wf[i+1]-2*wf[i]+wf[i-1])/(dx**2)) # we make the expression real
        #print(integrand[i])

        ene1[k] += integrand[i]*dx # Since we do not need a lot of precision we can do with simple trapezoids
# -

# We plot the energy evolution
plt.plot(ene1[0:1000])

# We take the last 500 energies and make an estimation of the final energy with error
Emean, tau, Esigma = autocorrelation(ene1[500:1000], dx)
print(Emean, Esigma)



# +
###########################################################################################################################################################################
#                                                                            SYSTEM 2                                                                                     #
###########################################################################################################################################################################

# Now we will use Crank - Nicolson method to observe the time evolution of the eigenstates as well as superpositions of theirs and compare 
# them with the theoretical predictions.

# Now we are ready to start implementing the time dependent potential. To do so we need to manipulate the Crank Nicolson function since the hamiltonian matrix changes in each iteration.
# We will work on generating a slowly erecting wall of potential in the middle of our domain and study the adiabatic and non-adiabatic behaviour of the system (1D Well).
# To do so, the Hamiltonian needs to be recomputed in each Crank Nicolson loop, since the potential is altered by a term of b*(iteration_number) on the elements in the middle of our domain.
# b is an adjustable constant and the elements of the grid chosen to carry out the time evolution of the potential are centered around (N+1)//2. An asymmetry is introduced though by extentding
# the time dependent region of the potential (wall) from -0.5*L to +0.55*L around the center of hour domain. This assymetry results in restricting the WF mainly to the larger part of the domain.
# -

V = np.zeros(N+1)
iterations = 1000
psi_tracker, Vnew, eigen_energies, actual_energies, actual_error = crank_nicolson3(np.sin(np.pi*x), V, N, dx, iterations, 1)

# +
# Now we plot the absolute value of the WF in different time-slots (iterations) over position:
# -

plt.plot(x, np.absolute(psi_tracker[:,0]), 'r-', label = '|ψ|')
plt.plot(x, 0.002*Vnew[:,0], 'b-', label = 'V')
plt.xlabel('x')
plt.ylabel('|ψ|')
plt.title('Splitting well ground state at t=0 (b=1)')
plt.legend()
plt.show()

plt.plot(x, np.absolute(psi_tracker[:,20]), 'r-', label = '|ψ|')
plt.plot(x, 0.002*Vnew[:,20], 'b-', label = 'V')
plt.xlabel('x')
plt.ylabel('|ψ|')
plt.title('Splitting well ground state at t=20 (b=1)')
plt.legend()
plt.show()

plt.plot(x, (np.absolute(psi_tracker[:,20]))**2, 'r-', label = '|ψ|^2')
plt.plot(x, 0.002*Vnew[:,20], 'b-', label = 'V')
plt.xlabel('x')
plt.ylabel('|ψ|^2')
plt.title('Splitting well ground state at t=20 (b=1)')
plt.legend()
plt.show()

plt.plot(x, np.absolute(psi_tracker[:,100]), 'r-', label = '|ψ|')
plt.plot(x, 0.002*Vnew[:,100], 'b-', label = 'V')
plt.xlabel('x')
plt.ylabel('|ψ|')
plt.title('Splitting well ground state at t=100 (b=1)')
plt.legend()
plt.show()

plt.plot(x, (np.absolute(psi_tracker[:,100]))**2, 'r-', label = '|ψ|^2')
plt.plot(x, 0.002*Vnew[:,100], 'b-', label = 'V')
plt.xlabel('x')
plt.ylabel('|ψ|^2')
plt.title('Splitting well ground state at t=100 (b=1)')
plt.legend()
plt.show()

plt.plot(x, np.absolute(psi_tracker[:,200]), 'r-', label = '|ψ|')
plt.plot(x, 0.002*Vnew[:,200], 'b-', label = 'V')
plt.xlabel('x')
plt.ylabel('|ψ|')
plt.title('Splitting well ground state at t=200 (b=1)')
plt.legend()
plt.show()

plt.plot(x, (np.absolute(psi_tracker[:,200]))**2, 'r-', label = '|ψ|^2')
plt.plot(x, 0.002*Vnew[:,200], 'b-', label = 'V')
plt.xlabel('x')
plt.ylabel('|ψ|^2')
plt.title('Splitting well ground state at t=200 (b=1)')
plt.legend()
plt.show()

plt.plot(x, np.absolute(psi_tracker[:,300]), 'r-', label = '|ψ|')
plt.plot(x, 0.002*Vnew[:,300], 'b-', label = 'V')
plt.xlabel('x')
plt.ylabel('|ψ|')
plt.title('Splitting well ground state at t=300 (b=1)')
plt.legend()
plt.show()

plt.plot(x, (np.absolute(psi_tracker[:,300]))**2, 'r-', label = '|ψ|^2')
plt.plot(x, 0.002*Vnew[:,300], 'b-', label = 'V')
plt.xlabel('x')
plt.ylabel('|ψ|^2')
plt.title('Splitting well ground state at t=300 (b=1)')
plt.legend()
plt.show()

plt.plot(x, (np.absolute(psi_tracker[:,500]))**2, 'r-', label = '|ψ|^2')
plt.plot(x, 0.002*Vnew[:,500], 'b-', label = 'V')
plt.xlabel('x')
plt.ylabel('|ψ|^2')
plt.title('Splitting well ground state at t=500 (b=1)')
plt.legend()
plt.show()

plt.plot(x, np.absolute(psi_tracker[:,500]), 'r-', label = '|ψ|')
plt.plot(x, 0.002*Vnew[:,500], 'b-', label = 'V')
plt.xlabel('x')
plt.ylabel('|ψ|')
plt.title('Splitting well ground state at t=500 (b=1)')
plt.legend()
plt.show()

plt.plot(x, (np.absolute(psi_tracker[:,700]))**2, 'r-', label = '|ψ|^2')
plt.plot(x, 0.002*Vnew[:,700], 'b-', label = 'V')
plt.xlabel('x')
plt.ylabel('|ψ|^2')
plt.title('Splitting well ground state at t=700 (b=1)')
plt.legend()
plt.show()

plt.plot(x, np.absolute(psi_tracker[:,700]), 'r-', label = '|ψ|')
plt.plot(x, 0.002*Vnew[:,700], 'b-', label = 'V')
plt.xlabel('x')
plt.ylabel('|ψ|')
plt.title('Splitting well ground state at t=700 (b=1)')
plt.legend()
plt.show()

plt.plot(x, (np.absolute(psi_tracker[:,1000]))**2, 'r-', label = '|ψ|^2')
plt.plot(x, 0.002*Vnew[:,1000], 'b-', label = 'V')
plt.xlabel('x')
plt.ylabel('|ψ|^2')
plt.title('Splitting well ground state at t=1000 (b=1)')
plt.legend()
plt.show()

plt.plot(x, np.absolute(psi_tracker[:,1000]), 'r-', label = '|ψ|')
plt.plot(x, 0.002*Vnew[:,1000], 'b-', label = 'V')
plt.xlabel('x')
plt.ylabel('|ψ|')
plt.title('Splitting well ground state at t=1000 (b=1)')
plt.legend()
plt.show()

# +
# Now that we have finished with the qualitative analysis we can proceed to a more quantitative one.
# We would like to compare the energy evolution in the adiabatic and the non-adiabatic case. To do so we first compare the computed energies of the time-dependent wavefunction at each given time t
# with the corresponding stationary eigenergies of the Hamiltonian at time t (t treated just as a parameter). We expect that in the adiabatic case the actual energies will follow the respective
# eigenenergies of the Hamiltonian at every step. In the non-adiabatic case though we do not expect such a behaviour.

# +
t = np.linspace(0, iterations, iterations+1)

plt.plot(t, eigen_energies, 'r-', label = 'Eigenergies')
plt.plot(t, actual_energies, 'b-', label = 'Computed energies')
plt.xlabel('t')
plt.ylabel('Energy')
plt.title('Eigenenergies vs computed energies')
plt.legend()
plt.show()
# -

print(eigen_energies[1000])
print(actual_energies[1000])

# +
# Now we try to find the error that arises from the computation of the actual energies. NOTE that the eigenenergies also have an error due to the discretization of the domain and the process
# that the numpy functions use to find the eigenvalues. We expect this error to be pretty similar to the one we obtain for the computation of the actual energies:

# +
t = np.linspace(0, iterations, iterations+1)

plt.plot(t, actual_error, 'r-', label = 'Error dE')
plt.xlabel('t')
plt.ylabel('dE')
plt.title('Computational error of the actual energies per time')
plt.legend()
plt.show()
# -

print(max(actual_error))


