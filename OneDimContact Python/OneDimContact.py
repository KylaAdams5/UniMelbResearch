###############################
### 1D Contact Interactions ###
###############################

# Following Mitches Matlab code
# redoing here so that I understand the code better/potentially make it quicker


# % This code calculates the ground state density of quasi-one-dimensional
# % condensates. These condenstates are constructed by a trapping potential
# % of the form U=(m/2)w^2(x^2+y^2)+(m/2)W^2z^2 where w>>W. This code
# % incorporates contact interactions. This code assumes a variational width
# % and hence solves the one-dimensional quasi-Gross-Pitaevskii equation with
# % a variational wavefunction, f, and width, sigma

##############################################################
import numpy as np
import time
t0 = time.time()
from OneDimPotential import OneDimPotential
from OneDimHarmonicTrap import OneDimHarmonicTrap
from OneDimEnergy import OneDimEnergy

# Initial physical parameters

N = 1e4  # number of bosons
a = 1e-3 #dimensionless s-wave scattering length, a=a_a/l_z
w = 10 # trap ansiotropu (omega_Perp/omega_z)
edd = 0   # relative dipole-dipole interaction strength
QF = 0;                 # quantum fluctuations, 1=on, 0=off
U0 = 2*N*a;             # scattering coupling constant

##############################################################

# Evaluation setup

Nz = 255;            # grid size, of form 2^n for increased fft efficiency
dz = 0.05;            # finite difference size
enorm = 1.0;          # energy normalisation
z = np.zeros(Nz)


for j in range(Nz):
    z[j] = -((Nz/2) - (j-1))*dz+0.5*dz # spatial z coord

##############################################################
# Physics setup
TrapPotential = np.zeros(Nz)
Sigma = np.zeros(Nz)
Psi = np.zeros(Nz)
for j in range(Nz):
    TrapPotential[j] = OneDimHarmonicTrap(z[j])
    Sigma[j] = 1/np.sqrt(w)
    Psi[j] = 1/np.sqrt(np.pi)*np.exp(-z[j]**2)
##############################################################
# Imaginary time algorithm

delt = -1j * 1e-4 #imaginary time step
error = 1e-9 # error threshold for energy convergence
max_iteration_t = 1e8 # max number of interactions
it_flag = 0 # initialise condition
it = 0 #initial iteration counter
diff = 1/(2*dz*dz) # gradient approx

EK = np.zeros(1)
EV = np.zeros(1)
EC = np.zeros(1)
ET = np.zeros(1)
converge = np.zeros(1)
# print(EK)

while it_flag == 0:
    k1 = np.zeros(Nz)
    k2 = np.zeros(Nz)
    k3 = np.zeros(Nz)
    k4 = np.zeros(Nz)


#using the RK4 method

    #find K1
    PsiSave = Psi
    V = 1j*delt*OneDimPotential(TrapPotential, Psi,Sigma, w, U0) # referring to potential funciton

    k1 = V*Psi-2*delt*1j*diff*Psi
    k1[0] = k1[0]+delt*1j*diff*Psi[1] # smoothing boundary problems
    k1[Nz-1] = k1[Nz-1]+delt*1j*diff*Psi[Nz-1] #smoothing boundary problems

    for j in range(2, Nz-2):
        k1[j] = k1[j]+(delt*1j*diff*Psi[j-1])+(delt*1j*diff*Psi[j+1])
    # adding kinetic energy using finite difference method

    Psi = PsiSave + k1/2
    renorm = sum(Psi*np.conj(Psi)*dz)
    Psi = Psi*np.sqrt(1/renorm)  # normalising the wave function

    #find K2

    V = 1j*delt*OneDimPotential(TrapPotential, Psi,Sigma, w, U0) # referring to potential funciton

    k2 = V*Psi-2*delt*1j*diff*Psi
    k2[0] = k1[0]+delt*1j*diff*Psi[1] # smoothing boundary problems
    k2[Nz-1] = k2[Nz-1]+delt*1j*diff*Psi[Nz-1] #smoothing boundary problems

    for j in range(2, Nz-2):
        k2[j] = k2[j]+(delt*1j*diff*Psi[j-1])+(delt*1j*diff*Psi[j+1])
    # adding kinetic energy using finite difference method

    Psi = PsiSave + k2/2
    renorm = sum(Psi*np.conj(Psi)*dz)
    Psi = Psi*np.sqrt(1/renorm)  # normalising the wave function

    #find K3

    V = 1j*delt*OneDimPotential(TrapPotential, Psi,Sigma, w, U0) # referring to potential funciton

    k3 = V*Psi-2*delt*1j*diff*Psi
    k3[0] = k3[0]+delt*1j*diff*Psi[1] # smoothing boundary problems
    k3[Nz-1] = k3[Nz-1]+delt*1j*diff*Psi[Nz-1] #smoothing boundary problems

    for j in range(2, Nz-2):
        k3[j] = k3[j]+(delt*1j*diff*Psi[j-1])+(delt*1j*diff*Psi[j+1])
    # adding kinetic energy using finite difference method

    Psi = PsiSave + k3
    renorm = sum(Psi*np.conj(Psi)*dz)
    Psi = Psi*np.sqrt(1/renorm)  # normalising the wave function

    #find K4

    V = 1j*delt*OneDimPotential(TrapPotential, Psi,Sigma, w, U0) # referring to potential funciton

    k4 = V*Psi-2*delt*1j*diff*Psi
    k4[0] = k4[0]+delt*1j*diff*Psi[1] # smoothing boundary problems
    k4[Nz-1] = k4[Nz-1]+delt*1j*diff*Psi[Nz-1] #smoothing boundary problems

    for j in range(2, Nz-2):
        k4[j] = k4[j]+(delt*1j*diff*Psi[j-1])+(delt*1j*diff*Psi[j+1])
    # adding kinetic energy using finite difference method

    Psi = PsiSave + (k1/6)+(k2/3)+(k3/3)+(k4/6)
    renorm = sum(Psi*np.conj(Psi)*dz)
    Psi = Psi*np.sqrt(1/renorm)  # normalising the wave function

    ##### Finding Sigma

    T0 = 1+U0*Psi*np.conj(Psi)
    T4 = -w**2
    S = np.zeros(Nz, dtype = complex)
    a =[]
    for j in range(Nz):
        r = np.roots([T4,0,0,0,T0[j]])
        s = r[(np.real(r) > 0) & (np.imag(r)==0)]
        s = np.array(s, dtype = float)

        if s != a:
            S[j] = s
        else:
            S[j] = Sigma[j]


    Sigma = S

    ##### Calculating the energy
    V = 1j*delt*OneDimPotential(TrapPotential, Psi, Sigma, w, U0)


    EK[it], EV[it], EC[it] = OneDimEnergy(V, Psi, dz, enorm, U0, Sigma, w, TrapPotential)
    ET[it] = EK[it] + EV[it] + EC[it]

    EK = np.append(EK, EK[it])
    EV = np.append(EV, EK[it])
    EC = np.append(EC, EK[it])
    ET = np.append(ET, EK[it])
    converge = np.append(converge, converge[it])

    if (it > 0):
        converge[it] = np.abs((ET[it]-ET[it-1])/(ET[it]*delt*1j))
        print("The convergence is", converge[it])
        if (converge[it] < error):
            it_flag = 1
            print("Imaginary time algorithm converged")
        if (it == max_iteration_t):
            it_flag = 1
            print("Imaginary-time algorithm not converged")

    it = it + 1

t1 = time.time()
total = t1-t0
print("The total code run time is:", total)


plt.plot()


# 626.7975707054138
