import numpy as np

def OneDimEnergy(V, Psi, dz, enorm, U0, Sigma, w, TrapPotential):
    # Calculates the energy densityof the condensates

    Sig2 = Sigma*Sigma
    Density = Psi*np.conj(Psi)
    Psi_X = np.gradient(Psi)

    KE = 0.5*(Psi_X*np.conj(Psi_X))/(dz*dz)
    VE = TrapPotential*Density+(0.5/Sig2)+(0.5*w**2*Sig2)
    CE = (U0/2)*Density*Density

    EK = np.sum(KE)*dz/enorm
    EV = np.sum(VE)*dz/enorm
    EC = np.sum(CE)*dz/enorm

    return EK, EV, EC
