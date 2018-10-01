import numpy as np

def OneDimPotential(TrapPotential, Psi, Sigma, w, U0):
    Density = Psi*np.conj(Psi)
    Sig2 = Sigma*Sigma

    PotentialArray = -(TrapPotential+U0*Density/(Sig2+0.5)/Sig2+w**2*Sig2/2)
    return PotentialArray
