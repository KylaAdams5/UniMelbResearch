function PotentialArray = OneDimPotential(TrapPotential, Psi, Sigma, w,U0)

% This function returns the quasi-one-dimensional potential

Density = Psi.*conj(Psi);
Sig2 = Sigma.*Sigma;

PotentialArray = -(TrapPotential+U0*Density./Sig2+(1/2.0)./Sig2+w^2*Sig2/2);

end

