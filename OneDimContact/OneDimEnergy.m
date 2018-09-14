function [EK,EV,EC] = OneDimEnergy(V,Psi,dz,enorm,U0,Sigma,w,TrapPotential)

% This function calculates the energy density of the condensate

Sig2 = Sigma.*Sigma;
Density = Psi.*conj(Psi); 
[Psi_X] = gradient(Psi); 

KE = (1/2.0)*(Psi_X.*conj(Psi_X))/(dz*dz);                    %kinetic                 
VE = TrapPotential.*Density+(1/2.0)./Sig2+(1/2.0)*w^2*Sig2;   % trap and mf
CE = (U0/2.0)*Density.*Density;                               % contact

EK = sum(KE)*dz/enorm;          % these sums mimic intergral over all space                    
EV = sum(VE)*dz/enorm;
EC = sum(CE)*dz/enorm;

end

