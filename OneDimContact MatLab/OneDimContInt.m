% ----------------------------------------------------------------------- %
% -------------------- 1D Gross-Pitasevskii Equation -------------------- %
% -------------------- Ground State Density Solution -------------------- %
% ----------------------------------------------------------------------- %

% This code calculates the ground state density of quasi-one-dimensional
% condensates. These condenstates are constructed by a trapping potential
% of the form U=(m/2)w^2(x^2+y^2)+(m/2)W^2z^2 where w>>W. This code
% incorporates contact interactions. This code assumes a variational width
% and hence solves the one-dimensional quasi-Gross-Pitaevskii equation with
% a variational wavefunction, f, and width, sigma.

% ----------------------------------------------------------------------- %
% ------------------------- Physical Parameters ------------------------- %
% ----------------------------------------------------------------------- %

% Input physical parameters.

N = 10^(4);             % number of bosons
a = 10^(-3);            % dimensionless s-wave scattering length, a=a_a/l_z
w = 10;                 % trap anisotropy (omega_perp/omega_z)
edd = 0.0;              % relative dipole-dipole interaction strength
QF = 0;                 % quantum fluctuations, 1=on, 0=off
U0 = 2*N*a;             % scattering coupling constant

% ----------------------------------------------------------------------- %
% --------------------------- Evaluation Setup -------------------------- %
% ----------------------------------------------------------------------- %

Nz = 256;            % grid size, of form 2^n for increased fft efficiency
dz = 0.05;            % finite difference size
enorm = 1.0;          % energy normalisation

for j = 1:Nz
    z(j) = -((Nz/2.0)-(j-1))*dz+0.5*dz;            % spatial z coordinate
end 

% ----------------------------------------------------------------------- %
% ---------------------------- Physics Setup ---------------------------- %
% ----------------------------------------------------------------------- %

for j = 1:Nz
    TrapPotential(j) = OneDimHarmonicTrap(z(j));     % trapping potential
            Sigma(j) = 1/sqrt(w);                    % initial width
              Psi(j) = 1/sqrt(pi)*exp(-z(j)^2);      % initial wavefunction           
end

% ----------------------------------------------------------------------- %
% ----------------------- Imaginary Time Algorithm ---------------------- %
% ----------------------------------------------------------------------- %

delt = -1i*0.0001;                 % imaginary time step
error = 1e-9;                      % error threshold for energy convergence
max_iteration_t = 100000000;       % maximum number of iterations
it_flag = 0;                       % initialise condition
it = 1;                            % initial iteration counter
diff = 1/(2*dz*dz);                % gradient approximation

while it_flag == 0

    k1 = zeros(Nz);
    k2 = zeros(Nz);
    k3 = zeros(Nz);
    k4 = zeros(Nz);                  % initialising k arrays for RK4 method

    % ---------------------------- Making k1 ---------------------------- %

    PsiSave = Psi;

    V=1i*delt*OneDimPotential(TrapPotential, Psi, Sigma, w,U0); 
                                                                % potential
    k1 = V.*Psi-2*delt*1i*diff*Psi;           % general k1
    k1(1) = k1(1)+delt*1i*diff*Psi(2);        % smoothing boundary problems
    k1(Nz) = k1(Nz)+delt*1i*diff*Psi(Nz-1);   % smoothing boundary problems

    for j = 2:Nz-1
        k1(j) = k1(j)+delt*1i*diff*Psi(j-1)+delt*1i*diff*Psi(j+1);
    end              % adding kinetic energy using finite difference method

    Psi = PsiSave + k1/2.0;
    renorm = sum(Psi.*conj(Psi))*dz;
    Psi = Psi*sqrt(1/renorm);                    % normalising wavefunction

    % ---------------------------- Making k2 ---------------------------- %

    V=1i*delt*OneDimPotential(TrapPotential, Psi, Sigma, w,U0); 
                                                                % potential
    k2 = V.*Psi-2*delt*1i*diff*Psi;           % general k1
    k2(1) = k2(1)+delt*1i*diff*Psi(2);        % smoothing boundary problems
    k2(Nz) = k2(Nz)+delt*1i*diff*Psi(Nz-1);   % smoothing boundary problems

    for j = 2:Nz-1
        k2(j) = k2(j)+delt*1i*diff*Psi(j-1)+delt*1i*diff*Psi(j+1);
    end              % adding kinetic energy using finite difference method

    Psi = PsiSave + k2/2.0;
    renorm = sum(Psi.*conj(Psi))*dz;
    Psi = Psi*sqrt(1/renorm);                    % normalising wavefunction

    % ---------------------------- Making k3 ---------------------------- %

    V=1i*delt*OneDimPotential(TrapPotential, Psi, Sigma, w,U0); 
                                                                % potential
    k3 = V.*Psi-2*delt*1i*diff*Psi;           % general k1
    k3(1) = k3(1)+delt*1i*diff*Psi(2);        % smoothing boundary problems
    k3(Nz) = k3(Nz)+delt*1i*diff*Psi(Nz-1);   % smoothing boundary problems

    for j = 2:Nz-1
        k3(j) = k3(j)+delt*1i*diff*Psi(j-1)+delt*1i*diff*Psi(j+1);
    end              % adding kinetic energy using finite difference method

    Psi = PsiSave + k3;
    renorm = sum(Psi.*conj(Psi))*dz;
    Psi = Psi*sqrt(1/renorm);                    % normalising wavefunction

    % ---------------------------- Making k4 ---------------------------- %

    V=1i*delt*OneDimPotential(TrapPotential, Psi, Sigma, w,U0); 
                                                                % potential
    k4 = V.*Psi-2*delt*1i*diff*Psi;           % general k1
    k4(1) = k4(1)+delt*1i*diff*Psi(2);        % smoothing boundary problems
    k4(Nz) = k4(Nz)+delt*1i*diff*Psi(Nz-1);   % smoothing boundary problems

    for j = 2:Nz-1
        k4(j) = k4(j)+delt*1i*diff*Psi(j-1)+delt*1i*diff*Psi(j+1);
    end              % adding kinetic energy using finite difference method

    Psi = PsiSave + k1/6.0+k2/3.0+k3/3.0+k4/6.0;
    renorm = sum(Psi.*conj(Psi))*dz;
    Psi = Psi*sqrt(1/renorm);                    % normalising wavefunction

    % -------------------------- Finding Sigma -------------------------- %

    T0 = 1+U0*Psi.*conj(Psi);
    T4 = -w^2;                      % terms in polynomial determining sigma

    for j = 1:Nz
           r = roots([T4 0 0 0 T0(j)]);    % solving polynomial
           s = (r(real(r)>0&imag(r)==0));        % taking positive solution

           if isempty(s) == 1     % if sol. exist, revert to previous value
                S(j) = Sigma(j);
           else 
                S(j) = s;
           end
    end
    Sigma = S;                  % setting sigma equal to array of new sigma

    % ---------------------- Calculating the Energy --------------------- %

    V=1i*delt*OneDimPotential(TrapPotential, Psi, Sigma, w,U0); 
                                                                % potential
    [EK(it),EV(it),EC(it)] = OneDimEnergy(V,Psi,dz,enorm,U0,Sigma,w,TrapPotential);     
                                                                   % energy
    ET(it) = EK(it)+EV(it)+EC(it);     % summing energy terms

     if (it>1)
         converge(it) = abs((ET(it)-ET(it-1))/(ET(it)*delt*1i));
         converge(it)
        if (converge(it) < error);            % ceasing code when converged
            it_flag = 1;
            fprintf('Imaginary-time algorithm converged.');
        end
        if (it == max_iteration_t)
            it_flag = 1;
            fprintf('Imaginary-time algorithm not convered.');
        end
     end 
    it = it +1;
end

% 642.606

