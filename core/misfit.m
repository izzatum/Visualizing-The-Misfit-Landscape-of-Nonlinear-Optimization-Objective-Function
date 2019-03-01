function [f,g,H] = misfit(m,D,alpha,L,model)
% Evaluate least-squares misfit
%
%   0.5||P^TA^{-1}(m)Q - D||_{F}^2 + 0.5\alpha||Lm||_2^2,
%
% where P, Q encode the receiver and source locations and L is the first-order FD matrix
%
% use:
%   [f,g,H] = misfit(m,D,model)
%
% input:
%   m - squared-slownes [s^2/km^2]
%   D - single-frequency data matrix
%   alpha - regularization parameter
%   model.h - gridspacing in each direction d = [d1, d2];
%   model.n - number of gridpoints in each direction n = [n1, n2]
%   model.f - frequency [Hz].
%   model.{zr,xr} - {z,x} locations of receivers [m] (must coincide with gridpoints)
%   model.{zs,xs} - {z,x} locations of sources [m] (must coincide with gridpoints)
%
%
% output:
%   f - value of misfit
%   g - gradient (vector of size size(m))
%   H - GN Hessian (Spot operator)


%% get matrices
m = m(:);

%% forward solve
[Dp,Jp] = F(m,model);

%% compute f
f = .5*norm(Dp - D)^2 + .5*alpha*norm(L*m)^2;

%% compute g
g = Jp'*(Dp - D) + alpha*(L'*L)*m;

%% get H
H = Jp'*Jp + alpha*(L'*L);

end
