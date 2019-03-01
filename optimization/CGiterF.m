%% Conjugate Gradient Method
%% Muhammad Izzatullah 
%% 15 August 2018


function [xk,hist,xfull,gfull,Hk] = CGiterF(fh,x0,model,D,tol,maxit)
% CG iteration to solve min_x f(x)
%
% input:
%   fh - function handle that returns value and gradient: [f,g] = fh(x)
%   x0 - initial iterate
%   model - model parameters
%   D - Observation data
%   tol   - stop when ||g||_2 <= tol
%   maxit - stop when iter > maxit
%
% output:
%   x    - final iterate
%   hist - array with rows [iter, f, g]

k       = 0;
%xfull = zeros(length(x0),maxit+1);
%gfull = zeros(length(x0),maxit+1);
xfull = [];
gfull = [];

xk      = x0;
[fk,gk] = fh(xk);
dk = -gk;

xfull(:,1) = xk;
gfull(:,1) = gk;

hist = [k,fk,norm(gk)];

fprintf(1,' k , fk          , ||gk||_2\n');
fprintf(1,'%3d, %1.5e , %1.5e \n',hist);

while (fk > tol)&&(k < maxit)
   
   g = gk;
   
   % compute alpha
   a = alphaCG(xk,model,dk,D);
   
   % update
   xk = xk + a*dk;
   
   [ft,gt] = fh(xk);
   
   fk = ft;
   gk = gt;
   
   beta1 = dot(gk,gk - g)/dot(dk,gk - g);
   beta2 = dot(gk,gk)/dot(dk,gk - g);
   beta = max(0,min(beta1,beta2));
   
   dk = -gk + beta*dk;

   k = k + 1;
   xfull(:,k + 1) = xk;
   gfull(:,k + 1) = gk;
   
   hist(k,:) = [k,fk,norm(gk)];
   fprintf(1,'%3d, %1.5e , %1.5e \n',k,fk,norm(gk));
end
[~,~,Hk] = fh(xk);
end