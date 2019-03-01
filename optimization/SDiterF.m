function [xk,hist,xfull,gfull,Hk] = SDiterF(fh,x0,tol,maxit,alpha)
% Steepest descent iteration to solve min_x f(x)
%
% input:
%   fh - function handle that returns value and gradient: [f,g] = fh(x)
%   x0 - initial iterate
%   tol   - stop when ||g||_2 <= tol
%   maxit - stop when iter > maxit
%   alpha - stepsize
%
% output:
%   x    - final iterate
%   hist - array with rows [iter, f, g]

k       = 0;
xfull = [];
gfull = [];

xk      = x0;
[fk,gk] = fh(xk);

xfull(:,1) = xk;
gfull(:,1) = gk;

hist = [k,fk,norm(gk)];

fprintf(1,' k , fk          , ||gk||_2\n');
fprintf(1,'%3d, %1.5e , %1.5e \n',hist);

while (norm(gk) > tol)&&(k < maxit)
   % update
   sk = -alpha*gk;
   xk = xk + sk;
   
   % gradient evaluation
   [fk,gk,Hk] = fh(xk);
   
   k = k + 1;
   xfull(:,k + 1) = xk;
   gfull(:,k + 1) = gk;
   
   hist(k,:) = [k,fk,norm(gk)];
   fprintf(1,'%3d, %1.5e , %1.5e \n',k,fk,norm(gk));
end