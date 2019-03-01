%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Numerical Example for 
%%  Visualizing The Misfit Landscape  - An Adaptation from Machine Learning
%%  (SEG 2019 Expanded Abstract)
%%  Muhammad Izzatullah, King Abdullah University of Sciecne and Technology (KAUST)
%%  01/03/2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% setup

% read model, dx = 20 or 50
dx = 50;
v  = dlmread(['marm_' num2str(dx) '.dat']);

% initial model
%v0 = @(zz,xx)v(1)+.7e-3*max(zz-350,0);          % Varying by depth initial
v0 = imgaussfilt(v,5);                           % Gaussian filtered initial

% set frequency, do not set larger than min(1e3*v(:))/(7.5*dx) or smaller
% than 0.5

f  = [0.5,1.5,2.5,3.5];

% receivers, xr = .1 - 10km, with 2*dx spacing, zr = 2*dx
xr = 100:2*dx:10000;
zr = 2*dx*ones(1,length(xr));

% sources, xr = .1 - 10km, with 4*dx spacing, zr = 2*dx
xs = 100:4*dx:10000;
zs = 2*dx*ones(1,length(xs));

%% observed data
% grid
n  = size(v);
h  = dx*[1 1];
z  = (0:n(1)-1)*h(1);
x  = (0:n(2)-1)*h(2);
[zz,xx] = ndgrid(z,x);

% parameters
model.f = f;
model.n = n;
model.h = h;
model.zr = zr;
model.xr = xr;
model.zs = zs;
model.xs = xs;

% Regularization parameters
alpha = 0.1;
L = getL(model.h,model.n); % first-order FD matrix


% Additive Gaussian noise in data
sigma = 0;

% model (slowness-squared)
m = 1./v(:).^2;

% initial model (slowness-squared)
%m0 = vec(1./v0(zz,xx).^2);
m0 = vec(1./v0.^2);

% data
D = F(m,model); % + sigma*randn(size(D));
 


%% inversion

% misfit
fh = @(m)misfit(m,D,alpha,L,model);

% Simple CG iteration
tic;
[mk,hist,mfull,Gf,H] = CGiterF(fh,m0,model,D,1e-4,100);
toc;

% Reconstructed velocity model
vk = reshape(real(1./sqrt(mk)),n);

%% Construct Gradient Matrix and Error Matrix for active subspaces 

% Error Matrix
E = mfull - mk;
E = E(:,1:end-1); 

% SVD of Error Matrix
[Ue, Se, Ve] = svds(E,10);


% Gradient Matrix and its SVD
C = Gf*Gf'; % C matrix = GG^{T}

% SVD of C matrix
[Uc, Sc, ~] = svd(C,'econ');

Scd = diag(Sc); % Eigenvalues of C

% SVD of Gradient Matrix
tic;
[Ug, Sg, ~] = svd(Gf,'econ');
toc;

S2gd = diag(Sg).^2; % Eigenvalues of C = GG^{T}


%% Visualization sampling
% Optimizer trajectories and contour along the trajectories
% Sampling range based on first 2 PCA directions of Error Matrix
alpha = linspace(-2.5,2.5,10);

fviz = zeros(10);

ue1 = Ue(:,1);
ue2 = Ue(:,2);

parfor i = 1:10
    for j = 1:10
        fviz(i,j) = fh(mk + alpha(i)*ue1 + alpha(j)*ue2);
    end
end

% Interpolation for refining the contour
[alpha1,beta1] = meshgrid(-2.5:0.2:2.5,-2.5:0.2:2.5);
fviz_int = interp2(alpha,alpha,fviz,alpha1,beta1,'spline');

% Global contour based on first 2 PCA directions of Gradient Matrix
% Sampling range based on triple first 2 PCA directions of Error Matrix
alpha = linspace(-10.0,10.0,30);

gviz = zeros(30);

% Using 2 PCA directions of Gradient Matrix
ug1 = Ug(:,1);
ug2 = Ug(:,2);

parfor i = 1:30
    for j = 1:30
        gviz(i,j) = fh(mk + alpha(i)*ug1 + alpha(j)*ug2);
    end
end

% Interpolation for refining the contour
[alpha2,beta2] = meshgrid(-10.0:0.2:10.0,-10.0:0.2:10.0);
gviz_int = interp2(alpha,alpha,gviz,alpha2,beta2,'spline');


save('res_marm_viz.mat');

%% Plotting

% Figure #1
figure;
ax1 = subplot(3,1,1);
imagesc(ax1,x,z,v,[min(v(:)) max(v(:))]);title(ax1,'True Velocity Model','FontSize', 16);axis equal tight;
colorbar; colormap jet; xlabel(ax1,'Distance [m]','FontSize', 16); ylabel(ax1,'Depth [m]','FontSize', 16);

ax2 = subplot(3,1,2);
imagesc(ax2,x,z,v0,[min(v(:)) max(v(:))]);title(ax2,'Initial Velocity Model','FontSize', 16);axis equal tight;
colorbar; colormap jet; xlabel(ax2,'Distance [m]','FontSize', 16); ylabel(ax2,'Depth [m]','FontSize', 16);

ax3 = subplot(3,1,3);
imagesc(ax3,x,z,vk,[min(v(:)) max(v(:))]);title(ax3,'Reconstructed Velocity Model','FontSize', 16);axis equal tight;
colorbar; colormap jet; xlabel(ax3,'Distance [m]','FontSize', 16); ylabel(ax3,'Depth [m]','FontSize', 16);

% Figure #
figure;
semilogy(hist(:,1),hist(:,2)/hist(1,2),'b-',hist(:,1),hist(:,3)/hist(1,3),'r-.');
title('Convergence History','FontSize', 16);
legend({'f','|g|'},'FontSize',16);
xlabel('Number of iterations','FontSize', 16);

% Figure #2
figure;
semilogy(1:length(S2gd),S2gd,'ro-',1:length(S2gd),Scd(1:length(S2gd)),'b*-.');
title('Eigenvalues of C','FontSize', 16); 
xlabel('Number of eigenvalues','FontSize', 16);
legend({'\lambda(GG^{T})', '\lambda(C)'},'FontSize', 16);

% Figure #3
figure;
ax1 = subplot(2,1,1);
contourf(ax1,alpha1,beta1,fviz_int,'ShowText','on');
hold on; 
plot(ax1,Ue(:,1)'*E,Ue(:,2)'*E,'r*-','LineWidth',3);
xlabel(ax1,'\alpha','FontSize', 16);
ylabel(ax1,'\beta','FontSize', 16); 
colorbar; colormap jet;

ax2 = subplot(2,1,2);
surf(ax2,alpha1,beta1,fviz_int,'FaceColor','interp','EdgeColor' ,'interp');
xlabel(ax2,'\alpha','FontSize', 16);
ylabel(ax2,'\beta','FontSize', 16); 
colorbar; colormap jet;

% Figure #4
figure;
ax1 = subplot(2,1,1);
contourf(ax1,alpha2,beta2,gviz_int,'ShowText','on');
xlabel(ax1,'\alpha','FontSize', 16);
ylabel(ax1,'\beta','FontSize', 16); 
colorbar; colormap jet;

ax2 = subplot(2,1,2);
surf(ax2,alpha2,beta2,gviz_int,'FaceColor','interp','EdgeColor' ,'interp');
xlabel(ax2,'\alpha','FontSize', 16);
ylabel(ax2,'\beta','FontSize', 16); 
colorbar; colormap jet;

