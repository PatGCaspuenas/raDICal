clc; clear all; close all;
%
load('FPc_60k_70k_nearf.mat');
load('FP_grid_nearf.mat');

u = U;
v = V;
%
nx = size(X,2);
ny = size(X,1);
nt = length(t);
%
nx_new = 96;
ny_new = 48;
%
xq = linspace(min(X(1,:)),max(X(1,:)),nx_new);
yq = linspace(min(Y(:,1)),max(Y(:,1)),ny_new);
[Xq,Yq] = meshgrid(xq,yq);
%
%%
%
u = reshape(u,[ny nx nt]);
v = reshape(v,[ny nx nt]);

%p = reshape(p,[ny nx nt]);

%du = reshape(du,[ny nx nt]);
%dv = reshape(dv,[ny nx nt]);
%
%%
%
for i = 1:nt
    %
    uq(:,:,i) = interp2(X,Y,u(:,:,i),Xq,Yq);
    vq(:,:,i) = interp2(X,Y,v(:,:,i),Xq,Yq);

    %pq(:,:,i) = interp2(X,Y,p(:,:,i),Xq,Yq);

    %duq(:,:,i) = interp2(X,Y,du(:,:,i),Xq,Yq);
    %dvq(:,:,i) = interp2(X,Y,dv(:,:,i),Xq,Yq);
    %
end
%
%%
%
B = interp2(X,Y,B,Xq,Yq);
X = Xq;
Y = Yq;
%
u = reshape(uq,[nx_new*ny_new nt]);
v = reshape(vq,[nx_new*ny_new nt]);

%p = reshape(pq,[nx_new*ny_new nt]);

%du = reshape(duq,[nx_new*ny_new nt]);
%dv = reshape(dvq,[nx_new*ny_new nt]);
%
U = u;
V = v;
%%
%
save('FPc_60k_70k_nearf.mat','U','V','t','Re','vF','vT','vB');
save('FP_grid_nearf2.mat','X','Y','B');