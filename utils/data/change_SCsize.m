clc; clear all; close all;
%
load('+data/SC_00k_00k.mat');
load('+data/SC_grid.mat');
%
nx = size(X,2);
ny = size(X,1);
nt = length(t);
%
nx_new = 384;
ny_new = 192;
%
xq = linspace(min(X(1,:)),max(X(1,:)),nx_new);
yq = linspace(min(Y(:,1)),max(Y(:,1)),ny_new);
[Xq,Yq] = meshgrid(xq,yq);
%
%%
%
u = reshape(u,[ny nx nt]);
v = reshape(v,[ny nx nt]);
w = reshape(w,[ny nx nt]);
%
%%
%
for i = 1:nt
    %
    uq(:,:,i) = interp2(X,Y,u(:,:,i),Xq,Yq);
    vq(:,:,i) = interp2(X,Y,v(:,:,i),Xq,Yq);
    wq(:,:,i) = interp2(X,Y,w(:,:,i),Xq,Yq);
    %
end
%
%%
%
X = Xq;
Y = Yq;
%
u = reshape(uq,[nx_new*ny_new nt]);
v = reshape(vq,[nx_new*ny_new nt]);
w = reshape(wq,[nx_new*ny_new nt]);
%
%%
%
save('+data\SC_00k_00k_AE.mat','u','v','w','t');
save('+data\SC_grid_AE.mat','X','Y');