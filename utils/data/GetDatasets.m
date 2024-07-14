clc; clear all; close all;
% 
% Datasets information
%
Nimg = 10000:13000
RootGrid = 'F:\Re150\Indipendent_3\Variable_total\Grid.dat';
RootOut = 'FPnc_10k_13k.mat';
RootFields = 'F:\Re150\no control\Flow.';
%
% Define the uniform grid
%
x = linspace(-5,15,192);
y = linspace(-5,5,96);
[X,Y] = meshgrid(x,y);

%
InterpolateDatasets(Nimg,X,Y,RootFields,RootGrid,RootOut,'NO')
%
