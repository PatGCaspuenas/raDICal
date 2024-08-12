clc; clear all; close all;
% 
% Datasets information
%
Nimg = sort(randperm(20000,10000))
RootGrid = 'F:\Re150\Indipendent_3\Variable_total\Grid.dat';
RootOut = 'FP_00k_20k.mat';
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
clc; clear all; close all;
% 
% Datasets information
%
Nimg = 1:10000
RootGrid = 'F:\Re150\Indipendent_3\Variable_total\Grid.dat';
RootOut = 'FP_00k_10k.mat';
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
