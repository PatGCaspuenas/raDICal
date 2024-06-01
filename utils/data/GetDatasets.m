clc; clear all; close all;
% 
% Datasets information
%
Nimg = 1:2400
RootGrid = 'F:\Re150\Indipendent_3\Validation\Grid.dat';
RootOut = 'FPc_00k_03k.mat';
RootFields = 'F:\Re150\Indipendent_3\Validation\Flow.';
%
% Define the uniform grid
%
x = -5:0.1:15;
y = -5:0.1:5;
[X,Y] = meshgrid(x,y);

%
InterpolateDatasets(Nimg,X,Y,RootFields,RootGrid,RootOut,'NO')
%
