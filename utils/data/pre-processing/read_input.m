clc; clear all; close all;
%
data = load('F:\Re150\Indipendent_3\Validation\InputValues.txt');

data = [0,0,0;data];


t = 1:2400
%
vF = data(t,1)
vT = data(t,2) 
vB = data(t,3)
%
save('FPc_00k_03k_u.mat','vF','vT','vB');
