clc;clear;close all;
restoredefaultpath;
addpath(genpath(pwd));
path = '470';
frameNum  = 4000; % frame num
colrow{1} = 3:6; % Range of columns​
colrow{2} = 2001:6000;% Range of frame
pixelsize = [163.8,175.1];
dimensional = '3D';
r = 4; % Clustering threshold​

[Drift_diff,PreCorrectData,noCorrectData] = CorrctAndReconstruct(path,frameNum,pixelsize,dimensional,r,colrow);
Drift_diff = cell_3D(Drift_diff);
save([path,'\Drift_diff.mat'],'Drift_diff'); 
save([path,'\PreCorrectData.mat'],'PreCorrectData'); % ​​Uncorrected and non-deduplicated data​
save([path,'\noCorrectData.mat'],'noCorrectData');% ​​Uncorrected and deduplicated data​





