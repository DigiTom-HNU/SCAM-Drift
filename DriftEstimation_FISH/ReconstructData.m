clc;clear;close all;
restoredefaultpath;
addpath(genpath(pwd));
frameNum    = 4000;
colrow{1} = 3:6;
colrow{2} = 2001:6000;
pixelsize = [163.8,175.1];
dimensional = '3D';
r = 4; 

path = '470';
load(fullfile(path,'imputation\Drift_diff.mat'));
load(fullfile(path,'PreCorrectData.mat'));
locTable = [];
locTable_raw = [];
for i = 1:length(PreCorrectData)%4列
    PreCorrectDataT = PreCorrectData{i};
    Drift_diffT = Drift_diff(i,:,:);
    Drift_diffT = reshape(Drift_diffT,[size(Drift_diffT,2),size(Drift_diffT,3)]);
    [locTable_noC,locTable_C] = reconstruct(PreCorrectDataT,Drift_diffT,dimensional,1);
    locTable_C = sortrows(locTable_C(:,2:end),1);
    locTable_C_col = [locTable_C,ones(size(locTable_C,1),1)*i];
    locTable_noC = sortrows(locTable_noC(:,2:end),1);
    locTable_noC_col = [locTable_noC,ones(size(locTable_noC,1),1)*i];
    locTable_raw = [locTable_raw;locTable_noC_col];
    locTable = [locTable;locTable_C_col];
end
save(fullfile(path,'locTable_C.mat'),'locTable');

