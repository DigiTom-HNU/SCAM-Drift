clc;clear;close all;
addpath('slanCM');
%% ​​Parameters for generating simulated data​
load('precision_00000.mat');
frameNum = 2000;% frame number
SignalAppearFrameNumTheory = 30; % The number of signal spatiotemporal upsamplings
AvgMolDens = 0.0015;%density localization/μm^2
driftRMSRange = 0.12; % RMS drrift, unit: pixel
framestep = 250/sqrt(2);% Step length of displacement table
pixelsize = 163.8;%unit: nm
FovSize = [1024,1024];
dimensional = '3D';
r = 4; % ​​Clustering threshold​. unit:pixel
R=3000; % radius of cluster​
ClusterPointNum = 16; % ​​Number of points within a cluster​

%% simulate
% [locTable,rawData,UpsamplingData,DriftX_diff_r,DriftY_diff_r,DriftZ_diff_r]  = simulatelLocFish(frameNum,SignalAppearFrameNumTheory,...
%     precision,framestep,pixelsize,FovSize,driftRMSRange,AvgMolDens,R,ClusterPointNum);
%% Figure 3 data
load(fullfile('simulated data\UpsamplingData.mat')); % paper figure 3 data
load(fullfile('simulated data\locTable.mat'))
load(fullfile('simulated data\rawData.mat'));
load(fullfile('simulated data\DriftX_diff_r.mat'));
load(fullfile('simulated data\DriftY_diff_r.mat'));
load(fullfile('simulated data\DriftZ_diff_r.mat'));

locTableT = [locTable(:,1),locTable(:,2:3)/pixelsize,locTable(:,4)/framestep];

rawDataR = reconstructRawdata(rawData,framestep);
figure(1);scatter3(rawDataR(:,2),rawDataR(:,4),rawDataR(:,3),5,rawDataR(:,3),'filled');
colormap(slanCM('viridis'));
h=colorbar('horizontal');  
set(h, 'Color', 'white');
set(gca, 'Color', 'k','fontsize',18, 'FontWeight', 'bold', 'FontName', 'Arial','LineWidth', 2); % 背景设置为黑色
set(gcf, 'Color', 'k');
set(gca, 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
grid off;
axis off; 
axis equal;
box on;
xlim([min(rawDataR(:,2)),max(rawDataR(:,2))]);
ylim([min(rawDataR(:,4)),max(rawDataR(:,4))]);
zlim([min(rawDataR(:,3)),max(rawDataR(:,3))]);
view(2);
rawDataRoi = rawDataR(rawDataR(:,2)>110 & rawDataR(:,2)<140 & rawDataR(:,4)>80 & rawDataR(:,4)<110,:);
figure(2);scatter3(rawDataRoi(:,2),rawDataRoi(:,4),rawDataRoi(:,3),35,rawDataRoi(:,3),'filled');
colormap(slanCM('viridis'));
h=colorbar('horizontal'); 
set(h, 'Color', 'white');
set(gca, 'Color', 'k','fontsize',18, 'FontWeight', 'bold', 'FontName', 'Arial','LineWidth', 2); % 背景设置为黑色
set(gcf, 'Color', 'k');
set(gca, 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
grid off;
axis off; 
axis equal;
box on;
xlim([min(rawDataRoi(:,2)),max(rawDataRoi(:,2))]);
ylim([min(rawDataRoi(:,4)),max(rawDataRoi(:,4))]);
zlim([min(rawDataRoi(:,3)),max(rawDataRoi(:,3))]);
view(2);


%% Spatiotemporal Cluster
locTable_Cluster = SpatiotemporalCluster(locTableT,r,framestep,pixelsize,dimensional);%先聚（融合三维,是为了两个点在轴向正好挨着），贴边去除

%% Drift estimate
tic;
[DriftX_diff,DriftY_diff,DriftZ_diff] = DriftEstimate(locTable_Cluster,frameNum,pixelsize,framestep,dimensional);
toc;

locTable_out_NoC = reconstruct(locTable_Cluster,pixelsize,framestep,[DriftX_diff,DriftY_diff,DriftZ_diff],dimensional,0);
figure(3);scatter3(locTable_out_NoC(:,2),locTable_out_NoC(:,4),locTable_out_NoC(:,3),5,locTable_out_NoC(:,3),'filled');
colormap(slanCM('viridis'));
h=colorbar('horizontal');  
set(h, 'Color', 'white');
set(gca, 'Color', 'k','fontsize',18, 'FontWeight', 'bold', 'FontName', 'Arial','LineWidth', 2); % 背景设置为黑色
set(gcf, 'Color', 'k');
set(gca, 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
grid off;
axis off; 
axis equal;
box on;
xlim([min(locTable_out_NoC(:,2)),max(locTable_out_NoC(:,2))]);
ylim([min(locTable_out_NoC(:,4)),max(locTable_out_NoC(:,4))]);
zlim([min(locTable_out_NoC(:,3)),max(locTable_out_NoC(:,3))]);
view(2);
locTable_out_NoCRoi = locTable_out_NoC(locTable_out_NoC(:,2)>110 & locTable_out_NoC(:,2)<140 & ...
    locTable_out_NoC(:,4)>80 & locTable_out_NoC(:,4)<110,:);
figure(4);scatter3(locTable_out_NoCRoi(:,2),locTable_out_NoCRoi(:,4),locTable_out_NoCRoi(:,3),35,locTable_out_NoCRoi(:,3),'filled');

colormap(slanCM('viridis')); 
h=colorbar('horizontal');  
set(h, 'Color', 'white');
set(gca, 'Color', 'k','fontsize',18, 'FontWeight', 'bold', 'FontName', 'Arial','LineWidth', 2); % 背景设置为黑色
set(gcf, 'Color', 'k'); 
set(gca, 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
grid off;
axis off;
axis equal;
box on;
xlim([min(locTable_out_NoCRoi(:,2)),max(locTable_out_NoCRoi(:,2))]);
ylim([min(locTable_out_NoCRoi(:,4)),max(locTable_out_NoCRoi(:,4))]);
zlim([min(locTable_out_NoCRoi(:,3)),max(locTable_out_NoCRoi(:,3))]);
view(2);

locTable_out = reconstruct(locTable_Cluster,pixelsize,framestep,[DriftX_diff,DriftY_diff,DriftZ_diff],dimensional,1);
figure(5);scatter3(locTable_out(:,2),locTable_out(:,4),locTable_out(:,3),5,locTable_out(:,3),'filled');
colormap(slanCM('viridis')); 
h=colorbar('horizontal');  
set(h, 'Color', 'white');
set(gca, 'Color', 'k','fontsize',18, 'FontWeight', 'bold', 'FontName', 'Arial','LineWidth', 2); % 背景设置为黑色
set(gcf, 'Color', 'k'); 
set(gca, 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
grid off;
axis off; 
axis equal;
box on;
xlim([min(locTable_out(:,2)),max(locTable_out(:,2))]);
ylim([min(locTable_out(:,4)),max(locTable_out(:,4))]);
zlim([min(locTable_out(:,3)),max(locTable_out(:,3))]);
view(2);
locTable_outRoi = locTable_out(locTable_out(:,2)>110 & locTable_out(:,2)<140 & ...
    locTable_out(:,4)>80 & locTable_out(:,4)<110,:);
figure(6);scatter3(locTable_outRoi(:,2),locTable_outRoi(:,4),locTable_outRoi(:,3),35,locTable_outRoi(:,3),'filled');
% xlabel('X (μm)');ylabel('Y (μm)'); zlabel('Z (μm)');% 设置背景颜色为黑色
colormap(slanCM('viridis')); % 使用'slanCM('viridis')'颜色图，也可以使用其他颜色图，如'parula', 'hsv', 'hot', 等等
h=colorbar('horizontal'); % 显示颜色条
set(h, 'Color', 'white');
set(gca, 'Color', 'k','fontsize',18, 'FontWeight', 'bold', 'FontName', 'Arial','LineWidth', 2); % 背景设置为黑色
set(gcf, 'Color', 'k'); % 整个窗口背景设置为黑色
set(gca, 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
grid off;
axis off; 
axis equal;
box on;
xlim([min(locTable_outRoi(:,2)),max(locTable_outRoi(:,2))]);
ylim([min(locTable_outRoi(:,4)),max(locTable_outRoi(:,4))]);
zlim([min(locTable_outRoi(:,3)),max(locTable_outRoi(:,3))]);
view(2);


%% rmse
DriftX = cumsum(DriftX_diff)*pixelsize;   
DriftY = cumsum(DriftY_diff)*pixelsize;
if strcmp(dimensional, '3D')
    DriftZ = cumsum(DriftZ_diff)*framestep;
    rmseZ = sqrt(mean((DriftZ - cumsum(DriftZ_diff_r)).^2));
end
rmseX = sqrt(mean((DriftX - cumsum(DriftX_diff_r)).^2));
rmseY = sqrt(mean((DriftY - cumsum(DriftY_diff_r)).^2));

%% Coordinate transformation
theta = 45*pi/180;
DriftX_r = cumsum(DriftX_diff_r);
DriftY_r = cumsum(DriftY_diff_r);
DriftZ_r = cumsum(DriftZ_diff_r);
dy = DriftY;
dz = DriftZ;
DriftY=  dy *cos(theta) - dz * sin(theta) ;
DriftZ = dy *sin(theta) + dz * cos(theta) ;
dy_r = DriftY_r;
dz_r = DriftZ_r;
DriftY_r=  dy_r *cos(theta) - dz_r * sin(theta) ;
DriftZ_r = dy_r *sin(theta) + dz_r * cos(theta) ;

figure(7);plot(DriftX_r,'k','Linewidth',3);
hold on; plot(DriftX,'--','Linewidth',3,'Color',[1, 0.5, 0]);
xlabel('Frame');
ylabel('X drift (nm)');
legend('GroundTruth', 'DTc', 'Box', 'off');
set(gca,'fontsize',20, 'FontName', 'Arial','LineWidth', 2);
xlim([0,frameNum]);
figure(8);plot(DriftY_r,'k','Linewidth',3);
hold on; plot(DriftY,'--','Linewidth',3,'Color',[1, 0.5, 0]);
xlabel('Frame');
ylabel('Z drift (nm)');
set(gca,'fontsize',20, 'FontName', 'Arial','LineWidth', 2);
xlim([0,frameNum]);
if strcmp(dimensional, '3D')
    figure(9);plot(DriftZ_r,'k','Linewidth',3);
    hold on; plot(DriftZ,'--','Linewidth',3,'Color',[1, 0.5, 0]);
    xlabel('Frame');
    ylabel('Y drift (nm)');
%     legend('Ground Truth', 'Drift');
    set(gca,'fontsize',20, 'FontName', 'Arial','LineWidth', 2);
    xlim([0,frameNum]);
end



