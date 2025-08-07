function  [Drift_diff_out,reconstructData_out,locTable_out] = CorrctAndReconstruct(path,frameNum,pixelsize,dimensional,r,colrow)
[data_eff,~] = CameraAndStageInfo(path);
load([path,'\locTable.mat']);
locTableR = locTableR(locTableR(:,5)>-1500 &locTableR(:,5)<1500,:);
locTableR = locTableR(ismember(locTableR(:,end),colrow{1})& ismember(locTableR(:,2),colrow{2}),:);
reconstructData_out = cell(1, length(colrow{1}));
Drift_diff_out = cell(1, length(colrow{1}));
locTable_out = cell(1, length(colrow{1}));
for i = 1:length(colrow{1})
    locPoint = locTableR(locTableR(:,end)== colrow{1}(i),1:5);
    locPoint = sortrows(locPoint,2);
    displaceInfo = data_eff(data_eff(:,1)== colrow{1}(i) & ismember(data_eff(:,3),colrow{2}),:);%改
    displaceInfo(:, 6) = displaceInfo(:, 6) - min(data_eff(:, 6));
    displaceInfo(:, 7) = displaceInfo(:, 7) - min(data_eff(:, 7));
    displaceInfo(:, 8) = displaceInfo(:, 8) - min(data_eff(:, 8));
    locPoint(:,3) = locPoint(:,3)*pixelsize(1);
    locPoint(:,4) = locPoint(:,4)*pixelsize(2);
    theta = 46.1*pi/180;
    locPoint(:,3) = locPoint(:,3)  / 1000;
    locPoint(:,4) = locPoint(:,4)  / 1000;
    locPoint(:,5) = locPoint(:,5)  / 1000;
    locPoint(:,3) = locPoint(:,3);
    y = locPoint(:,4);
    locPoint(:,4) = y *cos(theta) - locPoint(:,5) * sin(theta) ;
    locPoint(:,5) = y *sin(theta) + locPoint(:,5) * cos(theta) ;
    [uniqueFrame,~,~] = unique(locPoint(:,2));
    counts = histcounts(locPoint(:,2), [uniqueFrame; uniqueFrame(end)+1]);
    displaceInfoFramejudge = ismember(displaceInfo(:,3),uniqueFrame);
    displaceInfo = displaceInfo(displaceInfoFramejudge,:);
    displaceInfo_rep = repelem(displaceInfo,counts,1);
    locPoint(:,3) = locPoint(:,3) - displaceInfo_rep(:,7) * 1000;
    locPoint(:,4) = locPoint(:,4) + displaceInfo_rep(:,6) * 1000;
    locPoint(:,5) = locPoint(:,5) + displaceInfo_rep(:,8) * 1000;
    locTableT = locPoint(:,2:end);
    locTableT = sortrows(locTableT,1);
    
    locTable_Cluster = SpatiotemporalCluster(locTableT,r,pixelsize,dimensional);

    
    [DriftX_diff1,DriftY_diff1,DriftZ_diff1] = DriftEstimate(locTable_Cluster,frameNum,dimensional,colrow);
    Drift_diff  = [(colrow{2})',DriftX_diff1,DriftY_diff1];
    if strcmp(dimensional, '3D')
        Drift_diff = [Drift_diff,DriftZ_diff1];
    end
    reconstructData = locPoint(:,1:5);
    if size(reconstructData,2) <5
        reconstructData = [reconstructData,zeros(size(reconstructData,1),1)];
    end
    reconstructData_out{i} = reconstructData;

    [locTable,locTable_C_1] = reconstruct(reconstructData,Drift_diff,dimensional,0);
    locTable_C_11 = sortrows(locTable_C_1(:,2:end),1);
    locTable_Cluster2 = SpatiotemporalCluster(locTable_C_11,r,pixelsize,dimensional);%因为locTable_C_1的每个簇不连续 所以要重聚
    %
    [DriftX_diff2,DriftY_diff2,DriftZ_diff2] = DriftEstimate(locTable_Cluster2,frameNum,dimensional,colrow);
    Drift_diff(:,2) = Drift_diff(:,2) + DriftX_diff2;
    Drift_diff(:,3) = Drift_diff(:,3) + DriftY_diff2;
    if strcmp(dimensional, '3D')
        Drift_diff(:,4) = Drift_diff(:,4) + DriftZ_diff2;
    end
    Drift_diff_out{i} = Drift_diff;
    locTable_out{i} = locTable;
end
end
