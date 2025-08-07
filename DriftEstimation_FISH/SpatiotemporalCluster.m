function  arry = SpatiotemporalCluster(arry,r,pixelsize,dimensional)
% arry = sortrows(arry,1);
if strcmp(dimensional, '3D')
    col = 5;
else
    col = 4;
end
r = r * pixelsize(2)/1000;
% 【insert clusterID】
arry = [zeros(size(arry, 1), 1), arry];
arry(:, 1) = (1:size(arry, 1))';

%% 【get the localization points of the j and j+1 frames】
frameStart = arry(1, 2);
frameEnd   = arry(end, 2);

UpdateIdInfo = [];
frames = frameStart+1:frameEnd;
% theta = 46.5*pi/180;
parfor i  = 1:length(frames)  % will read until frameEnd-1
    fCur = frames(i);
    fBefor = fCur-1;
    
    fCurData   = arry(arry(:, 2) == fCur, :);    % copy operation
    fBeforData = arry(arry(:, 2) == fBefor, :);
    
    if isempty(fCurData) || isempty(fBeforData)
        continue;
    end

    UpdateIdInfoT = [];
    for j = 1:size(fCurData, 1)
        tree = KDTreeSearcher(fBeforData(:,3:col));
        [index,distance] = knnsearch(tree,fCurData(j,3:col),'k',1);
        if distance<r
            linkID = [fCurData(j,1),fBeforData(index,1)];
        else
            linkID = [];
        end
        UpdateIdInfoT = [UpdateIdInfoT; linkID];
        if ~isempty(UpdateIdInfoT)
            [~,uniqueUpdateIdInfoTid] = unique(UpdateIdInfoT(:,2));
            UpdateIdInfoT = UpdateIdInfoT(uniqueUpdateIdInfoTid,:);
        end
    end
    UpdateIdInfo = [UpdateIdInfo; UpdateIdInfoT];
end
for i = 1:length(UpdateIdInfo)
    arry(UpdateIdInfo(i,1),1) = arry(UpdateIdInfo(i,2),1);
end

% sort in ascending order from front to back
[~, sortIdx] = sort(arry(:, 1));
arry = arry(sortIdx, :);

% mark clusterID starting from 0 sequentially
clusterID = 0;
befValueTemp = arry(1, 1);
for curIIndex = 1:size(arry, 1)
    curValue  = arry(curIIndex, 1);
    if curValue ~= befValueTemp
        clusterID = clusterID + 1;
        befValueTemp = arry(curIIndex, 1);
        arry(curIIndex, 1) = clusterID;
    else
        arry(curIIndex, 1) = clusterID;
    end
end
end