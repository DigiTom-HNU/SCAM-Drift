function   arry = SpatiotemporalCluster(arry,r,frameStep,pixelsize,dimensional)
if strcmp(dimensional, '3D')
    col = 5;
else
    col = 4;
end
% 【insert clusterID】
arry = [zeros(size(arry, 1), 1), arry];
arry(:, 1) = (1:size(arry, 1))';

%% 【get the localization points of the j and j+1 frames】
frameStart = arry(1, 2);
frameEnd   = arry(end, 2);

% displaceInfoCol = data_eff(data_eff(:,1) == arry(1,end),:);
UpdateIdInfo = [];
frames = frameStart+1:frameEnd;
parfor i  = 1:length(frames)  % will read until frameEnd-1
    fCur = frames(i);
    fBefor = fCur-1;
    
    fCurData   = arry(arry(:, 2) == fCur, :);    % copy operation
    fBeforData = arry(arry(:, 2) == fBefor, :);
    
    if isempty(fCurData) || isempty(fBeforData)
        continue;
    end
    
    %     fCurdisplaceInfo = displaceInfoCol(displaceInfoCol(:,3) == fCur,:);
    %     fBeforedisplaceInfo = displaceInfoCol(displaceInfoCol(:,3) == fBefor,:);
    %     disYmoveUp = abs(fCurdisplaceInfo(:,6)-fBeforedisplaceInfo(:,6))*1000000/sqrt(2)/parameter.pixelsize(2);
    disYmoveUp = frameStep*abs(fCur-fBefor)/pixelsize;
    disZmoveUp = abs(fCur-fBefor);
    fCurData(:,4) = fCurData(:,4) + disYmoveUp;
    if strcmp(dimensional, '3D')
        fCurData(:,5) = fCurData(:,5) - disZmoveUp;
    end
    UpdateIdInfoT = [];
    for j = 1:size(fCurData, 1)
        tree = KDTreeSearcher(fBeforData(:,3:col));
        [index,distance] = knnsearch(tree,fCurData(j,3:col),'k',1);
        if distance<r
            linkID = [fCurData(j,1),fBeforData(index,1)];
            %             if ~isempty(linkIDAfter)&&any(linkIDAfter(2)==linkID(2))
            %                 linkID = [];
            %             end
        else
            %             fCur1 = fCur;
            %             frameStart1 = frameStart;
            %             while distance >= r && fCur > frameStart1 + 1 && frameStart1 + 1 <= frameStart+mulFrameAna - 2  %现在是最多和前三帧比较，再多就frameStart+2 再往上加
            %                 fBefor2 = fCur1-2;
            %                 fBefor2Data = arry(arry(:, 2) == fBefor2, :);
            % %                 fBefore2displaceInfo = displaceInfoCol(displaceInfoCol(:,3) == fBefor2,:);
            % %                 disYmoveUpT = abs(abs(fCurdisplaceInfo(:,6)-fBefore2displaceInfo(:,6))*1000000/sqrt(2)/parameter.pixelsize(2)-disYmoveUp);
            %                 disYmoveUpT = frameStep*abs(fCur-fBefor2)/pixelsize - disYmoveUp;
            %                 tree = KDTreeSearcher(fBefor2Data(:,3:col));
            %                 if strcmp(dimensional, '3D')
            %                     disZmoveUpT = abs(fCur-fBefor2) - disZmoveUp;
            %                     [index,distance] = knnsearch(tree,[fCurData(j,3),fCurData(j,4)+disYmoveUpT,fCurData(j,5)-disZmoveUpT],'k',1);
            %                 elseif  strcmp(dimensional, '2D')
            %                      [index,distance] = knnsearch(tree,[fCurData(j,3),fCurData(j,4)+disYmoveUpT],'k',1);
            %                 end
            %                 fCur1 = fCur1 -1;
            %                 frameStart1 = frameStart1 + 1;
            %             end
            %             if distance<r
            %                 linkID = [fCurData(j,1),fBefor2Data(index,1)];
            % %                 if ~isempty(UpdateIdInfo)&&any(UpdateIdInfo(:,2)==linkID(2))
            % %                     linkID = [];
            % %                 end
            %             else
            %                 linkID = [];
            %             end
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