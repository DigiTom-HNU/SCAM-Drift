function [locTable_out,locTable_out_C] = reconstruct(locTable,Drift_diff,dimensional,Dedup)
Drift = [Drift_diff(:,1),cumsum(Drift_diff(:,2:4))];
locTable_out = [];
ClusterId = unique(locTable(:,1));
parfor i = 1 : length(ClusterId)
    locT = locTable(locTable(:,1) == ClusterId(i),:);
    %     if size(locT,1)<15
    %         continue;
    %     end
    
    %     [~,Id] = min(abs(locT(:,5)));
    Id = round(size(locT,1)/2)
    locTable_out = [locTable_out;locT(Id,1:end)];
end

if Dedup
    locTable_out_C = locTable_out;
else
    locTable_out_C = locTable;
end

locTable_out_C = sortrows(locTable_out_C,2);

[uniqueFrame,~,~] = unique(locTable_out_C(:,2));
counts = histcounts(locTable_out_C(:,2), [uniqueFrame; uniqueFrame(end)+1]);

% 
% uniqueFrame1 = uniqueFrame;
Drift = Drift(ismember(Drift(:,1),uniqueFrame),:);
DriftX_uniqueFrame = Drift(:,2);
DriftY_uniqueFrame = Drift(:,3);
% DriftX_uniqueFrame = DriftX(uniqueFrame1,:);
% DriftY_uniqueFrame = DriftY(uniqueFrame1,:);
DriftX_uniqueFrame_rep = repelem(DriftX_uniqueFrame,counts,1);
DriftY_uniqueFrame_rep = repelem(DriftY_uniqueFrame,counts,1);
locTable_out_C(:,3) = locTable_out_C(:,3) - DriftX_uniqueFrame_rep;
locTable_out_C(:,4) = locTable_out_C(:,4) - DriftY_uniqueFrame_rep;
if strcmp(dimensional, '3D')
    DriftZ_uniqueFrame = Drift(:,4);
%     DriftZ_uniqueFrame = DriftZ(uniqueFrame1,:);
    DriftZ_uniqueFrame_rep = repelem(DriftZ_uniqueFrame,counts,1);
    locTable_out_C(:,5) = locTable_out_C(:,5) - DriftZ_uniqueFrame_rep;
end
end