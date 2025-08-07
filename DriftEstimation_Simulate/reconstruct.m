function locTable_out = reconstruct(locTable,pixelsize,framestep,Drift_diff,dimensional,correct)
locTable_out = [];
ClusterId = unique(locTable(:,1));
for i = 1 : length(ClusterId)
    locT = locTable(locTable(:,1) == ClusterId(i),:);
    if size(locT,1)<15
        continue;
    end
    [~,minId] = min(abs(locT(:,5)));
    locTable_out = [locTable_out;locT(minId,2:end)];
end
locTable_out = sortrows(locTable_out,1);
if correct
    DriftX = cumsum(Drift_diff(:,1));
    DriftY = cumsum(Drift_diff(:,2));
    
    [uniqueFrame,~,~] = unique(locTable_out(:,1));
    DriftX_uniqueFrame = DriftX(uniqueFrame,:);
    DriftY_uniqueFrame = DriftY(uniqueFrame,:);
    counts = histcounts(locTable_out(:,1), [uniqueFrame; uniqueFrame(end)+1]);
    DriftX_uniqueFrame_rep = repelem(DriftX_uniqueFrame,counts,1);
    DriftY_uniqueFrame_rep = repelem(DriftY_uniqueFrame,counts,1);
    locTable_out(:,2) = locTable_out(:,2) - DriftX_uniqueFrame_rep;
    locTable_out(:,3) = locTable_out(:,3) - DriftY_uniqueFrame_rep;
    if strcmp(dimensional, '3D')
        DriftZ = cumsum(Drift_diff(:,3));
        DriftZ_uniqueFrame = DriftZ(uniqueFrame,:);
        DriftZ_uniqueFrame_rep = repelem(DriftZ_uniqueFrame,counts,1);
        locTable_out(:,4) = locTable_out(:,4) - DriftZ_uniqueFrame_rep;
    end
end
theta = 45*pi/180;
locTable_out(:,2) = locTable_out(:,2) * pixelsize / 1000;
locTable_out(:,3) = locTable_out(:,3) * pixelsize / 1000;
locTable_out(:,4) = locTable_out(:,4) * framestep / 1000;
locTable_out(:,2) = locTable_out(:,2);
y = locTable_out(:,3);
locTable_out(:,3) = y *cos(theta) - locTable_out(:,4) * sin(theta) ;
locTable_out(:,4) = y *sin(theta) + locTable_out(:,4) * cos(theta) ;

displaceInfo = locTable_out(:,1) * framestep * sqrt(2)/1000;
displaceInfo = displaceInfo - displaceInfo(1);

locTable_out(:,3) = locTable_out(:,3) + displaceInfo;

end

