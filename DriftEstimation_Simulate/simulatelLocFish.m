% Generate simulated List-FISH spot position table
% by sqh
function [locTable,rawData,UpsamplingData,x_drift_diff,y_drift_diff,z_drift_diff] = simulatelLocFish(frameNum,SignalAppearFrameNumTheory,precision,framestep,pixelsize,FovSize,...
    driftRMSRange,AvgMolDens,R,ClusterPointNum)
FovSize = FovSize * pixelsize;
driftRMS = driftRMSRange(randi([1,length(driftRMSRange)]));
SignalAppearFrameNumActual = [SignalAppearFrameNumTheory-5,SignalAppearFrameNumTheory+5];
Roi = 15;
halfRoi = (Roi- 1)/2*pixelsize;
FovSizeXExpand = FovSize(1)*2;
FovSizeYExpand = FovSize(2)*2;
frameNumExpand = frameNum*1.5;
Xinitialposition = [FovSizeXExpand/2 - FovSize(1)/2 + halfRoi, FovSizeXExpand/2 + FovSize(1)/2 - halfRoi];
Yinitialposition = [FovSizeYExpand/2 - FovSize(2)/2 + halfRoi, FovSizeYExpand/2 + FovSize(2)/2 - halfRoi];
frames = 0;
Cycles = round(AvgMolDens*frameNum*(FovSizeXExpand/1000*(FovSizeYExpand/1000))/mean(SignalAppearFrameNumTheory)/ClusterPointNum);
circles = 0;
while frames<frameNum
    if circles>10
        disp('There is missing data，reset parameters.');
    end
    Drift = simulateDrift(driftRMS,frameNumExpand);
    Drift = Drift*pixelsize;
    xdrift = Drift(:,1);
    ydrift = Drift(:,2);
    zdrift = Drift(:,3);
    NewXYALL=[];
    rawData = [];
    UpsamplingData = [];
    parfor i = 1:Cycles
        deepFrame = randi(SignalAppearFrameNumActual);
        start = randi([-deepFrame, frameNum+deepFrame], 1, 1);
        x_c = rand * FovSizeXExpand; %
        y_c = rand * FovSizeYExpand;%(FovSize(2)- 100*pixelsize(1) - 100*pixelsize(1)) + 100*pixelsize(1); 
        
        %  CenterCoordinates= [CenterCoordinates;x_c,y_c];
        %  XYZ =GenSpherePoint(x_c,y_c,start,R ,ClusterPointNum,disThre);
        
        XYZ = GenCubicPoint(x_c, y_c, start,R,ClusterPointNum);
        XYZ(:,3) = round((XYZ(:,3)-start)/framestep+start);
        
        for j = 1:size(XYZ,1)
            x = XYZ(j,1);
            y = XYZ(j,2);
            NewXY = [repelem(XYZ(j,3),1)',x,y,0];%初始点
            NewXY1 = zeros(deepFrame, size(NewXY, 2));
            % 2024.05.30
            NewXY1(:,1) = NewXY(1,1) : NewXY(1,1)+deepFrame-1;
            NewXY1(:,2) = repelem(NewXY(:,2),deepFrame,1) + randsample(precision, deepFrame);%normrnd(0,precision,deepFrame,1);
            NewXY1(:,3) = repelem(NewXY(:,3),deepFrame,1) + randsample(precision, deepFrame);%normrnd(0,precision,deepFrame,1);
            frameSteRep = repelem(framestep,deepFrame-1);
            frameSteRep = cumsum(frameSteRep);
            NewXY1(:,3) = NewXY1(:,3) - [0;frameSteRep'];

            if size(NewXY1,1) == 0
                keyboard;
            end

            sizeNewXY1 = size(NewXY1,1);
            if mod(sizeNewXY1, 2) == 0
                left_num =sizeNewXY1/2-1;
                right_num =sizeNewXY1/2;
            else
                left_num =round(sizeNewXY1/2)-1;
                right_num =round(sizeNewXY1/2)-1;
            end
            centerZ = rand(1) * (100-(-100)) - 100;
            left_side = centerZ-framestep*left_num:framestep:centerZ-framestep ;
            right_side = centerZ + framestep:framestep:centerZ + (framestep * right_num);
            z = [left_side,centerZ,right_side]';
            [~,minZid] = min(abs(z));
            
            if isempty(NewXY1)
                continue;
            end
            z = z+randsample(precision, size(NewXY1,1));%normrnd(0,precision,size(NewXY1,1),1);
            UpsamplingDataT = [NewXY1(:,1:3),z];
            rawDataT = UpsamplingDataT(minZid,:);
            rawData = [rawData;rawDataT];
            UpsamplingData = [UpsamplingData;UpsamplingDataT];
            
            zdriftT = zeros(size(NewXY1,1),2);
            zdriftT(:,1) = NewXY1(:,1);
            frame = NewXY1(NewXY1(:,1)>0&NewXY1(:,1)<=frameNum,1);
            zdriftT(ismember(zdriftT(:,1),frame),2) = zdrift(frame);
            NewXY1(:,4) = z + zdriftT(:,2);
            NewXYALL = [NewXYALL;NewXY1];
        end
    end
    NewXYALL = sortrows(NewXYALL,1);
    NewXYALL = NewXYALL(NewXYALL(:,1)>0 & NewXYALL(:,1)<=frameNum,:);
    locTable = [];
    [uniqueF, ~, ~] = unique(NewXYALL(:, 1));
    for i = 1:length(uniqueF)
        loc = NewXYALL(NewXYALL(:,1)==uniqueF(i),:);
        xdriftT = xdrift(uniqueF(i));
        ydriftT = ydrift(uniqueF(i));
        locT = loc(loc(:,2)>Xinitialposition(1)-xdriftT & loc(:,2)<Xinitialposition(2)-xdriftT,:);
        locT(:,2) = locT(:,2) - (Xinitialposition(1)-xdriftT);
        locT = locT(locT(:,3)>Yinitialposition(1)-ydriftT & locT(:,3)<Yinitialposition(2)-ydriftT,:);
        locT(:,3) = locT(:,3) - (Yinitialposition(1)-ydriftT);
        locTable = [locTable;locT];
    end
    locTable = sortrows(locTable,[1,2,3,4]);
    locTable = locTable(locTable(:,2)>halfRoi & locTable(:,2)<FovSize(1)-halfRoi &...
        locTable(:,3)>halfRoi & locTable(:,3)<FovSize(2)-halfRoi &...
        locTable(:,4)>-round(SignalAppearFrameNumTheory)/2*framestep-500 &...
        locTable(:,4)<round(SignalAppearFrameNumTheory)/2*framestep+500,:);
    rawData = rawData(rawData(:,2)>Xinitialposition(1) & rawData(:,2)<Xinitialposition(2) &...
        rawData(:,3)>Yinitialposition(1) & rawData(:,3)<Yinitialposition(2)& rawData(:,1)>0 &...
        rawData(:,1)<=frameNum ,:);
    rawData(:,2) = rawData(:,2) - Xinitialposition(1);
    rawData(:,3) = rawData(:,3) - Yinitialposition(1);
    rawData = rawData(rawData(:,2)>halfRoi & rawData(:,2)<FovSize(1)-halfRoi &...
        rawData(:,3)>halfRoi & rawData(:,3)<FovSize(2)-halfRoi &...
        rawData(:,4)>-round(SignalAppearFrameNumTheory)/2*framestep-500 &...
        rawData(:,4)<round(SignalAppearFrameNumTheory)/2*framestep+500,:);
    UpsamplingData = UpsamplingData(UpsamplingData(:,2)>Xinitialposition(1) & UpsamplingData(:,2)<Xinitialposition(2) &...
        UpsamplingData(:,3)>Yinitialposition(1) & UpsamplingData(:,3)<Yinitialposition(2)& UpsamplingData(:,1)>0 &...
        UpsamplingData(:,1)<=frameNum ,:);
    UpsamplingData(:,2) = UpsamplingData(:,2) - Xinitialposition(1);
    UpsamplingData(:,3) = UpsamplingData(:,3) - Yinitialposition(1);
    UpsamplingData = UpsamplingData(UpsamplingData(:,2)>halfRoi & UpsamplingData(:,2)<FovSize(1)-halfRoi &...
        UpsamplingData(:,3)>halfRoi & UpsamplingData(:,3)<FovSize(2)-halfRoi &...
        UpsamplingData(:,4)>-round(SignalAppearFrameNumTheory)/2*framestep-500 &...
        UpsamplingData(:,4)<round(SignalAppearFrameNumTheory)/2*framestep+500,:);
    frames = length(unique(locTable(:,1)));
    circles = circles+1;
end
rawData = sortrows(rawData,1);
UpsamplingData = sortrows(UpsamplingData,1);
Drift = Drift(1:frameNum,:);


[uniqueFrame,~,~] = unique(locTable(:,1));
x_drift_diff = zeros(size(Drift, 1), 1);
x_drift_diff(uniqueFrame) = Drift(uniqueFrame, 1);
y_drift_diff = zeros(size(Drift, 1), 1);
y_drift_diff(uniqueFrame) = Drift(uniqueFrame, 2);
z_drift_diff = zeros(size(Drift, 1), 1);
z_drift_diff(uniqueFrame) = Drift(uniqueFrame, 3);

uniqueFrameE = zeros(frameNum,1);
uniqueFrameE(uniqueFrame) = uniqueFrame;

positions = find(uniqueFrameE ~= 0);
gaps = diff(positions) > 1;
segment_starts = [positions(1); positions(find(gaps) + 1)];
segment_ends = [positions(find(gaps)); positions(end)];

for i = 1:length(segment_starts)
    frameID = segment_starts(i):segment_ends(i);
    x_drift_diff(frameID) = [0; diff(x_drift_diff(frameID))];
    y_drift_diff(frameID) = [0; diff(y_drift_diff(frameID))];
    z_drift_diff(frameID) = [0; diff(z_drift_diff(frameID))];
end

end