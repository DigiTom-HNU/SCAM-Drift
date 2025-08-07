function Drift = simulateDrift(driftRMS,frameNUM)
n = 400;
driftSTD = driftRMS*frameNUM/n;
drift_xyz = normrnd(0, driftSTD,n,3);
% drift_xyz(drift_xyz > upper_quantile) = upper_quantile;
% drift_xyz(drift_xyz < lower_quantile) = lower_quantile;
drift_xyz = cumsum(drift_xyz,1);
drift_xyz = drift_xyz - drift_xyz(1,:);
ratio = (frameNUM-1)/(n-1);
driftX = interp1(1:ratio:frameNUM,drift_xyz(:,1) - drift_xyz(1,1),1:frameNUM,'spline');
driftY = interp1(1:ratio:frameNUM,drift_xyz(:,2) - drift_xyz(1,2),1:frameNUM,'spline');
driftZ = interp1(1:ratio:frameNUM,drift_xyz(:,3) - drift_xyz(1,3),1:frameNUM,'spline');
Drift = [driftX;driftY;driftZ]';
end
