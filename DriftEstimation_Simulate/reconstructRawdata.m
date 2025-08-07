function Rawdata = reconstructRawdata(Rawdata,framestep)
theta = 45*pi/180;
Rawdata(:,2) = Rawdata(:,2) / 1000;
Rawdata(:,3) = Rawdata(:,3) / 1000;
Rawdata(:,4) = Rawdata(:,4) / 1000;
Rawdata(:,2) = Rawdata(:,2);
y = Rawdata(:,3);
Rawdata(:,3) = y *cos(theta) - Rawdata(:,4) * sin(theta) ;
Rawdata(:,4) = y *sin(theta) + Rawdata(:,4) * cos(theta) ;

displaceInfo = Rawdata(:,1) * framestep * sqrt(2)/1000;
displaceInfo = displaceInfo - displaceInfo(1);

Rawdata(:,3) = Rawdata(:,3) + displaceInfo;