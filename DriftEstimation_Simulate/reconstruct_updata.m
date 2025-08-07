function locTable = reconstruct_updata(locTable,framestep)

theta = 45*pi/180;
locTable(:,2) = locTable(:,2) / 1000;
locTable(:,3) = locTable(:,3)  / 1000;
locTable(:,4) = locTable(:,4)  / 1000;
locTable(:,2) = locTable(:,2);
y = locTable(:,3);
locTable(:,3) = y *cos(theta) - locTable(:,4) * sin(theta) ;
locTable(:,4) = y *sin(theta) + locTable(:,4) * cos(theta) ;

displaceInfo = locTable(:,1) * framestep * sqrt(2)/1000;
displaceInfo = displaceInfo - displaceInfo(1);

locTable(:,3) = locTable(:,3) + displaceInfo;

end