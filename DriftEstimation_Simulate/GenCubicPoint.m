function XYZ = GenCubicPoint(x0, y0, z0,R,ClusterPointNum)
% inpput:(x0,y0,z0) as sphere center, radius R, ClusterPointNum points, 

dotN = round(sqrt(ClusterPointNum));
d = 2*R/dotN;
x = zeros(1,dotN*dotN);
y = zeros(1,dotN*dotN);
z = zeros(1,dotN*dotN);

a0 = rand()*pi*2;
for r=0:(dotN-1)
    x1 = x0 + r*d*sin(a0);
    y1 = y0 + r*d*cos(a0);
    for c=1:dotN
        x(1,r*dotN+c) = x1 + (c-1)*d*cos(2*pi-a0);
        y(1,r*dotN+c) = y1 + (c-1)*d*sin(2*pi-a0);
        z(1,r*dotN+c) = z0 + (rand()-0.5)*2*R;
    end
end
XYZ = [x',y',z'];
end