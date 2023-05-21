function s = propose_special_region()
%PROPOSE_SPECIAL_REGION Return samples from the crescent-shaped region with
%measure dmu(x,y) = (x^2 + y^2) dx dy

while true
    r = 2*rand()^(1/4);
    theta = 2*pi*rand();
    x = r*cos(theta);
    y = r*sin(theta);
    if (x-1)^2 + y^2 > 1
        break
    end
end
s = [x;y];
end