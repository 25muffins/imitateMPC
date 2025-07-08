function [x0, u0, goal1, goal2, ref] = randomDataNLMPC

%need to make x0, u0, goal1, goal2, 6inches in between each dot
maxSpeed = 64.2455;


%72 -> -72 inches
xRand = 144*rand -72;
yRand = 144*rand -72;
thetaRand = 2*pi  * rand - pi;

x0 =  [xRand, yRand, thetaRand]';

%u0 range - 64.2455
u0 = [(2 * 64.2455 * rand) - 64.2455, (2 * 64.2455 * rand) - 64.2455, (2 * 64.2455 * rand) - 64.2455, (2 * 64.2455 * rand) - 64.2455]';


%72 -> -72 inches
xRand1 = 144*rand -72;
yRand1 = 144*rand -72;
thetaRand1 = 2*pi  * rand - pi;
goal1 = [xRand1, yRand1, thetaRand1];

xRand2 = 144*rand -72;
yRand2 = 144*rand -72;
thetaRand2 = 2*pi  * rand - pi;
goal2 = [xRand2, yRand2, thetaRand2];

waypoints =  [x0';goal1; goal2];
xRef =  waypoints(:,1);
yRef =  waypoints(:,2);
hold off
hold on
%plot(waypoints(:,1), waypoints(:,2));

distanceSteps = sqrt(sum(diff(waypoints).^2, 2)); %distances
distAtEachStep = cumsum([0; distanceSteps]); % distance for each waypoint
totalDist = sum(distanceSteps);

howManyDots = ceil(totalDist/6);
pointsInBetween = linspace(0, totalDist, howManyDots);%how many points in between
%at this distance, we are at this x value, in between distances, we are at
%t
x2 = interp1(distAtEachStep, xRef, pointsInBetween, 'makima');
y2 = interp1(distAtEachStep, yRef, pointsInBetween,'makima');
%plot(x2, y2);
x3 = interp1(distAtEachStep, xRef, pointsInBetween, 'spline');
y3 = interp1(distAtEachStep, yRef, pointsInBetween,'spline');
%plot(x3, y3);
x4 = interp1(distAtEachStep, xRef, pointsInBetween, 'pchip');
y4 = interp1(distAtEachStep, yRef, pointsInBetween,'pchip');
%plot(x4, y4, 'ro');

path=[x4', y4']; %make sure to invert
theta_path = atan2(diff(path(:,2)), diff(path(:,1))); %atan2 does all 4 quadrants
theta_path = [theta_path; theta_path(end)]; %rads, add one more to end maybe add goal2(3)
fullPath = [path, theta_path];

if(rand >  0.1)
    initial_theta = atan2(yRef(2) - yRef(1), xRef(2) - xRef(1));
    x0 =  [xRand, yRand, initial_theta]';

end
ref = fullPath;
end