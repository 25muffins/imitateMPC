waypoints = [0, 0
             3, 10
             5, 5
             9, 9
             9, 10
             11, 12
             18, 20
             16, 9
             5, 2
             50, 20];

hold off
hold on
plot(waypoints(:,1), waypoints(:,2))
xRef = waypoints(:,1);
yRef = waypoints(:,2);

distanceSteps = sqrt(sum(diff(waypoints).^2, 2)); %distances
distAtEachStep = cumsum([0; distanceSteps]); % distance for each waypoint
totalDist = sum(distanceSteps);
pointsInBetween = linspace(0, totalDist, 100);%how many points in between


%at this distance, we are at this x value, in between distances, we are at
%t
x2 = interp1(distAtEachStep, xRef, pointsInBetween, 'makima');
y2 = interp1(distAtEachStep, yRef, pointsInBetween,'makima');
plot(x2, y2);
x3 = interp1(distAtEachStep, xRef, pointsInBetween, 'spline');
y3 = interp1(distAtEachStep, yRef, pointsInBetween,'spline');
plot(x3, y3);
x4 = interp1(distAtEachStep, xRef, pointsInBetween, 'pchip');
y4 = interp1(distAtEachStep, yRef, pointsInBetween,'pchip');
plot(x4, y4);




save("PathXY.mat", "x4", "y4", 'xRef', 'yRef')