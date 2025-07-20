load('FastStrafeNetwork.mat')
load("testNNdata.mat")
Ts = 1;       
rb = 7.5;
nx = 3;    


Ypredict = predict(imitateMPCNetwork, testDataInput);
rmse(testDataOutput, Ypredict)
mean(abs(Ypredict - testDataOutput))

testData = zeros(1e3,10); %start, goal1, goal2, nextTest, nextReal

divisors =  [144, 144, 3.1415, 144, 3.1415, 144, 3.1415, 1];%, 64.2455, 64.2455, 64.2455, 64.2455];
number = 1;
list = zeros(10,3);
Data(number,:);
x0 = [Data(number,1), Data(number,2), Data(number,3)];
goal1 = [Data(number,8),  Data(number,9), Data(number,10)];
goal2 = [Data(number,11), Data(number,12),  Data(number,13)];
relativeG1 = [Data(number,8) - x0(1), Data(number,9) - x0(2)];
relativeG2 = [Data(number,11) - x0(1), Data(number,12) - x0(2)];
originalX0 = x0;
u = [0;0;0;0];
goalswap = 0;
for g = 1:30
    dist1  = sqrt((goal1(1) - x0(1))^2 + (goal1(2) -  x0(2))^2);
   angle1 = atan2(goal1(2) -  x0(2), goal1(1)  -  x0(1));
   dist2  = sqrt((goal2(1) - x0(1))^2 + (goal2(2) -  x0(2))^2);
   angle2 = atan2(goal2(2) -  x0(2), goal2(1)  -  x0(1));
   distAngle1  = [dist1, angle1, goal1(3)];
    distAngle2 = [dist2,angle2, goal2(3)];
    if dist1 <= 5  && goalswap ~= 1
        goalswap  = 1;
    end
    inputData = [x0(1)-originalX0(1),x0(2)-originalX0(2), x0(3), distAngle1(1), distAngle1(2), distAngle2(1), distAngle2(2), goalswap]%, u(1), u(2), u(3), u(4)];

    inputData = inputData ./ divisors;
    Ypredict = predict(imitateMPCNetwork, inputData);
    u = Ypredict';
    u = u*64.2455;
    x0 = mecanumStateFcn(x0', u, Ts, rb, nx)';
    x0(3) = wrapToPi(x0(3));
    x0;
    list(g,:) = x0';
end

%plotTestData(1, testData)
plotMPC(number,trajectories, supposedTrajectory)

plot(list(:,1), list(:,2), 'go')

function plotTestData(ct, testData)
    clf
    hold on
    waypoints = [testData(ct, 1) testData(ct, 2)
             testData(ct, 3),testData(ct, 4)
             testData(ct, 5), testData(ct,6)]
    plot(waypoints(:,1), waypoints(:,2))
    plot(testData(ct,1), testData(ct,2), 'go')
    plot(testData(ct,7), testData(ct,8), 'ro')
    plot(testData(ct,9), testData(ct,10), 'go')
end


function xnext = mecanumStateFcn(x, u, Ts, rb, nx)
    % Forward kinematics matrix (body velocities)
    J = 1/4 * [% [fr, fl, br, bl]
        1/2,  -1/2, -1/2,  1/2; %x
        1, 1, 1, 1; %y
       1/(2*rb), -1/(2*rb), 1/(2*rb), -1/(2*rb)
    ];

    
    A = eye(nx);
    B = Ts * J;
    first = A * x;
    second =  B  * u;

    theta = x(3) + second(3);
    xvalue = (second(1) * sin(theta)) +  (second(2) * cos(theta));
    yvalue = (second(1) * cos(theta)) +  (second(2) * sin(theta));
    xnext = [xvalue;
             yvalue;
             theta];
    xnext = xnext  + first;
end

function plotMPC(ct, trajectories, supposedTrajectory)
    clf
    hold on
    for i = 1:10
        plot(trajectories(ct*10-10+i, 1), trajectories(ct*10-10 + i,2),  'ro')
    end    
    supposedX =nonzeros(supposedTrajectory(ct*2-1,:))';
    supposedY =nonzeros(supposedTrajectory(ct*2,:))';
    plot(supposedX, supposedY)

end