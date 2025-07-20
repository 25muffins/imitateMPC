function MPCasPathmaker()
% Creates the MAT file 'InputDataFileImLKA.mat' based on the value of
% 'isRandom'

% If isRandom is true (1), then random input data is generated and stored in
% the 'InputDataFileImLKA.mat' If isRandom is false (0), then input data is
% generated from the grid formed by changing the values of lateral
% velocity, yaw angle rate, lateral deviation, relative yaw angle, last
% steering angle, curvature under the 'variables' section in the script,
% and stores the data in 'InputDataFileImLKA.mat'

% Copyright 2019 The MathWorks, Inc.


% use a different seed such as rng('shuffle') to create differing data
rng(1)

% Generate MPC object.
Ts = 1;         % Sampling time
rb = 7.5; % track radius
maxRev = 300 / 60; %per second
wheelCircumference = 4.09 * pi;

nu = 4;           % [fr, fl, br, bl]
nx = 6;           % [x; y; theta; vx, vy, omega]
ny = 3;
N=30;
planner = nlmpcMultistage(N, nx, nu);
planner.Ts = Ts;


planner.Model.StateFcn = @mecanumStateFcn;
planner.Model.IsContinuousTime = false;
for i = 1:4
    planner.MV(i).Min = -maxRev * wheelCircumference;
    planner.MV(i).Max = maxRev * wheelCircumference;
end
for i = 2:N
    planner.Stages(i).CostFcn = @stageCost;
    planner.Stages(i).ParameterLength = 6;
end
planner.Stages(N+1).CostFcn = @terminalCost;
planner.Stages(N+1).ParameterLength = 6;
goal = [1;1;1; 0; 0;  0];
simData.StageParameter = repmat(goal', N, 1);
simData = getSimulationData(planner, 'TerminalState');
simData.TerminalState = goal;

validateFcns(planner, zeros(nx,1), zeros(nu,1), simData);



goal1 = [50, -50, 0.5,  0, 0, 0];
goal2 = [-3, 20, 2, 0, 0, 0];
waypoints =  [goal1;  goal2];

clf
% Initialize
x = [0; 0; 0; 0; 0; 0];
u = zeros(nu,1);
size(waypoints,1)

midGoal = waypoints(1,:)'
finalGoal = waypoints(2,:)'
for i = 2:N
    planner.Stages(i).CostFcn = @stageCost;
    planner.Stages(i).ParameterLength = 6;
end
simData = getSimulationData(planner , 'TerminalState');
planner.Stages(N+1).CostFcn = @terminalCost;
planner.Stages(N+1).ParameterLength = 6;
sp =  [repmat(midGoal, N/2, 1); repmat(finalGoal, N/2, 1)];
simData.StageParameter = sp;

simData.TerminalState = finalGoal;

[u, ~, info] = nlmpcmove(planner, x, u, simData);
simData.StageParameter;
xTrackHistory = info.Xopt
uHistory  = info.MVopt;
hold on
plot(xTrackHistory(:, 1),xTrackHistory(:, 2))
plot(1:31, xTrackHistory(1:31,3))
a  = size(uHistory);
x0 = [0;0;0;0;0;0];
for  i = 1:a(1)

    plot(x0(1), x0(2), 'bo');
    x0 =  mecanumStateFcn(x0, uHistory(i,:)')
    
end
plot(waypoints(:,1), waypoints(:,2), 'go');
Data = zeros(10,10);
% Create MAT file
save('MPCmadePath','Data')


function returnState = mecanumStateFcn(x, u)
    % Forward kinematics matrix (body velocities)
    J = 1/4 * [
        1,  -1, -1,  1; %x
        1, 1, 1, 1; %y
       1/(2*7.5), -1/(2*7.5), 1/(2*7.5), -1/(2*7.5)
    ];
    A = eye(6);
    B = 1 * J;
    first = A * x;
    second =  B  * u;
    x(3) = wrapToPi(x(3));
    theta = x(3) + second(3);
    theta = wrapToPi(theta);
    xvalue = (second(1) * sin(theta)) +  (second(2) * cos(theta));
    yvalue = (second(1) * cos(theta)) +  (second(2) * sin(theta));
    xvel = [xvalue;
             yvalue;
             theta];
    xnext = [xvalue + first(1);
             yvalue + first(2);
             theta + first(3)];
    xnext(3) = wrapToPi(xnext(3));
    returnState  = [xnext; xvel];

end
function c = stageCost(stage,x,u, stageParam)
    goal = stageParam;
    posErr = norm(x(1:2) - goal(1:2));
    angErr = wrapToPi(x(3) - goal(3));
    velErr = norm(x(4:5)); 
    angVelErr = abs(x(6)); 
    

    distanceToGoal = posErr;

    posWeight = 500;   
    angWeight = 50;
    trackingCost = posWeight * posErr^2 + angWeight * angErr^2;

    velCost = 2000;
    velocityCost = velCost * velErr^2 + 20 * angVelErr^2;
    controlCost = 0.1 * sum(u.^2);
    controlSmoothness = 0.1 * sum(abs(diff(u)));
    
    % Combine costs
    c = trackingCost + velocityCost + controlCost;
end
function c = terminalCost(stage, x,u, stageParam)
    goal = stageParam;

    posErr = norm(x(1:2) - goal(1:2));
    angErr = wrapToPi(x(3) - goal(3));
    velErr = norm(x(4:5));
    angVelErr = abs(x(6));
    
    positionCost = 1000 * posErr^2;      
    orientationCost = 500 * angErr^2;    
    velocityCost = 200 * velErr^2;       
    angularVelCost = 200 * angVelErr^2; 
    
    % Combine terminal costs
    c = positionCost + orientationCost + velocityCost + angularVelCost;
end
end