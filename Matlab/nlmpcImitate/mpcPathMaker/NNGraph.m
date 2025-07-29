load("v22Network.mat") %v22 has better angles, really good has better path following, take v22
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

nu = 3;           % [fr, fl, br, bl]
nx = 6;           % [x; y; theta; vx, vy, omega]
ny = 3;

N = 18;

planner = nlmpcMultistage(N, nx, nu);
planner.Ts = Ts;


planner.Model.StateFcn = @mecanumStateFcn;
planner.Model.IsContinuousTime = false;
planner.States(3).Min  = -pi;
planner.States(3).Min  = -4*pi;
planner.States(3).Max =  4*pi;


clf
for ct= 1:1e0
    ct
    x = [0; 0; 0; 0; 0; 0];
    goal1 = [-72,  -72, 3,0,0 0];
    goal2 = [60, -10, -1, 0, 0, 0];

    goal1Candidates = [goal1(3) - 2*pi, goal1(3) + 2*pi, goal1(3), goal1(3) - 4*pi, goal1(3) + 4*pi];
    bc1 = findBestCandidate(goal1Candidates, x(3));
    goal2Candidates = [goal2(3) - 2*pi, goal2(3) + 2*pi, goal2(3), goal2(3) - 4*pi, goal2(3) + 4*pi];
    bc2 = findBestCandidate(goal2Candidates, bc1);
    
    goal1(3) = bc1;
    goal2(3) = bc2;

    waypoints =  [goal1;  goal2];
    u = zeros(nu,1);

    planner.MV(1).Min = -30;
    planner.MV(1).Max = 30;
    planner.MV(2).Min = -30;
    planner.MV(2).Max = 30;
    planner.MV(3).Min = -3.13;
    planner.MV(3).Max = 3.13;

    midGoal = waypoints(1,:)';
    finalGoal = waypoints(2,:)';
    for i = 2:N
        if i == (N/2+1)
            planner.Stages(i).CostFcn = @terminalCost;
        else
            planner.Stages(i).CostFcn = @stageCost;
        end
        planner.Stages(i).ParameterLength = 6;
    end
    simData = getSimulationData(planner , 'TerminalState');
    planner.Stages(N+1).CostFcn = @terminalCost;
    planner.Stages(N+1).ParameterLength = 6;
    
    
    sp =  [repmat(midGoal, N/2, 1); repmat(finalGoal, N/2, 1)];
    simData.StageParameter = sp;
    simData.TerminalState = finalGoal;
    [u, ~, info] = nlmpcmove(planner, x, u, simData);
    xTrackHistory = info.Xopt;
    uHistory  = info.MVopt;
    hold on
    plot(xTrackHistory(:, 1),xTrackHistory(:, 2))
    plot(xTrackHistory(:, 1),xTrackHistory(:, 2), 'ro')
    plot(1:N+1, xTrackHistory(1:N+1,3))
    plot(1:N+1, wrapToPi(xTrackHistory(1:N+1,3)))
    plot(1:N+1, xTrackHistory(1:N+1,6))
    
    a  = size(uHistory);
    plot(waypoints(:,1), waypoints(:,2), 'go');
    
    %x,  y, theta, xvel, yvel, thetavel,  
    % goal1x, goal1y, goal1theta,  goal1xvel, goal1yvel,  goal1thetavel, goal1dist, goal1angle
    % goal2x, goal2y, goal2theta,  goal2xvel, goal2yvel,  goal2thetavel, goal2dist, goal2angle
    % optimal u  (x4), past u (x4), goalswitch 
    %total size = 31
    
    %divisors =  [72, 72, 3.14,...
        %72, 72, 3.14,...
        %72, 72, 3.14,...
        %1];
    divisors =  [72, 72, 1, 1,...
        72, 72, 1, 1,...
        72, 72, 1, 1,...
        1];
    x0 = [0;0;0;0;0;0];
    goalswitch = 0;
    testData = zeros(1,10);
    for  i = 1:30
        plot(x0(1), x0(2),'k>')
        plot(i, x0(3), 'go')
        
        dist1 = sqrt((x0(1)-goal1(1))^2 +  (x0(2)-goal1(2))^2);
        dist2 = sqrt((x0(1)-goal2(1))^2 +  (x0(2)-goal2(2))^2);
        angle1 =  atan2(goal1(2)  -  x0(2), goal1(1) - x0(1));
        angle2 =  atan2(goal2(2)  -  x0(2), goal2(1) - x0(1));
        if(dist1 <= 2 && goalswitch~=1)
            goalswitch  = 1;
        end

        %inputData = [x0(1:3)', goal1(1:3), goal2(1:3), goalswitch];
        inputData = [x0(1:2)', sin(x0(3)), cos(x0(3)), goal1(1:2), sin(goal1(3)), cos(goal1(3)), goal2(1:2), sin(goal2(3)), cos(goal2(3)) goalswitch];

        inputData = inputData ./ divisors;
        Ypredict = predict(imitateMPCNetwork, inputData);
        u = Ypredict';
        u = [u(1) * 30, u(2) * 30, u(3) * 3.14];
        %u = u*64.2455;
        x0 = velStateFcn(x0, u);
        %x0 = mecanumStateFcn(x0, u);
        testData(i,:) =  [x0(:)', u(:)', goalswitch];
        
    end
end

% Create MAT file
save('testData','testData')


function returnState = mecanumStateFcn(x, u)  %u is xvel, yvel, thetavel (relative to  body)
    % Forward kinematics matrix (body velocities)
    J = [1,0,0;
         0,1,0;
         0,0,1];
    A = eye(6);
    B = 1 * J;
    first = A * x;
    second =  B  * u;
    theta = x(3) + second(3);
    xvalue = (second(1) * cos(x(3))) -  (second(2) * sin(x(3))); %these  kinematics are flipped
    yvalue = (second(1) * sin(x(3))) +  (second(2) * cos(x(3)));
    xvel = [xvalue;
             yvalue;
             second(3)];
    xnext = [xvalue + first(1);
             yvalue + first(2);
             theta];
    %xnext(3) = wrapToPi(xnext(3));
    %xvel(3) = wrapToPi(xvel(3));
    returnState  = [xnext; xvel];

end
function returnState = velStateFcn(x, u)

    xvel = [u(1);
             u(2);
             u(3)];
    xnext = [x(1) + xvel(1);
             x(2) + xvel(2);
             x(3) +  xvel(3)];
    xnext(3) = wrapToPi(xnext(3));
    returnState  = [xnext; xvel];
end

function c = stageCost(stage,x,u, stageParam)
    goal = stageParam;
    posErr = norm(x(1:2) - goal(1:2));
    angErr = x(3) - goal(3);
    velErr = norm(x(4:5)) - norm(goal(4:5)); 
    angVelErr = abs(x(6)  - goal(6)); 
    

    distanceToGoal = posErr;

    posWeight = 150;
    angWeight = 3000;
    trackingCost = posWeight * posErr^2 + angWeight * angErr^2;
    strafeCost = u(2)^2 * 0;
    
    forwardReward = u(1)^2 * -0;

    velCost = 300;
    velocityCost = velCost * velErr^2 + 520 * angVelErr^2;
    controlCost = 0.1 * sum(u.^2);
    
    % Combine costs
    c = trackingCost/5 + velocityCost + strafeCost + forwardReward;
end
function c = terminalCost(stage, x,u, stageParam)
    goal = stageParam;
    posErr = norm(x(1:2) - goal(1:2));
    angErr = x(3) - goal(3);
    velErr = norm(x(4:5)) - norm(goal(4:5)); 
    angVelErr = abs(x(6)  - goal(6)); 
    
    positionCost = 20000 * posErr^2;      
    orientationCost = 152000 * angErr^2;    
    velocityCost = 1200 * velErr^2;       
    angularVelCost = 5200 * angVelErr^2; 
    
    % Combine terminal costs
    c = 2*(positionCost + orientationCost + velocityCost + angularVelCost);
end
function bestCandidate = findBestCandidate(candidates, startTheta)
    lowestDist = 100;
    bestCandidate = 0;
    for c = 1:5
        dist = abs(candidates(c) - startTheta);
        if dist < lowestDist
            bestCandidate = candidates(c);
            lowestDist = dist;
        end
    end
end