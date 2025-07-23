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

% Generate MPC object.
Ts = 1;         % Sampling time
rb = 7.5; % track radius
maxRev = 300 / 60; %per second
wheelCircumference = 4.09 * pi;

nu = 4;           % [fr, fl, br, bl]
nx = 6;           % [x; y; theta; vx, vy, omega]
ny = 3;

%goal1 = [-72, -50, 0.5,  0, 0, 0];
%goal2 = [-30, 72, 2, 0, 0, 0];
%waypoints =  [goal1;  goal2];
%dist1 = sqrt(goal1(1)^2   + goal1(2^2));
%dist2 = sqrt((goal1(1) - goal2(1))^2  + (goal1(2) - goal2(2))^2);
%approximateN = round((dist1+dist2)/10);
%if mod(approximateN,  2) ==  1
%    approximateN = approximateN+1;
%end
N = 18;

planner = nlmpcMultistage(N, nx, nu);
planner.Ts = Ts;


planner.Model.StateFcn = @mecanumStateFcn;
planner.Model.IsContinuousTime = false;
planner.States(3).Min  = -pi;
planner.States(3).Max =  pi;
%for i = 1:4
 %   planner.MV(i).Min = -maxRev * wheelCircumference;
  %  planner.MV(i).Max = maxRev * wheelCircumference;
%end
%for i = 2:N
%    if i == (N/2  +1)
%        planner.Stages(i).CostFcn = @terminalCost;
%    else
%        planner.Stages(i).CostFcn = @stageCost;
%    end
%    planner.Stages(i).ParameterLength = 6;
%    
%end
%planner.Stages(N+1).CostFcn = @terminalCost;
%planner.Stages(N+1).ParameterLength = 6;
%goal = [1;1;1; 0; 0;  0];
%simData.StageParameter = repmat(goal', N, 1);
%simData = getSimulationData(planner, 'TerminalState');
%simData.TerminalState = goal;

%validateFcns(planner, zeros(nx,1), zeros(nu,1), simData);


clf
rng(40)
%clf
% initialize
d = zeros(1,34); %rows will automatically fill up
for ct= 1:5e0
    ct
    goal1 = [144*rand-72, 144*rand-72, 2*pi*rand - pi,  0, 0, 0]
    goal2 = [144*rand-72, 144*rand-72, 2*pi*rand - pi, 0, 0, 0]
    waypoints =  [goal1;  goal2];
    x = [0; 0; 0; 0; 0; 0];
    u = zeros(nu,1);
    for i = 1:4
        planner.MV(i).Min = -maxRev * wheelCircumference;
        planner.MV(i).Max = maxRev * wheelCircumference;
    end
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
    plot(1:N+1, xTrackHistory(1:N+1,6))
    
    a  = size(uHistory);
    plot(waypoints(:,1), waypoints(:,2), 'go');
    
    %x,  y, theta, xvel, yvel, thetavel,  
    % goal1x, goal1y, goal1theta,  goal1xvel, goal1yvel,  goal1thetavel, goal1dist, goal1angle
    % goal2x, goal2y, goal2theta,  goal2xvel, goal2yvel,  goal2thetavel, goal2dist, goal2angle
    % optimal u  (x4), past u (x4), goalswitch 
    % nextxVel nextyVel, nextthetaVel
    %total size = 34
    
    goalswitch = 0;
    for  i = 1:a(1)
        currentPos = xTrackHistory(i,:);
    
        dist1 = sqrt((currentPos(1)-goal1(1))^2 +  (currentPos(2)-goal1(2))^2);
        dist2 = sqrt((currentPos(1)-goal2(1))^2 +  (currentPos(2)-goal2(2))^2);
        angle1 =  atan2(goal1(2)  -  currentPos(2), goal1(1) - currentPos(1));
        angle2 =  atan2(goal2(2)  -  currentPos(2), goal2(1) - currentPos(1));
        
        g1 = [goal1,  dist1, angle1];
        g2 = [goal2,  dist2, angle2];
        optimalU  = uHistory(i,:);
    
        if(i==1) lastU =  [0,0,0,0];
        else  lastU =  uHistory(i-1,:);
        end
    
        if(i >= (a(1)/2 + 1))
            goalswitch = 1;
        end

        if(i==a(1)) nextVels = [0,0,0];
        else nextVels = xTrackHistory(i+1, 4:6);
        end
        
        d((ct-1) * a(1) + i,:) =  [currentPos(:)',  g1(:)', g2(:)', optimalU(:)', lastU(:)', goalswitch, nextVels];
    end
end

% Create MAT file
save('v1','d')


function returnState = mecanumStateFcn(x, u)
    % Forward kinematics matrix (body velocities)
    J = 1/4 * [
        1/2,  -1/2, -1/2,  1/2; %x
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
    xvalue = (second(1) * cos(theta)) +  (second(2) * sin(theta));
    yvalue = (second(1) * sin(theta)) +  (second(2) * cos(theta));
    xvel = [xvalue;
             yvalue;
             second(3)];
    xnext = [xvalue + first(1);
             yvalue + first(2);
             theta];
    xnext(3) = wrapToPi(xnext(3));
    returnState  = [xnext; xvel];

end
function c = stageCost(stage,x,u, stageParam)
    goal = stageParam;
    posErr = norm(x(1:2) - goal(1:2));
    angErr = abs(wrapToPi(x(3) - goal(3)));
    velErr = norm(x(4:5)) - norm(goal(4:5)); 
    angVelErr = abs(x(6)  - goal(6)); 
    

    distanceToGoal = posErr;

    posWeight = 100;
    angWeight = 100000;
    trackingCost = posWeight * posErr^2 + angWeight * angErr^2;

    velCost = 120;
    velocityCost = velCost * velErr^2 + velCost * angVelErr^2;
    controlCost = 0.1 * sum(u.^2);
    
    % Combine costs
    c = trackingCost/5 + velocityCost;
end
function c = terminalCost(stage, x,u, stageParam)
    goal = stageParam;
    posErr = norm(x(1:2) - goal(1:2));
    angErr = abs(wrapToPi(x(3) - goal(3)));
    velErr = norm(x(4:5)) - norm(goal(4:5)); 
    angVelErr = abs(x(6)  - goal(6)); 
    
    positionCost = 800 * posErr^2;      
    orientationCost = 30000 * angErr^2;    
    velocityCost = 120 * velErr^2;       
    angularVelCost = 120 * angVelErr^2; 
    
    % Combine terminal costs
    c = 2*(positionCost + orientationCost + velocityCost + angularVelCost);
end
end