function SimpleMPCasPathmaker()
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

nu = 3;           % [fr, fl, br, bl]
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
planner.States(3).Min  = -6*pi;
planner.States(3).Max =  6*pi;


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
rng(998713)
%clf
% initialize
d = zeros(1,110); %rows will automatically fill up
for ct= 1:2000
    ct
    x = [0; 0; 0; 0; 0; 0];
    goal1 = [144*rand-72, 144*rand-72, 2*pi*rand - pi,  0, 0, 0];
    goal2 = [144*rand-72, 144*rand-72, 2*pi*rand - pi, 0, 0, 0];
    %goal1 = [10, 30, -pi,  0, 0, 0];
    %goal2 = [20, -70, pi, 0, 0, 0];
  
    %goal1Candidates = [goal1(3) - 2*pi, goal1(3) + 2*pi, goal1(3), goal1(3) - 4*pi, goal1(3) + 4*pi];
    bc1 = goal1(3);%findBestCandidate(goal1Candidates, x(3));
    %goal2Candidates = [goal2(3) - 2*pi, goal2(3) + 2*pi, goal2(3), goal2(3) - 4*pi, goal2(3) + 4*pi];
    bc2 = goal2(3);%findBestCandidate(goal2Candidates, bc1);
    
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
            planner.Stages(i).CostFcn = @stageCost;
        else
            planner.Stages(i).CostFcn = @stageCost;
        end
        planner.Stages(i).ParameterLength = 6;
    end
    simData = getSimulationData(planner , 'TerminalState');
    planner.Stages(N+1).CostFcn = @terminalCost2;
    planner.Stages(N+1).ParameterLength = 6;
    
    
    sp =  [repmat(midGoal, N/2, 1); repmat(finalGoal, N/2, 1)];
    simData.StageParameter = sp;
    simData.TerminalState = finalGoal;
    Optimization.CustomEqConFcn = @(stage,x,u,stageParam) periodicConstraint(x);
    [u, ~, info] = nlmpcmove(planner, x, u, simData);
    xTrackHistory = info.Xopt;
    uHistory  = info.MVopt;
    hold on
    plot(xTrackHistory(:, 1),xTrackHistory(:, 2))
    plot(xTrackHistory(:, 1),xTrackHistory(:, 2), 'ro')
    plot(1:N+1, xTrackHistory(1:N+1,3))
    plot(N/2 + 1, goal1(3), 'bo')
    plot(N+1, goal2(3), 'bo')
    
    a  = size(uHistory);
    plot(waypoints(:,1), waypoints(:,2), 'go');
    
    %x,  y, theta, xvel, yvel, thetavel,  1-6
    % goal1x, goal1y, goal1theta,  goal1xvel, goal1yvel,  goal1thetavel,
    % goal1dist, goal1angle 7-14
    % goal2x, goal2y, goal2theta,  goal2xvel, goal2yvel,  goal2thetavel,
    % goal2dist, goal2angle 15-22
    % optimal u  (x3), past u (x3), goalswitch  23-29
    % nextxVel nextyVel, nextthetaVel 30,31,32
    % current sin theta, cos theta 33,34
    % goal 1 sin theta, cos theta, 35, 36
    % goal 2 sin theta, cos theta, 37, 38
    % history: [x,y, theta, sintheta, costheta, xvel, yvel, thetavel,
    % goalswitch] x 8 = 72
    %total size = 38 + 72
    
    goalswitch = 0;
    historyMatrix = zeros(1,72);
    lastHistoryRow = zeros(1,9);
    for  i = 1:a(1)

        
        currentPos = xTrackHistory(i,:);
        diff = [currentPos(1:3)-lastHistoryRow(1:3), sin(currentPos(3)),cos(currentPos(3)), lastHistoryRow(4:7)];
        historyMatrix = [historyMatrix(10:end), diff];

        dist1 = sqrt((currentPos(1)-goal1(1))^2 +  (currentPos(2)-goal1(2))^2);
        dist2 = sqrt((currentPos(1)-goal2(1))^2 +  (currentPos(2)-goal2(2))^2);
        angle1 =  atan2(goal1(2)  -  currentPos(2), goal1(1) - currentPos(1));
        angle2 =  atan2(goal2(2)  -  currentPos(2), goal2(1) - currentPos(1));
        
        g1 = [goal1,  dist1, angle1];
        g2 = [goal2,  dist2, angle2];
        optimalU  = uHistory(i,:);
    
        if(i==1) lastU =  [0,0,0];
        else  lastU =  uHistory(i-1,:);
        end
        
        %if ((goalswitch ~= 1) & (dist1 <= 6))
        %   goalswitch = 1;
        %end
        if(i >= (a(1)/2))
           goalswitch = 1;
        end

        if(i==a(1)) nextVels = [0,0,0];
        else nextVels = xTrackHistory(i+1, 4:6);
        end

        currentPos(3) = wrapToPi(currentPos(3));
        g1(3) = wrapToPi(g1(3));
        g2(3) = wrapToPi(g2(3));
        sinEmbeddings = [sin(currentPos(3)); cos(currentPos(3)); sin(goal1(3)); cos(goal1(3)); sin(goal2(3)); cos(goal2(3))];
        lastHistoryRow = [currentPos(1:3), sinEmbeddings(1:2)', currentPos(4:6), goalswitch];

        d((ct-1) * a(1) + i,:) =  [currentPos(:)',  g1(:)', g2(:)', optimalU(:)', lastU(:)', goalswitch, nextVels, sinEmbeddings', historyMatrix];
    end
end
%plot(1:N+1, d(1:N+1,3))

% Create MAT file
save('GRUDiffV2','d')


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
function c = stageCost(stage,x,u, stageParam)
    goal = stageParam;
    posErr = norm(x(1:2) - goal(1:2));
    angErr = x(3) - goal(3);
    velErr = norm(x(4:5)) - norm(goal(4:5)); 
    angVelErr = abs(x(6)  - goal(6)); 
    

    distanceToGoal = posErr;

    posWeight = 25;
    angWeight = 60;
    trackingCost = posWeight * posErr^2 + angWeight * angErr^2;
    strafeCost = u(2)^2 * 0;
    
    forwardReward = u(1)^2 * 0;

    velCost = 13;
    velocityCost = velCost * velErr^2 + 15 * angVelErr^2;
    controlCost = 0.1 * sum(u.^2);
    
    % Combine costs
    c = trackingCost/5 + velocityCost + strafeCost + forwardReward;
end
function c = terminalCost1(stage, x,u, stageParam)
    goal = stageParam;
    posErr = norm(x(1:2) - goal(1:2));
    angErr = x(3) - goal(3);
    velErr = norm(x(4:5)) - norm(goal(4:5)); 
    angVelErr = abs(x(6)  - goal(6)); 
    
    positionCost = 2.5 * posErr^2;      
    orientationCost = 45 * angErr^2;    
    velocityCost = 1 * velErr^2;       
    angularVelCost = 1 * angVelErr^2; 
    
    % Combine terminal costs
    c = 5*(positionCost + orientationCost + velocityCost + angularVelCost);
end
function c = terminalCost2(stage, x,u, stageParam)
    goal = stageParam;
    posErr = norm(x(1:2) - goal(1:2));
    angErr = x(3) - goal(3);
    velErr = norm(x(4:5)) - norm(goal(4:5)); 
    angVelErr = abs(x(6)  - goal(6)); 
    
    positionCost = 5 * posErr^2;      
    orientationCost = 45 * angErr^2;    
    velocityCost = 1 * velErr^2;       
    angularVelCost = 1 * angVelErr^2; 
    
    % Combine terminal costs
    c = 5*(positionCost + orientationCost + velocityCost + angularVelCost);
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
end