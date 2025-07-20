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
nx = 3;           % [x; y; theta; vx, vy, omega]
ny = 3;
N=15;
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
    planner.Stages(i).ParameterLength = 3;
end
planner.Stages(N+1).CostFcn = @terminalCost;
planner.Stages(N+1).ParameterLength = 3;
goal = [1;1;1];
simData.StageParameter = repmat(goal', N, 1);
simData = getSimulationData(planner, 'TerminalState');
simData.TerminalState = goal;

validateFcns(planner, zeros(nx,1), zeros(nu,1), simData);


nlobj = nlmpc(nx, nx, nu);  % states, outputs, inputs
nlobj.Ts = Ts;
nlobj.PredictionHorizon = 10;
nlobj.ControlHorizon = 5;
nlobj.Model.StateFcn = @(x, u) mecanumStateFcn(x, u);
nlobj.Model.IsContinuousTime = false;
nlobj.Model.OutputFcn = @(x, u) x;  % outputs = states
% Input constraints (wheel speeds)
for i = 1:4
    nlobj.MV(i).Min = -maxRev * wheelCircumference;
    nlobj.MV(i).Max = maxRev * wheelCircumference;
end

% Weights
nlobj.Weights.OutputVariables = [100 100 10];     % [x y theta]
nlobj.Weights.ManipulatedVariablesRate = [5 5 5 5];
nlobj.OV(3).Max = pi/4;
nlobj.OV(3).Min = -pi/4;

% Validate functions
validateFcns(nlobj, rand(nx,1), rand(nu,1));
mv0 = zeros(nu,1);
nloptions = nlmpcmoveopt;
%nloptions.Parameters = {Ts, r, l, w};


% Generate random data
Data = zeros(10,18);

x0 = [0, 0, 0];
u0 = [0,0,0,0];
goal1 = [50, -10, 1.5];
goal2 = [-20, 20, 1.5];
waypoints =  [goal1;  goal2];

clf
% Initialize
x = [0; 0; 0];
u = zeros(nu,1);
size(waypoints,1)
for wp = 1:size(waypoints,1)
    nextGoal = waypoints(wp,:)';
    fprintf("Navigating to waypoint %d: [%.1f, %.1f]\n", wp, nextGoal(1), nextGoal(2));
    for i = 2:N
        planner.Stages(i).CostFcn = @stageCost;
        planner.Stages(i).ParameterLength = 3;
    end
    simData = getSimulationData(planner , 'TerminalState');
    planner.Stages(N+1).CostFcn = @terminalCost;
    planner.Stages(N+1).ParameterLength = 3;
    simData.StageParameter = repmat(nextGoal, N, 1);
    
    simData.TerminalState = nextGoal;

    [u, ~, info] = nlmpcmove(planner, x, u, simData);
    simData.StageParameter;
    xTrackHistory = info.Xopt
    x = xTrackHistory(end, :);
    hold on
    plot(xTrackHistory(:, 1),xTrackHistory(:, 2))
end
plot(waypoints(:,1), waypoints(:,2), 'go');

% Create MAT file
save('MPCmadePath','Data')


function returnState = mecanumStateFcn(x, u)
    % Forward kinematics matrix (body velocities)
    J = 1/4 * [
        1,  -1, -1,  1; %x
        1, 1, 1, 1; %y
       1/(2*7.5), -1/(2*7.5), 1/(2*7.5), -1/(2*7.5)
    ];
    A = eye(3);
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
    xnext = xvel + first;
    xnext(3) = wrapToPi(xnext(3));
    returnState  = xnext;

end
function c = stageCost(stage,x,u, stageParam)
    goal = stageParam;
    posErr = norm(x(1:2) - goal(1:2));
    angErr = wrapToPi(x(3) - goal(3));
    c = .01 * sum(u.^2);
end
function c = terminalCost(stage, x,u, stageParam)
    goal = stageParam;
    posErr = norm(x(1:2) - goal(1:2));
    angErr = wrapToPi(x(3) - goal(3));
    c = 100 * posErr^2 + 10 * angErr^2;
end
end