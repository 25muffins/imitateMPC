function collectDataOneStep()
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

nlobj = nlmpc(nx, nx, nu);  % states, outputs, inputs
nlobj.Ts = Ts;
nlobj.PredictionHorizon = 10;
nlobj.ControlHorizon = 5;
nlobj.Model.StateFcn = @(x, u) mecanumStateFcn(x, u, Ts, rb, nx);
nlobj.Model.IsContinuousTime = false;
nlobj.Model.OutputFcn = @(x, u) x;  % outputs = states
% Input constraints (wheel speeds)
for i = 1:4
    nlobj.MV(i).Min = -maxRev * wheelCircumference;
    nlobj.MV(i).Max = maxRev * wheelCircumference;
end

% Weights
nlobj.Weights.OutputVariables = [100 100 10, 10, 10, 10];     % [x y theta]
nlobj.Weights.ManipulatedVariablesRate = [15 15 15 15];
nlobj.OV(3).Max = pi/4;
nlobj.OV(3).Min = -pi/4;

% Validate functions
validateFcns(nlobj, rand(nx,1), rand(nu,1));
mv0 = zeros(nu,1);
nloptions = nlmpcmoveopt;
%nloptions.Parameters = {Ts, r, l, w};


% Generate random data
Data = zeros(10,27);

x0 = [0, 0, 0, 0, 0, 0];
u0 = [0,0,0,0];
goal1 = [50, -10, 1.5, 0, 0, 0];
goal2 = [-20, 20, 1.5,  0, 0, 0];
waypoints =  [goal1; goal2];

waypoints(1:2, :)
currentStep = 1;
for ct = 1:30
    dist = sqrt((goal1(1) -  x0(1))^2 + (goal1(2) - x0(2))^2);
    if(dist <= 6 && currentStep~=2)
        currentStep = 2;
    end
    currentStep
    [u, ~] = nlmpcmove(nlobj, x0, u0, waypoints(currentStep:2, :), [], nloptions);
    u0 = u
    ct
    Data(ct,:) = [x0(:)', u0(:)', goal1(:)', goal2(:)', u(:)',  currentStep];
    x0 = mecanumStateFcn(x0', u, Ts, rb, nx)';
    
end
Data
clf
hold
plot(Data(:,1), Data(:,2));
plot(1:10, Data(1:10,3));
plot(goal1(1), goal1(2), 'ro');
plot(goal2(1), goal2(2), 'ro');

% Create MAT file
save('expandedData','Data')


function returnState = mecanumStateFcn(x, u, Ts, rb, nx)
    % Forward kinematics matrix (body velocities)
    J = 1/4 * [
        1,  -1, -1,  1; %x
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
    xvel = [xvalue;
             yvalue;
             theta];
    xnext = [xvalue + first(1);
             yvalue + first(2);
             theta + first(3)];
    returnState  = [xnext; xvel]; 

end

end