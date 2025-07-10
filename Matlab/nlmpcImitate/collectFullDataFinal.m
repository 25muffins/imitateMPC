function collectDataFinal()
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

% Input: 4 wheel velocities
nu = 4;           % [fr, fl, br, bl]
nx = 3;           % [x; y; theta]
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
nlobj.Weights.OutputVariables = [100 100 10];     % [x y theta]
nlobj.Weights.ManipulatedVariablesRate = [1 1 1 1];
nlobj.OV(3).Max = pi/4;
nlobj.OV(3).Min = -pi/4;

% Validate functions
validateFcns(nlobj, rand(nx,1), rand(nu,1));
mv0 = zeros(nu,1);
nloptions = nlmpcmoveopt;
%nloptions.Parameters = {Ts, r, l, w};


% Generate random data
Data = zeros(1e4,17);

for ct = 1:1e4
    [x0, u0, goal1, goal2, ref] = randomDataNLMPC;
    u0 = [0; 0; 0; 0];
    a = size(ref(:,1));
    for g = 1:a
        [u, ~] = nlmpcmove(nlobj, x0, u0, ref(i+1:a(1), :), [], nloptions);
        u
        x0 = mecanumStateFcn(x0, u, Ts, rb, nx);
        % x0 (3), u0(4), goal1(3), goal2(3), u(4)
        Data(ct,:) = [x0(:)', u0(:)', goal1(:)', goal2(:)', u(:)'];
    end  
    ct
end


% Create MAT file
save('nlmpcFullData','Data')


function xnext = mecanumStateFcn(x, u, Ts, rb, nx)
    % Forward kinematics matrix (body velocities)
    J = 1/4 * [
        1,  1,  1,  1;
        1, -1, -1, 1;
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

end