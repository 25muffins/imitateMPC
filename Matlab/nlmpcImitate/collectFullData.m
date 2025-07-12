function collectFullData()
% Copyright 2019 The MathWorks, Inc.


% use a different seed such as rng('shuffle') to create differing data
rng(4)

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
nlobj.Weights.ManipulatedVariablesRate = [5 5 5 5];
nlobj.OV(3).Max = pi/4;
nlobj.OV(3).Min = -pi/4;

% Validate functions
validateFcns(nlobj, rand(nx,1), rand(nu,1));
mv0 = zeros(nu,1);
nloptions = nlmpcmoveopt;

Data = zeros(1e5,17);

trajLength = zeros(100, 1);
trajLength(1) = 1;
for ct = 1:5e2 %500 * 20 = 10,000 so it should take ~50 mins
    ct
   [x0, u0, goal1, goal2, ref] = randomDataNLMPC;
   %we dont actually need u0 randomized
   u0 = [0;0;0;0];
   if(ct>=2)
        trajLength(ct,1) = a(1) + trajLength(ct-1,1); %cumulative sum for later
        trajLength(ct,1)
   end
   a = size(ref(:,1));
   for i = 1:a(1)
        x0(3) = wrapToPi(x0(3));
        [u, ~] = nlmpcmove(nlobj, x0, u0, ref(i+1:a(1), :), [], nloptions);
        u
        Data(trajLength(ct,1)+i-1,:) = [x0(:)', u0(:)', goal1(:)', goal2(:)', u(:)'];
        u0 = u';
        x0 = mecanumStateFcn(x0, u, Ts, rb, nx);
   end
    
end


% Create MAT file
save('fullNLMPCtest2','Data', 'trajLength')


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

d = Data(~all(Data == 0, 2), :);
size(d)

end