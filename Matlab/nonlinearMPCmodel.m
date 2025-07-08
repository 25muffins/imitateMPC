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

load("PathXY.mat")

path=[x4', y4']; %make sure to invert
theta_path = atan2(diff(path(:,2)), diff(path(:,1))); %atan2 does all 4 quadrants
theta_path = [theta_path; theta_path(end)]; %rads, add one more to end
fullPath = [path, theta_path];

choppedPath = [xRef, yRef];
theta_choppedPath = atan2(diff(choppedPath(:,2)), diff(choppedPath(:,1))); %atan2 does all 4 quadrants
theta_choppedPath = [theta_choppedPath; theta_choppedPath(end)]; %rads, add one more to end
%fullPath = [choppedPath, theta_choppedPath];
%repeats = 5;
%fullPathRepeated = repelem(fullPath, repeats, 1);  % Repeat each row 5 times


x0 = [0; 0; 0];  % initial state
trajectory = x0';

for k = 1:size(fullPath,1)
    yref = fullPath(k:size(fullPath,1),:);
    % Compute optimal control
    [u, ~] = nlmpcmove(nlobj, x0, mv0, yref, [], nloptions);
    % Update state using model
    x0 = mecanumStateFcn(x0, u, Ts, rb, nx);
    trajectory = [trajectory; x0'];
    u
    
end


figure;
plot(path(:,1), path(:,2), 'k--', 'LineWidth', 1.5); hold on;
plot(trajectory(:,1), trajectory(:,2), 'b-', 'LineWidth', 2);
%plot(x4, y4, '*');
plot(xRef, yRef, "ro")
xlabel('X'); ylabel('Y'); title('MPC Path Following');
plot(xRef, yRef);
plot(trajectory(:,1), trajectory(:,3))
legend('Reference Path', 'Robot Trajectory', "Points");
axis equal;

%not used currently
function dist = distance(a, b)
        dist = sqrt((a(1)-b(1))^2 + (a(2)-b(2))^2);
end


function xnext = mecanumStateFcn(x, u, Ts, rb, nx)
     % x = [x; y; theta]
     persistent counter;
     if isempty(counter) % Initialize counter on the first call
        counter = 0;
        end
    

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
   % counter = counter + 1
end




