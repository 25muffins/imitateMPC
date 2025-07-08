Ts = .2;         % Sampling time
rb = 0.5;

% Input: 4 wheel velocities
nu = 4;           % [fr, fl, br, bl]
nx = 3;           % [x; y; theta]
ny = 3;

% Input matrix based on mecanum forward kinematics
J = 1/4 * [
    1,  1,  1,  1;
    1, -1, -1, 1;
   1/(2*rb), -1/(2*rb), 1/(2*rb), -1/(2*rb)
];

% Discrete-time A and B
A = eye(nx);
B = Ts * J;
C = eye(ny);
D = zeros(ny, nu);

load("PathXY.mat")

sys = ss(A,B,C,D, Ts);
%plant = setmpcsignals(sys, 'MV', [1 2]);
oldStatus = mpcverbosity('off');
clnup = onCleanup(@()mpcverbosity(oldStatus));

mpcObj = mpc(sys, Ts);
mpcObj.p = 50;
mpcObj.c = 20;

% Input constraints (wheel speed limits)
for i = 1:4
    mpcObj.MV(i).Min = -10;  % rad/s
    mpcObj.MV(i).Max = 10;
end

% Weights
mpcObj.Weights.MVRate = 0.1 * ones(1, nu);
mpcObj.Weights.OV = [100 100 1];  % [x y theta]
setEstimator(mpcObj, 'custom');
path=[x4', y4']; %make sure to invert

theta_path = atan2(diff(path(:,2)), diff(path(:,1))); %atan2 does all 4 quadrants
theta_path = [theta_path; theta_path(end)]; %rads, add one more to end
fullPath = [path, theta_path];

choppedPath = [xRef, yRef]
theta_choppedPath = atan2(diff(choppedPath(:,2)), diff(choppedPath(:,1))); %atan2 does all 4 quadrants
theta_choppedPath = [theta_choppedPath; theta_choppedPath(end)]; %rads, add one more to end
fullPath = [choppedPath, theta_choppedPath];
%repeats = 5;
%fullPathRepeated = repelem(fullPath, repeats, 1);  % Repeat each row 5 times


x0 = [0; 0; 0];  % initial state
trajectory = x0';
mpcState = mpcstate(mpcObj);

for k = 1:size(fullPath,1)
    
    yref = fullPath(k,:)';  % [x, y, theta]
    yref(3) = wrapToPi(yref(3));

     % Compute control input
    u = mpcmove(mpcObj, mpcState, x0, yref);
    
    % Simulate forward (linear model)
    first = A * x0;
    second =  J * u;
    theta = x0(3) + second(3);
    xvalue = (second(1) * sin(theta)) +  (second(2) * cos(theta));
    yvalue = (second(1) * cos(theta)) +  (second(2) * sin(theta));
    xnext = [xvalue;
             yvalue;
             theta];
    xnext = xnext  + first;
    x0 = xnext
    trajectory = [trajectory; x0'];
    

end


figure;
plot(path(:,1), path(:,2), 'k--', 'LineWidth', 1.5); hold on;
plot(trajectory(:,1), trajectory(:,2), 'b-', 'LineWidth', 2);
plot(xRef, yRef, 'ro');
xlabel('X'); ylabel('Y'); title('MPC Path Following');
plot(xRef, yRef);
legend('Reference Path', 'Robot Trajectory', "Points");
axis equal;

%not used currently
function dist = distance(a, b)
        dist = sqrt((a(1)-b(1))^2 + (a(2)-b(2))^2);
end