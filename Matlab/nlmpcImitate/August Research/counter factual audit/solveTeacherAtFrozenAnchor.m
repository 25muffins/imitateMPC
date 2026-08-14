function u = solveTeacherAtFrozenAnchor(anchor, chosenG2Pos, start)
% Solve the original MPC at a completely frozen anchor.
%
% INPUTS
%   anchor       - one element from makeAuditAnchors()
%   chosenG2Pos  - [x y] position of the intervened g2
%
% OUTPUT
%   u            - first MPC control [vx vy omega]

    % ============================================================
    % MPC configuration -- copied from SimpleMPCasPathmaker
    % ============================================================

    Ts = 1;

    nu = 3;
    nx = 6;
    N = 18;

    planner = nlmpcMultistage(N, nx, nu);

    planner.Ts = Ts;

    planner.Model.StateFcn = @mecanumStateFcn;
    planner.Model.IsContinuousTime = false;

    planner.States(3).Min = -6*pi;
    planner.States(3).Max =  6*pi;

    planner.MV(1).Min = -30;
    planner.MV(1).Max =  30;

    planner.MV(2).Min = -30;
    planner.MV(2).Max =  30;

    planner.MV(3).Min = -3.13;
    planner.MV(3).Max =  3.13;

    for i = 2:N
        planner.Stages(i).CostFcn = @stageCost;
        planner.Stages(i).ParameterLength = 6;
    end

    planner.Stages(N+1).CostFcn = @terminalCost2;
    planner.Stages(N+1).ParameterLength = 6;
    x = start.x;


    g1 = anchor.g1;


    g2 = anchor.g2Original;

    clipChosenPos = [clip(chosenG2Pos(1), -72, 72), clip(chosenG2Pos(2), -72, 72)];
    g2(1:2) = clipChosenPos(:);


    midGoal   = g1(1:6);
    finalGoal = g2(1:6);

    sp = [ ...
        repmat(midGoal,   N/2, 1);
        repmat(finalGoal, N/2, 1)
    ];


    simData = getSimulationData(planner, 'TerminalState');

    simData.StageParameter = sp;
    simData.TerminalState = finalGoal;


    u0 = zeros(nu,1);


    Optimization.CustomEqConFcn = ...
        @(stage,x,u,stageParam) periodicConstraint(x);


    [u,~,info] = nlmpcmove( ...
        planner, ...
        x, ...
        u0, ...
        simData);
     hold on
    xTrackHistory = info.Xopt;
    uHistory = info.MVopt;
    plot(xTrackHistory(:, 1),xTrackHistory(:, 2))
    plot(xTrackHistory(:, 1),xTrackHistory(:, 2), 'bo')
    %plot(1:N+1, xTrackHistory(1:N+1,3))

    u = uHistory;
end

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
