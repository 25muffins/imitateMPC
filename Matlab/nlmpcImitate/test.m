load('nlmpcDataTest')

% 1- x(x) 
% 2 - x(y) 
% 3 - x(theta)(
% 4 - u0(1)
% 5 - u0(2)
% 6 - u0(3)
% 7 - u0(4)
% 8 - goal1(x)
% 9 - goal1(y)
% 10 - goal1(theta)
% 11 - goal2(x)
% 12 - goal2(y)
% 13 - goal2(theta)
% 14 - u(1)
% 15 - u(2)
% 16 - u(3)
% 17 - u(4)

%plotIt(6, Data, trajectories, supposedTrajectory)

function plotIt(ct, Data, trajectories, supposedTrajectory)
    waypoints = [Data(ct, 1) Data(ct, 2)
             %trajectories(12, 1), trajectories(12,2)
             %trajectories(13, 1), trajectories(13,2)
             %trajectories(14, 1), trajectories(14,2)
             Data(ct, 8),Data(ct, 9)
             Data(ct, 11), Data(ct,12)];
    clf
    hold on
    plot(waypoints(:,1), waypoints(:,2))
    plot(trajectories(ct*3-2, 1), trajectories(ct*3-2,2),  'ro')
    %plot(trajectories(ct*3-1, 1), trajectories(ct*3-1,2),  'ro')
    %plot(trajectories(ct*3, 1), trajectories(ct*3,2),  'ro')
    %trajectories(1:10, :)
    
    supposedX = [Data(ct, 1) nonzeros(supposedTrajectory(ct*2-1,:))'];
    supposedY = [Data(ct, 2) nonzeros(supposedTrajectory(ct*2,:))'];
    plot(supposedX, supposedY)

end