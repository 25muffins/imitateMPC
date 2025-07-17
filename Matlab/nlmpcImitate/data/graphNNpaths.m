load('FullNetwork.mat')
load("onestepNNtest.mat")
Ts = 1;       
rb = 7.5;
nx = 3;    


Ypredict = predict(imitateMPCNetwork, testDataInput);
rmse(testDataOutput, Ypredict)
mean(abs(Ypredict - testDataOutput))

testData = zeros(1e3,10); %start, goal1, goal2, nextTest, nextReal
%for ct = 1:1
 %   x0 = [testDataInput(ct,1) testDataInput(ct,2) testDataInput(ct,3)];
  %  uReal = testDataOutput(ct,:);
   % uTest = Ypredict(ct,:);
   % xReal = mecanumStateFcn(x0', uReal', Ts, rb, nx);
   % xTest = mecanumStateFcn(x0', uTest', Ts, rb, nx);
   % testData(ct,:) = [testDataInput(ct,1) testDataInput(ct,2) testDataInput(ct,4) testDataInput(ct,5) testDataInput(ct,6) testDataInput(ct,7) xTest(1) xTest(2) xReal(1) xReal(2)];
   
%end

number = 7;
list = zeros(10,3);
Data(number,:)
x0 = [Data(number,1), Data(number,2), Data(number,3)];
u = [0;0;0;0];
for g = 1:30
    inputData = [x0(1), x0(2), x0(3), Data(number,8), Data(number,9), Data(number,11), Data(number,12), u(1), u(2), u(3), u(4)];
    inputData
    Ypredict = predict(imitateMPCNetwork, inputData);
    u = Ypredict';
    x0 = mecanumStateFcn(x0', Ypredict', Ts, rb, nx)';
    x0(3) = wrapToPi(x0(3));
    x0;
    list(g,:) = x0';
end

%plotTestData(1, testData)
plotMPC(number,trajectories, supposedTrajectory)

plot(list(:,1), list(:,2), 'go')

function plotTestData(ct, testData)
    clf
    hold on
    waypoints = [testData(ct, 1) testData(ct, 2)
             testData(ct, 3),testData(ct, 4)
             testData(ct, 5), testData(ct,6)]
    plot(waypoints(:,1), waypoints(:,2))
    plot(testData(ct,1), testData(ct,2), 'go')
    plot(testData(ct,7), testData(ct,8), 'ro')
    plot(testData(ct,9), testData(ct,10), 'go')
end


function xnext = mecanumStateFcn(x, u, Ts, rb, nx)
    % Forward kinematics matrix (body velocities)
    J = 1/4 * [% [fr, fl, br, bl]
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
    xnext = [xvalue;
             yvalue;
             theta];
    xnext = xnext  + first;
end

function plotMPC(ct, trajectories, supposedTrajectory)
    clf
    hold on
    for i = 1:10
        plot(trajectories(ct*10-10+i, 1), trajectories(ct*10-10 + i,2),  'ro')
    end    
    supposedX =nonzeros(supposedTrajectory(ct*2-1,:))';
    supposedY =nonzeros(supposedTrajectory(ct*2,:))';
    plot(supposedX, supposedY)

end