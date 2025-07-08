load('network')
Ts = 1;       
rb = 7.5;
nx = 3;    


Ypredict = predict(imitateMPCNetwork, testDataInput);
rmse(testDataOutput, Ypredict)
mean(abs(Ypredict - testDataOutput))

testData = zeros(1e3,10); %start, goal1, goal2, nextTest, nextReal
for ct = 1:10
    x0 = [testDataInput(ct,1) testDataInput(ct,2) testDataInput(ct,3)];
    uReal = testDataOutput(ct,:);
    uTest = Ypredict(ct,:);
    xReal = mecanumStateFcn(x0', uReal', Ts, rb, nx);
    xTest = mecanumStateFcn(x0', uTest', Ts, rb, nx);
    testData(ct,:) = [testDataInput(ct,1) testDataInput(ct,2) testDataInput(ct,4) testDataInput(ct,5) testDataInput(ct,6) testDataInput(ct,7) xTest(1) xTest(2) xReal(1) xReal(2)];
   
end

plotTestData(1, testData)
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
