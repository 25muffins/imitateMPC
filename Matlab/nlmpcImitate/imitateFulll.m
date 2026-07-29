%imitateMPC Matlab example code



%f = load('fasterStrafing.mat');
%f2 = load('fasterStrafing2.mat');
%f3 = load('fasterStrafing3.mat');

%mmp = load('MPCmadePath.mat');
%wvels = load('MPCmadePathWVels.mat');
%wvels2 = load('MPCmadePathWVels2.mat');
%da = load('MPCDistAngle1.mat');
v21 = load('v21cossinwithextrasauce.mat');
v22 = load('v22cossinwithextrasauce.mat');



%DataF = f.Data(~all(f.Data==0, 2), :);
%DataF2 = f2.Data(~all(f2.Data==0, 2), :);
%DataF3 = f3.Data(~all(f3.Data==0, 2), :);


%[sys,Vx] = createModelForMPCImLKA;
%[mpcobj,initialState] = createMPCobjImLKA(sys);
data = [v21.d; v22.d];
size(data)

%normalize
%divisors =  [144, 144, 3.1415, 64.2455, 64.2455, 64.2455, 64.2455, 144, 3.1415, 3.1415, 144, 3.1415, 3.1415, 64.2455, 64.2455, 64.2455, 64.2455, 1];
divisors =  [72, 72, 3.1415, 30, 30, 3.1415,...  %current (1-6)
    72, 72, 3.1415, 30, 30, 3.1415, 144, 3.1415,...  %goal1 (7-14)
    72, 72, 3.1415, 30, 30, 3.1415, 144, 3.1415,...  %goal2 (15-22)
    30, 30, 3.1415,... %optimalU (23-25)
    30, 30, 3.1415... %pastU (26-28)
    1,...%goalSwitch (29)
    30, 30, 3.1415, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];...%vels (30-42)];  

%for column = 1:38
%    data(:,column) = (data(:,column) - mean(data(:,column))) /std(data(:,column));
%end


data = data./divisors;

totalRows = size(data,1);
validationSplitPercent = 0.2;
numValidationDataRows = floor(validationSplitPercent*totalRows);

testSplitPercent = 0.05;
numTestDataRows = floor(testSplitPercent*totalRows);

%pick random validation + test
randomIdx = randperm(totalRows,numValidationDataRows+numTestDataRows);
randomData = data(randomIdx,:);
validationData = randomData(1:numValidationDataRows,:);
testData = randomData(numValidationDataRows + 1:end,:);

%train data is everything else
trainDataIdx = setdiff(1:totalRows,randomIdx);
trainData = data(trainDataIdx,:);

numTrainDataRows = size(trainData,1);
%shuffle train data (for some reason)
shuffleIdx = randperm(numTrainDataRows);
shuffledTrainData = trainData(shuffleIdx,:);

%TODO // FIX
numObs = 13;
numActions = 3;
%x,  y, theta, xvel, yvel, thetavel,  1-6
    % goal1x, goal1y, goal1theta,  goal1xvel, goal1yvel,  goal1thetavel,
    % goal1dist, goal1angle 7-14
    % goal2x, goal2y, goal2theta,  goal2xvel, goal2yvel,  goal2thetavel,
    % goal2dist, goal2angle 15-22
    % optimal u  (x3), past u (x3), goalswitch  23-29
    % nextxVel nextyVel, nextthetaVel 31,32,33
    % current sin theta, cos theta 33,34
    % goal 1 sin theta, cos theta, 35, 36
    % goal 2 sin theta, cos theta, 37, 38
    
%TODO // FIX
%trainInput = shuffledTrainData(:,[1:3 8:9 11:12 18]);
%trainOutput = shuffledTrainData(:,14:17);
%validationInput = validationData(:,[1:3 8:9 11:12 18]);
%validationOutput = validationData(:,14:17);
trainInput = shuffledTrainData(:,[1:2 33:34 7:8 35:36 15:16 37:38 29]);
trainOutput = shuffledTrainData(:,23:25);
validationInput = validationData(:,[1:2 33:34 7:8 35:36 15:16 37:38 29]);
validationOutput = validationData(:,23:25);

%[1:2 33:34 13 39:40 21 41:42 29]
validationCellArray = {validationInput,validationOutput};

%testDataInput = testData(:,[1:3 8:9 11:12 18]);
%testDataOutput = testData(:,14:17);
testDataInput = testData(:,[1:2 33:34 7:8 35:36 15:16 37:38 29]);
testDataOutput = testData(:,23:25);

rng(3);
imitateMPCLayers = [
    featureInputLayer(numObs) 
    %layerNormalizationLayer
    fullyConnectedLayer(450)
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(300)
    reluLayer
    fullyConnectedLayer(200)
    layerNormalizationLayer
    reluLayer
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(45)
    reluLayer
    fullyConnectedLayer(numActions)
    tanhLayer
    %scalingLayer(Scale=64.2456)
];

%plot(dlnetwork(imitateMPCLayers))

options = trainingOptions("adam", ...
    'L2Regularization', 1e-4,...
    LearnRateSchedule= 'piecewise',...
    LearnRateDropFactor= 0.5,...
    LearnRateDropPeriod= 50,...
    Verbose=true, ...
    Plots="training-progress", ...
    Metrics="mae", ...
    Shuffle="every-epoch", ...
    MaxEpochs=120, ...
    MiniBatchSize=512, ...
    ValidationData=validationCellArray, ...
    InitialLearnRate=1e-3, ...
    GradientThresholdMethod="absolute-value", ...
    ExecutionEnvironment="cpu", ...
    GradientThreshold=10, ...
    Epsilon=1e-8);

imitateMPCNetwork = trainnet( ...
    trainInput, ...
    trainOutput, ...
    imitateMPCLayers, ...
    @customLossWeightedMAE, ...
    options);

save("testN", "imitateMPCNetwork", "testDataInput","testDataOutput")

function loss = customLossWeightedMAE(Y, T)
    xVelErr = mean(abs(Y(1,:) - T(1,:)), 'all');
    yVelErr = mean(abs(Y(2,:) - T(2,:)), 'all');
    thetaVelErr = mean(abs(Y(3,:) - T(3,:)), 'all');

    % Weighted MAE
    loss = 1 * xVelErr + 1 * yVelErr + 20 * thetaVelErr;  % emphasize theta velocity
end