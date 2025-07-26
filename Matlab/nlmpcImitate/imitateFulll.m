%imitateMPC Matlab example code



%f = load('fasterStrafing.mat');
%f2 = load('fasterStrafing2.mat');
%f3 = load('fasterStrafing3.mat');

%mmp = load('MPCmadePath.mat');
%wvels = load('MPCmadePathWVels.mat');
%wvels2 = load('MPCmadePathWVels2.mat');
%da = load('MPCDistAngle1.mat');
v7 = load('v7.mat');
v7v2 = load('v7v2.mat');

%DataF = f.Data(~all(f.Data==0, 2), :);
%DataF2 = f2.Data(~all(f2.Data==0, 2), :);
%DataF3 = f3.Data(~all(f3.Data==0, 2), :);


%[sys,Vx] = createModelForMPCImLKA;
%[mpcobj,initialState] = createMPCobjImLKA(sys);
data = [v7.d; v7v2.d];
size(data)

%normalize
%divisors =  [144, 144, 3.1415, 64.2455, 64.2455, 64.2455, 64.2455, 144, 3.1415, 3.1415, 144, 3.1415, 3.1415, 64.2455, 64.2455, 64.2455, 64.2455, 1];
divisors =  [72, 72, 3.1415, 30, 30, 3.1415,...  %current (1-6)
    72, 72, 3.1415, 30, 30, 3.1415, 144, 3.1415,...  %goal1 (7-14)
    72, 72, 3.1415, 30, 30, 3.1415, 144, 3.1415,...  %goal2 (15-22)
    30, 30, 3.1415,... %optimalU (23-26)
    30, 30, 3.1415... %pastU (27-30)
    1,...%goalSwitch (31)
    30, 30, 3.1415];  %vels (32-34)

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
numObs = 10;
numActions = 3;

%TODO // FIX
%trainInput = shuffledTrainData(:,[1:3 8:9 11:12 18]);
%trainOutput = shuffledTrainData(:,14:17);
%validationInput = validationData(:,[1:3 8:9 11:12 18]);
%validationOutput = validationData(:,14:17);
trainInput = shuffledTrainData(:,[1:3 7:9 15:17 29]);
trainOutput = shuffledTrainData(:,30:32);
validationInput = validationData(:,[1:3 7:9 15:17 29]);
validationOutput = validationData(:,30:32);

validationCellArray = {validationInput,validationOutput};

%testDataInput = testData(:,[1:3 8:9 11:12 18]);
%testDataOutput = testData(:,14:17);
testDataInput = testData(:,[1:6 7:9 15:17 29]);
testDataOutput = testData(:,30:32);

rng(0);
imitateMPCLayers = [
    featureInputLayer(numObs) 
    fullyConnectedLayer(450)
    reluLayer
    %dropoutLayer(0.2)
    fullyConnectedLayer(400)
    reluLayer
    fullyConnectedLayer(300)
    reluLayer
    fullyConnectedLayer(200)
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
    MaxEpochs=100, ...
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
    "mae", ...
    options);

save("v7Network", "imitateMPCNetwork", "testDataInput","testDataOutput")