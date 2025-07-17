%imitateMPC Matlab example code

d1 = load('full1.mat');
d2 = load('full2.mat');
d3 = load('full3.mat');


Data1 = d1.Data(~all(d1.Data == 0, 2), :);
Data2 = d2.Data(~all(d2.Data == 0, 2), :);
Data3 = d3.Data(~all(d3.Data == 0, 2), :);


%[sys,Vx] = createModelForMPCImLKA;
%[mpcobj,initialState] = createMPCobjImLKA(sys);
data = [Data1; Data2; Data3];  %38k
size(data)
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
numObs = 11; % current x  y theta, goal1x, goal1y, goal2x, goal2y, u0 (x4)
numActions = 4;

%TODO // FIX
trainInput = shuffledTrainData(:,[1:3 8:9 11:12 4:7]);
trainOutput = shuffledTrainData(:,14:17);
validationInput = validationData(:,[1:3 8:9 11:12 4:7]);
validationOutput = validationData(:,14:17);

validationCellArray = {validationInput,validationOutput};

testDataInput = testData(:,[1:3 8:9 11:12 4:7]);
testDataOutput = testData(:,14:17);

rng(0);
imitateMPCLayers = [
    featureInputLayer(numObs) 
    fullyConnectedLayer(450)
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(450)
    reluLayer
    fullyConnectedLayer(450)
    reluLayer
    fullyConnectedLayer(450)
    reluLayer

    fullyConnectedLayer(numActions)
    tanhLayer
    scalingLayer(Scale=64.2456)
];

%plot(dlnetwork(imitateMPCLayers))

options = trainingOptions("adam", ...
    'L2Regularization', 1e-4,...
    Verbose=true, ...
    Plots="training-progress", ...
    Metrics="mae", ...
    Shuffle="every-epoch", ...
    MaxEpochs=320, ...
    MiniBatchSize=512, ...
    ValidationData=validationCellArray, ...
    InitialLearnRate=1e-4, ...
    GradientThresholdMethod="absolute-value", ...
    ExecutionEnvironment="cpu", ...
    GradientThreshold=10, ...
    Epsilon=1e-8);

imitateMPCNetwork = trainnet( ...
    trainInput, ...
    trainOutput, ...
    imitateMPCLayers, ...
    "mse", ...
    options);

save("TestNetwork", "imitateMPCNetwork", "testDataInput","testDataOutput")