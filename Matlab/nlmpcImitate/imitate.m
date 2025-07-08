%imitateMPC Matlab example code

coolData = load('nlmpcDataFinal.mat');
coolData2 =load('nlmpcDataFinal2.mat');
coolData3 =load('nlmpcDataFinal3.mat');
coolData4 =load('nlmpcDataFinal3.mat');
%[sys,Vx] = createModelForMPCImLKA;
%[mpcobj,initialState] = createMPCobjImLKA(sys);
data = [coolData.Data; coolData2.Data; coolData3.Data; coolData4.Data];  %20k
size(data)
totalRows = size(data,1);
validationSplitPercent = 0.1;
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
numObs = 7; 
numActions = 4;

%TODO // FIX
trainInput = shuffledTrainData(:,[1:3 8:9 11:12]);
trainOutput = shuffledTrainData(:,14:17);
validationInput = validationData(:,[1:3 8:9 11:12]);
validationOutput = validationData(:,14:17);

validationCellArray = {validationInput,validationOutput};

testDataInput = testData(:,[1:3 8:9 11:12]);
testDataOutput = testData(:,14:17);

rng(0);
imitateMPCLayers = [
    featureInputLayer(numObs)    
    fullyConnectedLayer(45)
    reluLayer
    fullyConnectedLayer(45)
    reluLayer
    fullyConnectedLayer(45)
    reluLayer
    fullyConnectedLayer(45)
    reluLayer
    fullyConnectedLayer(45)
    reluLayer
    fullyConnectedLayer(45)
    reluLayer
    fullyConnectedLayer(45)
    reluLayer
    fullyConnectedLayer(45)
    reluLayer
    fullyConnectedLayer(45)
    reluLayer
    fullyConnectedLayer(numActions)
    tanhLayer
    scalingLayer(Scale=64.2456)
];

%plot(dlnetwork(imitateMPCLayers))

options = trainingOptions("adam", ...
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
    "mse", ...
    options);

save("network", "imitateMPCNetwork", "testDataInput","testDataOutput")