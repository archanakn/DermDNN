clear; clc;

%% Paths & basic setup
dataDir    = fullfile(pwd,'give folder path');              % <-- change if needed
imageSize  = [224 224];                          % small & fast for <800 images
rng(42);                                         % reproducibility

%% Read images (labels from subfolder names)
imds = imageDatastore(dataDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Stratified split: 70% train, 15% val, 15% test
[imdsTrain, imdsRest] = splitEachLabel(imds,0.80, 'randomized');
[imdsVal,   imdsTest] = splitEachLabel(imdsRest, 0.50, 'randomized');

classes = categories(imdsTrain.Labels);
numClasses = numel(classes);

%% Preprocess: build a 10-channel stack per image
% 3  = HSV (H,S,V)
% 4  = Gabor magnitudes (texture) @ 4 orientations
% 3  = Shape: Canny edges, distance transform, LoG magnitude
preprocData = load('preprocNet_new.mat');
netPreproc = preprocData.net;
imdsTrain.ReadFcn = @(filename) makeSample(filename, imageSize,netPreproc);
imdsVal.ReadFcn   = @(filename) makeSample(filename, imageSize,netPreproc);
imdsTest.ReadFcn  = @(filename) makeSample(filename, imageSize,netPreproc);
% imdsTrain.ReadFcn = @(filename) makeSample(filename, imageSize);
% imdsVal.ReadFcn   = @(filename) makeSample(filename, imageSize);
% imdsTest.ReadFcn  = @(filename) makeSample(filename, imageSize);

% Now create augmented datastores directly
augTrain = augmentedImageDatastore(imageSize, imdsTrain);
augVal   = augmentedImageDatastore(imageSize, imdsVal);
augTest  = augmentedImageDatastore(imageSize, imdsTest);
% Compute class weights (inverse-frequency) for imbalance
classWeights = computeClassWeights(imdsTrain.Labels, classes);
miniBatchSize = 32;
epochs        = 50;
%% Define the network (small CNN, 10-channel input)
layers = [
    imageInputLayer([imageSize 10], ...
        'Normalization','zscore', ...
        'Name','input')

    convolution2dLayer(3, 24, 'Padding','same', 'Name','conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    maxPooling2dLayer(2,'Stride',2,'Name','pool1')

    convolution2dLayer(3, 48, 'Padding','same', 'Name','conv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    maxPooling2dLayer(2,'Stride',2,'Name','pool2')

    convolution2dLayer(3, 96, 'Padding','same', 'Name','conv3')
    batchNormalizationLayer('Name','bn3')
    reluLayer('Name','relu3')

    globalAveragePooling2dLayer('Name','gap')
    dropoutLayer(0.3,'Name','drop')

    fullyConnectedLayer(numClasses,'Name','fc')
    softmaxLayer('Name','sm')

    % Imbalance-aware loss: class-weighted cross-entropy
   weightedClassificationLayer(classWeights,'ce')];

%% Training options
opts = trainingOptions('adam', ...
    'InitialLearnRate', 4e-4, ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', epochs, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', ceil(numel(imdsTrain.Files)/miniBatchSize), ...
    'ExecutionEnvironment', 'auto', ...
    'Verbose', true, ...
    'L2Regularization',1e-4,...
    'Plots', 'training-progress');

%% Train
net = trainNetwork(augTrain, layers, opts);

%% Evaluate (final accuracy printed at end)
[YPred,scores]= classify(net, augTest);
YTrue = imdsTest.Labels;
% acc   = mean(YPred == YTrue);
% 
% disp('--------------------------');
% disp(['Test Accuracy: ' num2str(acc*100, '%.2f') '%']);
confMat = confusionmat(YTrue, YPred);
disp('Confusion Matrix:');
disp(confMat);

% Extract TP, FP, FN, TN for binary classification
TP = confMat(1,1);
FN = confMat(1,2);
FP = confMat(2,1);
TN = confMat(2,2);

% Metrics
precision = TP / (TP + FP);
recall    = TP / (TP + FN);
f1        = 2 * (precision * recall) / (precision + recall);
accuracy  = mean(YPred == YTrue);

% Display
disp('--------------------------');
fprintf('Accuracy : %.2f %%\n', accuracy*100);
fprintf('Precision: %.2f %%\n', precision*100);
fprintf('Recall   : %.2f %%\n', recall*100);
fprintf('F1-score : %.2f %%\n', f1*100);
if isprop(net,'Layers')
    netClasses = net.Layers(end).Classes;
% Case 2: trainedNet is dlnetwork (custom training loop)
elseif isprop(net,'Classes')
    netClasses = net.Classes;
else
    error('Cannot determine network classes. Check trainedNet object type.');
end

% Example for binary classification
positiveClass = categories(YTrue); % take first or second class as positive
positiveClass = positiveClass{2};  % usually 'disease' or target class

% If you have scores from classify/predict
yScore = scores(:, strcmp(string(netClasses), string(positiveClass))); % probability for positive class
yTrueLogical = (YTrue == positiveClass);

% ROC and AUC
[fpr,tpr,~,auc] = perfcurve(yTrueLogical, yScore, true);
figure; plot(fpr,tpr,'LineWidth',2);
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title(sprintf('ROC Curve (AUC = %.3f)', auc)); grid on;

% PR curve
[recallPR, precisionPR, ~, ap] = perfcurve(yTrueLogical, yScore, true, 'xCrit','reca','yCrit','prec');
figure; plot(recallPR, precisionPR,'LineWidth',2);
xlabel('Recall'); ylabel('Precision');
title(sprintf('Precision-Recall Curve (AP = %.3f)', ap)); grid on;


