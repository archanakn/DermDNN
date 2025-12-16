clear; clc;

%% Paths & basic setup
trainFolder = 'give train folder path';
testFolder  = 'give test folder path';
imageSize  = [224 224];                          % small & fast for <800 images
rng(42);                                         % reproducibility

%% Read images (labels from subfolder names)
imdsTrain = imageDatastore(trainFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

imdsTest = imageDatastore(testFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
% Stratified split: 70% train, 15% val, 15% test
% [imdsTrain, imdsRest] = splitEachLabel(imds,0.80, 'randomized');
[imdsVal,   imdsTest] = splitEachLabel(imdsTest, 0.50, 'randomized');

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
    % maxPooling2dLayer(2,'Stride',2,'Name','pool3')
    % dropoutLayer(0.3,'Name','drop1')

    % convolution2dLayer(3, 192, 'Padding','same', 'Name','conv4')
    % batchNormalizationLayer('Name','bn4')
    % reluLayer('Name','relu4')
    % maxPooling2dLayer(2,'Stride',2,'Name','pool4')

    % convolution2dLayer(3, 96, 'Padding','same', 'Name','conv5')
    % batchNormalizationLayer('Name','bn5')
    % reluLayer('Name','relu5')
    
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
    'L2Regularization',1e-5,...
    'Plots', 'training-progress');

%% Train
net = trainNetwork(augTrain, layers, opts);

%% Evaluate (final accuracy printed at end)
[YPred, scores] = classify(net, augTest);
YTrue = imdsTest.Labels;
% acc   = mean(YPred == YTrue);
% 
% disp('--------------------------');
% disp(['Test Accuracy: ' num2str(acc*100, '%.2f') '%']);
confMat = confusionmat(YTrue, YPred);
disp('Confusion Matrix:');
disp(confMat);

% Extract TP, FP, FN, TN for binary classification
numClasses = size(confMat,1);

precision = zeros(numClasses,1);
recall    = zeros(numClasses,1);
f1        = zeros(numClasses,1);
support   = sum(confMat,2); % samples per class

for c = 1:numClasses
    TP = confMat(c,c);
    FP = sum(confMat(:,c)) - TP;
    FN = sum(confMat(c,:)) - TP;
    TN = sum(confMat(:)) - TP - FP - FN;

    if (TP+FP) > 0
        precision(c) = TP / (TP + FP);
    else
        precision(c) = 0;
    end

    if (TP+FN) > 0
        recall(c) = TP / (TP + FN);
    else
        recall(c) = 0;
    end

    if (precision(c)+recall(c)) > 0
        f1(c) = 2 * (precision(c)*recall(c)) / (precision(c)+recall(c));
    else
        f1(c) = 0;
    end
end

% Overall accuracy
accuracy = mean(YPred == YTrue);

% Macro averages
macroPrecision = mean(precision);
macroRecall    = mean(recall);
macroF1        = mean(f1);

% Weighted averages (by class support size)
weightedPrecision = sum(precision .* support) / sum(support);
weightedRecall    = sum(recall .* support) / sum(support);
weightedF1        = sum(f1 .* support) / sum(support);

% Display results
disp('--------------------------');
fprintf('Overall Accuracy : %.2f %%\n', accuracy*100);
fprintf('Macro Precision  : %.2f %%\n', macroPrecision*100);
fprintf('Macro Recall     : %.2f %%\n', macroRecall*100);
fprintf('Macro F1-score   : %.2f %%\n', macroF1*100);
fprintf('Weighted Precision: %.2f %%\n', weightedPrecision*100);
fprintf('Weighted Recall   : %.2f %%\n', weightedRecall*100);
fprintf('Weighted F1-score : %.2f %%\n\n', weightedF1*100);

% Per-class results
disp('Class-wise Metrics:');
for c = 1:numClasses
    fprintf('Class %d -> Precision: %.2f %% | Recall: %.2f %% | F1: %.2f %% | Support: %d\n', ...
        c, precision(c)*100, recall(c)*100, f1(c)*100, support(c));
end

% Case 1: trainedNet is SeriesNetwork or DAGNetwork (most transfer learning)
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

% disp('Per-class metrics:');
% C = confusionmat(YTrue, YPred, 'Order', classes);
% confChart = confusionchart(YTrue, YPred, 'Order', classes);
% confChart.Title = 'Confusion Matrix (Test Set)';

% Precision/Recall/F1 per class
% prec = diag(C) ./ max(1,sum(C,1))';
% rec  = diag(C) ./ max(1,sum(C,2));
% f1   = 2 * (prec.*rec) ./ max(1e-12,(prec+rec));
% T = table(classes, prec, rec, f1, 'VariableNames',{'Class','Precision','Recall','F1'});
% disp(T);

    % 'LearnRateSchedule','piecewise',...
    % 'LearnRateDropFactor',0.1,...
    % 'LearnRateDropPeriod',10,...
    % 'L2Regularization',1e-5, ...
