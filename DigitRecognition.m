clear all
load('azip.mat');  % Training images
load('dzip.mat');  % Training labels
load('testzip.mat');  % Test images
load('dtest.mat');  % Test labels

% Normalize training data
meanTrain = mean(azip, 2);
stdTrain = std(azip, 0, 2);
azip = (azip - meanTrain) ./ stdTrain;

% Normalize test data
testzip = (testzip - meanTrain) ./ stdTrain;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
net = patternnet(20);  % hidden layer 
net.divideParam.trainRatio = 0.8;  % 80% of data for training
net.divideParam.valRatio = 0.1;    % 10% for validation
net.divideParam.testRatio = 0.1;   % 10% for testing

[net,tr] = train(net, azip, full(ind2vec(dzip + 1)));  % dzip + 1 to match MATLAB indexing
%ind2vec fonksiyonu, etiketleri vektör formatına çevirir.
testOutputs = net(testzip);
[~, predictedLabels] = max(testOutputs, [], 1);
accuracy = sum(predictedLabels == dtest + 1) / numel(dtest);  % Calculate accuracy
disp(['Accuracy of the model is: ', num2str(accuracy * 100), '%']);

% Display random sample of 10 images
figure;
for i = 1:5
    subplot(1, 5, i);
    randIndex = randi([1, numel(dtest)]);  % Random index
    ima2(testzip(:,randIndex));
    title(['Pred: ', num2str(predictedLabels(randIndex)-1), ', Target: ', num2str(dtest(randIndex))]);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize the counters for correct and incorrect predictions
numClasses = max(dtest) + 1; % Assuming classes are 0 to max(dtest)
correctCounts = zeros(numClasses, 1);
totalCounts = zeros(numClasses, 1);

% Analyze predictions
for i = 0:numClasses-1
    classIndices = dtest == i;
    totalCounts(i+1) = sum(classIndices);
    correctCounts(i+1) = sum(predictedLabels(classIndices) == i + 1);
end

% Display the results for each class
for i = 0:numClasses-1
    fprintf('Digit %d: Correctly predicted %d/%d (%.2f%%)\n', ...
        i, correctCounts(i+1), totalCounts(i+1), ...
        (correctCounts(i+1) / totalCounts(i+1)) * 100);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hesaplanan tahminler ve gerçek etiketler kullanılarak confusion matrisi oluşturulur
confMat = confusionmat(dtest + 1, predictedLabels);
figure;
% Confusion matris
labels = arrayfun(@num2str, 0:9, 'UniformOutput', false); % Etiketleri 0'dan 9'a kadar olan string dizi olarak oluştur
confChart = confusionchart(confMat, labels);
confChart.Title = 'Confusion Matrix for Model Predictions';
confChart.Parent.Position = [10 10 800 600];  % Position and size


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Precision, Recall ve F1 Score hesaplama
numClasses = size(confMat, 1);
precision = zeros(1, numClasses);
recall = zeros(1, numClasses);
f1Scores = zeros(1, numClasses);

for i = 1:numClasses
    TP = confMat(i, i);
    FP = sum(confMat(:, i)) - TP;
    FN = sum(confMat(i, :)) - TP;
    TN = sum(confMat(:)) - TP - FP - FN;
    
    precision(i) = TP / (TP + FP);
    recall(i) = TP / (TP + FN);
    f1Scores(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
end

% Makro ortalamaları hesapla
macroPrecision = mean(precision);
macroRecall = mean(recall);
macroF1 = mean(f1Scores);

% Sonuçları görüntüle
fprintf('Makro Precision: %.2f%%\n', macroPrecision * 100);
fprintf('Makro Recall: %.2f%%\n', macroRecall * 100);
fprintf('Makro F1 Score: %.2f%%\n', macroF1 * 100);
