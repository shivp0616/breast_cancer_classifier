datas = csvread('cancer_data.csv');  %read the data
target = datas(:,2);                 %set the target as the second column contains the class
datas = datas(:, 3:32);              %the attributes are contains in columns 3 to 32

%% classification tree
X = datas;                      %X contains the features
Y = target;                     %Y contains the target or you can say labels
tc = fitctree(X,Y);             %fits the input and the target to the classifier
C0 = predict(tc,X);             %predicts the result after training
cMat0 = confusionmat(Y,C0);     %confusion matrix

%% NaiveBayes 
O1 = fitNaiveBayes(datas, target); 
C1 = O1.predict(datas);
cMat1 = confusionmat(target,C1);

%% k-Nearest Neighbor Classifier
X = datas;
Y = target;
Mdl = fitcknn(X,Y);
C2 = predict(Mdl,X);
cMat2 = confusionmat(Y,C2);

%% SVM
X = datas;
Y = target;
CVSVMModel = fitcsvm(X,Y,'Holdout',0.15,'Standardize',true);
CompactSVMModel = CVSVMModel.Trained{1}; % Extract trained, compact classifier
testInds = test(CVSVMModel.Partition);   % Extract the test indices
XTest = X(testInds,:);
YTest = Y(testInds,:);
C3 = predict(CompactSVMModel,XTest);
cMat3 = confusionmat(YTest,C3);
table(YTest,C3,'VariableNames',{'TrueLabel','PredictedLabel'})

%% Discriminant Analysis Classifier
DAC = fitcdiscr(X,Y);
dacP = predict(DAC,X);
cMat0 = confusionmat(Y,dacP);
