

function bdhTest

fileStr = 'bdhtrainEdit.csv';
data = load(fileStr);

size(data(:,1));
datalengthcolumn = length(data(:,1));
datalengthrow = length(data(1,:));
choice = data(:,1);
a = data(:,2:12);
b = data(:,13:23);
a(1,:);
b(1,:);

together = cat(1,a,b);
tgNormedLengthColumn = length(together(:,1));
tgNormedLengthRow = length(together(1,:));


[dNormedTg,avgTg,stdevTg] = simpleNormOriginal(together); %tests
avgTg;
stdevTg;

tgNormedLengthColumn = length(dNormedTg(:,1));
tgNormedLengthRow = length(dNormedTg(1,:));


%[dNormedA,avgA,stdevA] = simpleNormOriginal(a); %tests
%avgA;
%stdevA;

%[dNormedB,avgB,stdevB] = simpleNormOriginal(b); %tests
%avgB;
%stdevB;

%aNormedLengthColumn = length(dNormedA(:,1))
%aNormedLengthRow = length(dNormedA(1,:))

dNormedA = dNormedTg(1:5500,:);
dNormedB = dNormedTg(5501:11000,:);

aNormedLengthColumn = length(dNormedA(:,1));
aNormedLengthRow = length(dNormedA(1,:));
bNormedLengthColumn = length(dNormedB(:,1));
bNormedLengthRow = length(dNormedB(1,:));


%figure(1)
%plot(dNormedA(:,1),dNormedB(:,1))
%figure(2)
%plot(a(:,1),b(:,1))
%figure(3)
%plot(dNormedA,dNormedB)
%figure(4)
%plot(a,b)

normedTogetherA = cat(2,choice,dNormedA);
normedTogether =  cat(2,normedTogetherA,dNormedB);

normedTogetherColumn = length(normedTogether(:,1));
normedTogetherRow    = length(normedTogether(1,:));
normedTogether(1,:);

%csvwrite('bdhNormedStandardTrain.csv',normedTogether);

keepDimensions = 11;
[pcaNormedTogether,eigVectors] = pcaOriginal(dNormedTg,keepDimensions);

pcaNormedA = pcaNormedTogether(1:5500,:);
pcaNormedB = pcaNormedTogether(5501:11000,:);

pcaNormedTogetherA = cat(2,choice,pcaNormedA);
pcaNormedTogether =  cat(2,pcaNormedTogetherA,pcaNormedB);

pcaNormedTogetherColumn = length(pcaNormedTogether(:,1));
pcaNormedTogetherRow    = length(pcaNormedTogether(1,:));

%csvwrite('bdhPcaNormedStandardTrain.csv',pcaNormedTogether);
%csvwrite('bdhPcaEigVectorsTrain.csv',eigVectors);

%scaling together

maxArray = max(together);
minArray = min(together);

bottom = maxArray - minArray;

for i=1:11000
    upperdelim = (together(i,:) - minArray);
    scaleTogether(i,:) =  upperdelim ./ bottom;
end


scaleA = scaleTogether(1:5500,:);
scaleB = scaleTogether(5501:11000,:);

scaleAColumn = length(scaleA(:,1));
scaleARow    = length(scaleA(1,:));
scaleBColumn = length(scaleB(:,1));
scaleBRow    = length(scaleB(1,:));
%scaleTogether

scaleSaveA = cat(2,choice,scaleA);
scaleSave =  cat(2,scaleSaveA,scaleB);

scaleSaveColumn = length(scaleSave(:,1));
scaleSaveRow    = length(scaleSave(1,:));

maxArray = max(scaleSave);
minArray = min(scaleSave);

%csvwrite('bdhScaledTrain.csv',scaleSave);

%a/b
%follower_following_ratio
original = cat(2,a,b);                                          %0.82
divOriginal = a ./ b;
difOriginal = a - b;                                            %0.79
origWithExtraDiv = cat(2,original,divOriginal);
origWithExtraDif = cat(2,original,difOriginal);                 %0.81
origWithExtraDivDif = cat(2,origWithExtraDiv,difOriginal);
divNormTest = dNormedA ./ dNormedB;                             %0.63
difNormTest = dNormedA - dNormedB;                              %0.80
difScaleTest = scaleA - scaleB;                                 %0.79
divScaleTest = scaleA ./ scaleB;
normTest  = cat(2,dNormedA,dNormedB);                           %0.79
scaleTest = cat(2,scaleA,scaleB);                               %0.78
normWithExtraDiv  = cat(2,normTest,divNormTest);                %0.65
scaleWithExtraDiv = cat(2,scaleTest,divScaleTest);              
normWithExtraDif  = cat(2,normTest,difNormTest);                %0.80
scaleWithExtraDif = cat(2,scaleTest,difScaleTest);              %0.78
normWithExtraDifDiv  = cat(2,normWithExtraDiv,difNormTest);     
scaleWithExtraDifDiv = cat(2,scaleWithExtraDiv,difScaleTest);


[trainingData,trainingDataTargets] = randomizeDataForKFold(scaleTest,choice);
kFold = 5;
aucs = 0;

for i=1:kFold      
    [trainData,testData,trainDataHoldout,testDataHoldout] = nFoldCrossValidation(trainingData,trainingDataTargets,i,kFold);
    
    %RBF = 1
    %MLP = 2
    whichModel = 2;

    switch whichModel
    case 1  
        output = rbf(trainData,testData,trainDataHoldout);
        
    case 2        
        output = mlp(trainData,testData,trainDataHoldout);
    end

    
    %Calculate error
    %accuracy = calculateError(output,testDataHoldout) %tests
    [x,y,t,auc] = perfcurve(testDataHoldout,output,1);
    auc
    aucs = aucs + auc;
end
aucAvg = aucs/kFold



function [accuracy] = calculateError(output,trainingDataTargets)
    correct = 0;
    length(trainingDataTargets)
    %output
    
    
    for i=1:length(trainingDataTargets)
        if (output(i,1) < 0.5)
            if trainingDataTargets(i,1) == 0
                correct = correct + 1;
            end    
        elseif trainingDataTargets(i,1) == 1
            correct = correct + 1;
        end
    end
    accuracy = correct/length(trainingDataTargets);
    
    
    
    



function [output] = rbf(ppData,targetsNormed,ppDataHoldout)

    %train RBF
    
    %whichRBF
    %Normal = 1
    %Guassian = 2
    %Multiquadrics = 3
    %Inverse Multiquadrics = 4
    whichRBF = 4;

    %RBF Kmeans
    %just in case we are using quadratics or inverse quadratics, we set c
    c = 5;
    %c = 6.8.....mseAvg = 19.9....aveErr = 3.0495
    %c = 5.......mseAvg = 19.2....aveErr = 2.9763
    %c = 4.......mseAvg = 19.06...aveErr = 2.9447
    %c = 3.5.....mseAvg = 19.05...aveErr = 2.9192
    
    %k = number of centres
    k = 55;
    sigma = 3;
    [weights,centres] = trainRBFnKM(ppData,targetsNormed,whichRBF,k,c,sigma);
    
    %run RBF
    [output] = runRBFnKM(ppDataHoldout,weights,centres,sigma,whichRBF,c); %kfold
    %[output] = runRBFnKM(ppDataHoldout,weights,centres,sigma,whichRBF,c); %tests to predict
    %[TRoutput] = runRBFnKM(ppData,weights,centres,sigma,whichRBF,c); %tests training
    
    


function[weights,centres] = trainRBFnKM(trainingData,targets,whichRBF,k,c,sigma)

[r,centres] = getEuclideanDistanceSquareKMeans(trainingData,k);

%sigma = 1;

switch whichRBF
    case 1  
        %identity rbf
        fi = eye(length(r)) * r;
    case 2
        %guassian rbf
        %dynamic sigma according to dmax/sqrt(2m)
        %dmaxMTRX = max(centres) - min(centres);
        %dmax = max(dmaxMTRX) - min(dmaxMTRX);
        %sigma = dmax / sqrt(2*length(centres));
        fi = exp(-((r.^2) / (2 * sigma.^2)));
    case 3
        fi = sqrt((r.^2 + c.^2));
    case 4
        fi = (r.^2 + c.^2).^(-1/2);
        
end

weights = pinv(fi) * targets;

%regularization
%lambda = 0.2;
%weights = pinv((fi' * fi) + (lambda * eye(k,k)))*fi'*targets;


function[r,centres] = getEuclideanDistanceSquareKMeans(trainingPoints,k)

[idx,centres] = kmeans(trainingPoints,k,'replicates',5);

for i=1:k
    c = repmat(centres(i,:), length(trainingPoints),1);
    difference = (trainingPoints - c).^2;
    r(:,i) = sqrt(sum(difference,2));
end



function[output] = runRBFnKM(holdoutData,weights,centres,sigma,whichRBF,c)

[r] = getEuclideanDistanceSquare(holdoutData,centres);

switch whichRBF
    case 1  
        %identity rbf
        fi = eye(length(r)) * r;
    case 2
        %guassian rbf
        %fi = exp(-(power(r,2) / (2 * power(sigma,2))));
        fi = exp(-((r.^2) / (2 * sigma.^2)));
    case 3
        fi = sqrt((r.^2 + c.^2));
    case 4
        fi = (r.^2 + c.^2).^(-1/2);
    
end

output = fi * weights;



function[r] = getEuclideanDistanceSquare(trainingPoints,centres)

for i=1:length(centres)
    nan_locations = find(isnan(trainingPoints));
    trainingPoints(nan_locations) = 0;
    c = repmat(centres(i,:), length(trainingPoints(:,1)),1);
    difference = (trainingPoints - c).^2;
    r(:,i) = sqrt(sum(difference,2));
end


%--------------------------------------------------------------
%--------------------------------------------------------------


       
function[deNormalisedOutput] = mlp(trainingData, targets, trainingDataHoldout)
    
trainSize = size(trainingData);
holdoutSize = size(trainingDataHoldout);

numInputNeurons = trainSize(2);
hiddenLayers = 1;
layer1 = 15;
layer2 = 15;
lRate = 0.01;

%initialise weights
w1 = (rand(numInputNeurons+1, layer1)*0.4) - 0.2;
w2 = (rand(layer1+1, layer2)*0.4) - 0.2;
w3 =  (rand(layer2+1, 1)*0.4) - 0.2;
%maxim = 0.1*times;
%w1 = (rand(numInputNeurons+1, layer1) * maxim) - (maxim/2);
%w2 = (rand(layer1+1, layer2) * maxim) - (maxim/2);
%w3 =  (rand(layer2+1, 1) * maxim) - (maxim/2);


%Train MLP
for epochs=1:1000
    errors = zeros(1,trainSize(1));
    for i=1:trainSize(1)
        inputNeurons = trainingData(i,:);
        inputNeurons = [inputNeurons 1]; %adding the bias to every row
        target = targets(i);
        %[output(i), w1, w2] = trainMLP1Layer(inputNeurons, target, lRate, w1, w2);
        [output(i), w1, w2, w3] = trainMLP2Layer(inputNeurons, target, lRate, w1, w2, w3);
    end;
    %stats = [mean(errors) std(errors)] 
end;


%Test MLP
for i=1:holdoutSize(1)
    inputNeuronsH = trainingDataHoldout(i,:);
    inputNeuronsH = [inputNeuronsH 1]; %adding the bias to every row
    %[deNormalisedOutput(i)] = runMLP1Layer(inputNeuronsH, w1, w2);
    [deNormalisedOutput(i)] = runMLP2Layer(inputNeuronsH, w1, w2, w3);
end;




function [output, nw1, nw2] = trainMLP1Layer(inputNeurons, target, lRate, w1, w2)

%feed forward pass
hiddenLayer = tanh(inputNeurons * w1);
hiddenLayer = [hiddenLayer 1]; 

output = tanh(hiddenLayer * w2);

%Back propagation
deltaOutput = (target - output) * (1 - output^2);

%change hidden to output weights
w2delta = (lRate * hiddenLayer * deltaOutput)';
nw2 = w2 + w2delta;

%change input to hidden weights
deltaHiddenLayer = diag(1 - hiddenLayer.^2);
deltaHiddenLayer = deltaHiddenLayer * w2 * deltaOutput;
%deltaHiddenLayer = diag(1 - hiddenLayer.^2) * w2 * deltaOutput;
w1delta = (lRate * deltaHiddenLayer * inputNeurons)';
w1delta(:,end) = [];
nw1 = w1 + w1delta;

    

function [output] = runMLP1Layer(inputNeurons, w1, w2)
    
hiddenLayer = tanh(inputNeurons*w1);
hiddenLayer = [hiddenLayer 1]; 
    
output = tanh(hiddenLayer*w2);

  



function [output, nw1, nw2, nw3] = trainMLP2Layer(inputNeurons, target, lRate, w1, w2, w3)

hiddenLayer1 = tanh(inputNeurons * w1);
hiddenLayer1 = [hiddenLayer1 1]; 

hiddenLayer2 = tanh(hiddenLayer1 * w2);
hiddenLayer2 = [hiddenLayer2 1];

output = tanh(hiddenLayer2 * w3);
    
deltaOutput = (target-output)*(1-output^2);


%output weights
w3delta = (lRate * hiddenLayer2 * deltaOutput)';
nw3 = w3delta + w3;

%hiddenLayer2 weights
deltaHiddenLayer2 = diag(1 - hiddenLayer2.^2);
deltaHiddenLayer2 = deltaHiddenLayer2 * w3 * deltaOutput;
%deltaHiddenLayer2 = diag(1 - hiddenLayer2.^2) * w3 * deltaOutput;
w2delta = (lRate * deltaHiddenLayer2 * hiddenLayer1)';
w2delta(:,end) = [];
nw2 = w2 + w2delta;

%hiddenLayer1 weights
deltaHiddenLayer1 = diag(1 - hiddenLayer1.^2);
deltaHiddenLayer1 = deltaHiddenLayer1 * w2 * deltaHiddenLayer2(1:end-1,:);
%deltaHiddenLayer1 = diag(1 - hiddenLayer1.^2) * w2 * deltaHiddenLayer2(1:end-1,:);
deltaw1 = (lRate * deltaHiddenLayer1 * inputNeurons)';
deltaw1(:,end) = [];
nw1 = w1 + deltaw1;


    
    
function [output] = runMLP2Layer(inputNeurons, w1, w2, w3)

hiddenLayer1 = tanh(inputNeurons * w1);
hiddenLayer1 = [hiddenLayer1 1]; 

hiddenLayer2 = tanh(hiddenLayer1 * w2);
hiddenLayer2 = [hiddenLayer2 1];

output = tanh(hiddenLayer2 * w3);


%--------------------------------------------------------------
%--------------------------------------------------------------



















function [trainData,testData,trainDataHoldout,testDataHoldout] = nFoldCrossValidation(trainingData,trainingDataTargets,i,k)

fold = round(length(trainingDataTargets)/k)-1;

trainDataHoldout = trainingData(1+(i*fold)-fold : i*fold, : );
testDataHoldout  = trainingDataTargets(1+(i*fold)-fold : i*fold);

if i~=1 & i~=k
    trainData = trainingData([1:(i*fold)-fold  1+(i*fold):end], : );
    testData  = trainingDataTargets([1:(i*fold)-fold  1+(i*fold) : end]);
    
elseif i==1
    trainData = trainingData(1+(i*fold) : end, : );
    testData  = trainingDataTargets(1+(i*fold) : end);
    
elseif i==k
    trainData = trainingData(1:(i*fold)-fold, : );
    testData  = trainingDataTargets(1:(i*fold)-fold);
    
end



function [trainData,testData] = randomizeDataForKFold(trainingData,trainingDataTargets)

rnd = 1:length(trainingDataTargets);
rnd(randperm(numel(rnd))) = rnd;

trainData = trainingData(rnd,:);
testData  = trainingDataTargets(rnd);




function[dNormed,avg,stdev] = simpleNormOriginal(data)

d = data';
[m,n] = size(d);

%Calculate mean of matrix (mean of each row) and deduct it from matrix
avg = mean(d,2);
dMinusMean = d - repmat(avg,1,n);

dMinusMean = dMinusMean';

for i = 1:1:m
    stdev(i) = sqrt(var(data(:,i)));
    dNormed(:,i) = dMinusMean(:,i) / stdev(i);
end





function[pcaData,eigVectors] = pcaOriginal(data,firstD)

dCov = cov(data);

try
    
%eigVectors are Principal Components
[eigVectors,eigValues] = eig(dCov);
eigValues = diag(eigValues);

eigValues = flipud(eigValues);
eigVectors = flipud(eigVectors')';

%take first d eigVectors as principal components
eigVectors = eigVectors(:,1:firstD);
pcaData = data * eigVectors;

catch e
disp(e.message)
pcaData = data;
end
