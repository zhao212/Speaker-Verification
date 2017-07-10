clear all
% EE 214A project
% Authors: Wei Qian, Jiaoyang Li, Qiyue Zhao, Gena Xie
tic
%--------------------------
% Define parameter
%--------------------------
frameShift = 0.001;     % [sec]

%NumFeatures = 1;        % Number of features
%NumFolds = 5;           % Number of cross validation folds

%--------------------------
% Get label
%--------------------------


label = load('dataLabel.mat');

dataLabel = cat(1,label.FEMALE,label.MALE);

crossInd = load('crossValIdx.mat');
crossValIdx = cat(1,crossInd.FEMALE,crossInd.MALE);
crossValIdx = crossValIdx==1;
NumPairs = size(dataLabel,1);
NumFolds = size(crossValIdx,2);


% %%
% %--------------------------
% % Prepare for cross-validation
% %--------------------------

fileNames = unique(dataLabel(:,1:2));
nFile = size(fileNames,1);


%%
h=waitbar(0,'extracting features');

for iFile = 1:nFile
    
    [snd,Fs] = audioread(['WavDataEdit/' fileNames{iFile} ]);   % NEW MATLAB VERSION
    
    %       [f0, ~] = fast_mbsc_fixedWinlen_tracking(snd, Fs);  F0??
    %       F0 = nanmean(f0(f0>0)); % Averaged F0
    %      save(['Features/F0/' fileNames{iFile}(1:end-3) 'mat'],'F0','-v7.3');
    
    %MFCC
    mfcc = melcepst(snd, Fs, 'dD');
    mfcc = mean(mfcc);
    
    %         % LPCC
    %         LPCC = lpccovar(snd, 12);
    %         LPCC = LPCC(2:end);
    %   %PLP
    %     [cepstra2, spectra1, pspectrum1, lpcas1] = rastaplp(snd, Fs, 0, 13);%PLP
    %
    % %     % first derative
    %     cep_d1 = cepstra2(:,1:end-1) - cepstra2(:,2:end);
    %     cep_d1_m = mean(cep_d1');
    %     cep_d1_m = cep_d1_m / norm(cep_d1_m,2);
    %     % cep
    %     cep_m = mean(cepstra2');
    %     cep_m = cep_m / norm(cep_m,2);
    %
    %     cep_m = [cep_m, cep_d1_m];
    %     %asp_m = mean(aspectrum2');
    %     %psp_m = mean(pspectrum2');
    %     %save(['Features/asp/' fileNames{iFile}(1:end-3) 'mat'],'asp_m','-v7.3');
    %     %save(['Features/psp/' fileNames{iFile}(1:end-3) 'mat'],'psp_m','-v7.3');
    
    %     save(['Features/cep/' fileNames{iFile}(1:end-3) 'mat'],'cep_m','-v7.3');
    %     save(['Features/LPCC/' fileNames{iFile}(1:end-3) 'mat'],'LPCC','-v7.3');
    save(['Features/MFCC/' fileNames{iFile}(1:end-3) 'mat'],'mfcc','-v7.3');
    
end

%%

x = NaN*ones(NumPairs, 21);  % features
y = NaN*ones(NumPairs, 1);  % variable to predict
z = NaN*ones(NumPairs, 1);  % class label

fprintf('loading dataset\n')
for n=1:NumPairs
    waitbar(n/NumPairs)
    
    sd1 = dataLabel{n,1};
    sd2 = dataLabel{n,2};
    
    
    %--------------------------
    % Load feature
    %--------------------------
    %F0
    f0_1 = load(['Features/F0/' sd1(1:end-3) 'mat']);
    f0_2 = load(['Features/F0/' sd2(1:end-3) 'mat']);
    f0_1 = f0_1.strF0;
    f0_2 = f0_2.strF0;
    f0_1 = nanmean(f0_1(f0_1>0));
    f0_2 = nanmean(f0_2(f0_2>0));
    %     %F1~F4
    %     fermant1 = load(['Features/F1_F4/' sd1(1:end-3) 'mat']);
    %     fermant2 = load(['Features/F1_F4/' sd2(1:end-3) 'mat']);
    %     f1_1 = nanmean(fermant1.sF1(fermant1.sF1>0));
    %     f2_1 = nanmean(fermant1.sF2(fermant1.sF2>0));
    %     f3_1 = nanmean(fermant1.sF3(fermant1.sF3>0));
    %     f4_1 = nanmean(fermant1.sF4(fermant1.sF4>0));
    %
    %     f1_2 = nanmean(fermant2.sF1(fermant2.sF1>0));
    %     f2_2 = nanmean(fermant2.sF2(fermant2.sF2>0));
    %     f3_2 = nanmean(fermant2.sF3(fermant2.sF3>0));
    %     f4_2 = nanmean(fermant2.sF4(fermant2.sF4>0));
    %
    %     %Energy
    %     Energy1 = load(['Features/energy/' sd1(1:end-3) 'mat']);
    %     Energy2 = load(['Features/energy/' sd2(1:end-3) 'mat']);
    %     Eng1 = nanmean(Energy1.Energy');
    %     Eng2 = nanmean(Energy2.Energy');
    %
    %
    % %     asp_1 = load(['Features/asp/' sd1(1:end-3) 'mat']);
    % %     asp_2 = load(['Features/asp/' sd2(1:end-3) 'mat']);
    % %     psp_1 = load(['Features/psp/' sd1(1:end-3) 'mat']);
    % %     psp_2 = load(['Features/psp/' sd2(1:end-3) 'mat']);
    
    %     cep_1 = load(['Features/cep/' sd1(1:end-3) 'mat']);
    %     cep_2 = load(['Features/cep/' sd2(1:end-3) 'mat']);
    %     LPCC_1 = load(['Features/LPCC/' sd1(1:end-3) 'mat']);
    %     LPCC_2 = load(['Features/LPCC/' sd2(1:end-3) 'mat']);
    
    cpp1 = load(['Features/feature_data/' sd1(1:end-3) 'mat']);
    cpp2 = load(['Features/feature_data/' sd2(1:end-3) 'mat']);
    cpp1 = nanmean(cpp1.CPP);
    cpp2 = nanmean(cpp2.CPP);
    
    mfcc1 = load(['Features/mfcc/' sd1(1:end-3) 'mat']);
    mfcc2 = load(['Features/mfcc/' sd2(1:end-3) 'mat']);
    mfcc11 = mfcc1.mfcc;
    mfcc22 = mfcc2.mfcc;
    %     mfcc11 = [mfcc11(1:12)/norm(mfcc11(1:12),2), mfcc11(13:18)/norm(mfcc11(13:18),2)];
    %     mfcc22 = [mfcc22(1:12)/norm(mfcc22(1:12),2), mfcc22(13:18)/norm(mfcc22(13:18),2)];
    mfcc11 = [mfcc11(1:12), mfcc11(13:18)];
    mfcc22 = [mfcc22(1:12), mfcc22(13:18)];
    
    h1 = load(['Features/feature_data/' sd1(1:end-3) 'mat']);
    h2 = load(['Features/feature_data/' sd2(1:end-3) 'mat']);
    
    H1H2c_1 = nanmean(h1.H1H2c(h1.H1H2c>0));
    H1H2c_2 = nanmean(h2.H1H2c(h2.H1H2c>0));
    %     H2H4c_1 = nanmean(h1.H2H4c(h1.H2H4c>0));
    %     H2H4c_2 = nanmean(h2.H2H4c(h2.H2H4c>0));
    
    feat1 = [ f0_1, mfcc11, cpp1, H1H2c_1]; %, H2H4c_1
    feat2 = [ f0_2, mfcc22, cpp2, H1H2c_2]; %, H2H4c_2
    
    %--------------------------
    % Save into variables
    %--------------------------
    x(n,:) = abs(feat1 - feat2); % Use the difference between mean pitch
    z(n) = dataLabel{n,3}; % intra-speaker indicator
end
delete(h);
%%
% x1 = pca(x', 'NumComponents', 20);


rmsErr = NaN*ones(NumFolds,1);
errRate= NaN*ones(NumFolds,1);

for n=1:NumFolds
    
    %fprintf('crossvalidation\n')
    
    % features
    
    x_train = x(~crossValIdx(:,n),:);
    x_test  = x(crossValIdx(:,n),:);
    
    %
    % intra-speaker indication label
    
    z_train = z(~crossValIdx(:,n));
    z_test  = z(crossValIdx(:,n));
    
    %--------------------------
    % Linear regression
    %--------------------------
    
    %% Modify your code here >>>>>>>>>>>>>>>>>>>>>>>>>>>>
    %fprintf('modeling\n')
    
    
    %--------------------------
    % Naive Bayse classifier
    %--------------------------
    
    
    NBModel = fitcnb(x_train,z_train);
    z_test_hat = predict(NBModel,x_test);
    
    %random forest
    %         Md = TreeBagger(500,x_train, z_train);
    %         z_test_hat = predict(Md, x_test);
    %         z_test_hat = str2num(cell2mat(z_test_hat));
    
    %         %neural network
    %         net = fitnet(50);
    %         net = train(net,x_train',z_train');
    %          z_test_hat = net(x_test');
    %          z_test_hat = z_test_hat';
    %          for i = 1:length(z_test_hat)
    %              if z_test_hat(i)>0.5
    %                  z_test_hat(i) = 1;
    %              else
    %                  z_test_hat(i) = 0;
    %              end
    %          end
    
    %         % decision tree model
    %         DTModel = fitctree(x_train,z_train);
    %         z_test_hat = predict(DTModel,x_test);
    
    %     % svm
    %         SVMModel = fitcsvm(x_train,z_train);
    %         z_test_hat = predict(SVMModel,x_test);
    
    %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<  Modify your code here
    
    err = z_test_hat ~= z_test;
    errRate(n) = sum(err)/length(z_test);
end

fprintf('averaged classification error = %.2f %% \n', 100* mean(errRate));
toc



