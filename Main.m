clc
clear all
close all

method = input('Ingrese el método con el cual lo quiere resolver: KNN=1, Random Forest=2, RNA=3, GMM=4, SVM=5, Pearson=6, fisher=7, SFS=8, PCA=9, classify=10 : ');

load('Test.mat');    
load('Data.mat');

Data(:,4) = [];
Test(:,4) = [];

myData = [Data;Test];

[N, columnas] = size(myData);

repetitiones = 10;

numClasses = length(unique(myData(:, 1)));

efficiencyTest = zeros(1,repetitiones);
% 
% clase1 = 0;
% clase2 = 0;
% clase3 = 0;
% clase4 = 0;
% clase5 = 0;
% clase6 = 0;
% clase7 = 0;
% 
% for i=1:N
%     if myData(i,1) == 1
%         clase1 = clase1 + 1;
%     elseif myData(i,1) == 2
%         clase2 = clase2 + 1;
%     elseif myData(i,1) == 3
%         clase3 = clase3 + 1;
%     elseif myData(i,1) == 4
%         clase4 = clase4 + 1;
%     elseif myData(i,1) == 5
%         clase5 = clase5 + 1;
%     elseif myData(i,1) == 6
%         clase6 = clase6 + 1;
%     elseif myData(i,1) == 7
%         clase7 = clase7 + 1;
%     end
% end
% 
% disp(clase1);
% disp(clase2);
% disp(clase3);
% disp(clase4);
% disp(clase5);
% disp(clase6);
% disp(clase7);

if method == 1 % KNN

        
    for fold=1:repetitiones
       
       rng('default');
       partition = cvpartition(N,'Kfold',repetitiones);
        
       indices = partition.training(fold); 

       Xtrain = myData(partition.training(fold), 2:19);
       Ytrain = myData(partition.training(fold), 1);

       Xtest = myData(partition.test(fold), 2:19);
       Ytest = myData(partition.test(fold), 1);
        
        [Xtrain,mu,sigma] = zscore(Xtrain);
        Xtest = Normalize(Xtest,mu,sigma);

        k = 4;
        Yesti = KNN(Xtest, Xtrain, Ytrain, k); 
       
        matrixConfusion = zeros(numClasses,numClasses);
        for i=1:size(Xtest,1)
            matrixConfusion(Yesti(i),Ytest(i)) = matrixConfusion(Yesti(i),Ytest(i)) + 1;
        end
        efficiencyTest(fold) = sum(diag(matrixConfusion))/sum(sum(matrixConfusion));
        
        impixelregion(imagesc(matrixConfusion));
    end
        
    %efficiency = (sum(Yesti==Ytest)) / length(Ytest);
    %error = 1 - efficiency;
    
    efficiency = mean(efficiencyTest);
    IC = std(efficiencyTest);
    
    text = ['La eficiencia obtenida fue = ', num2str(efficiency),' +- ',num2str(IC)];
    disp(text);
    
elseif method == 2 % Random Forest
        
    % SFS
    %{
    myData(:,6) = []; %caracteristica 5
    myData(:,6) = []; %caracteristica 6
    myData(:,6) = []; %caracteristica 7
    myData(:,6) = []; %caracteristica 8
    myData(:,6) = []; %caracteristica 9
    myData(:,9) = []; %caracteristica 13
    %}
    
    for fold=1:repetitiones
       
       rng('default');
       partition = cvpartition(N,'Kfold',repetitiones);
        
       indices = partition.training(fold); 

        Xtrain = myData(partition.training(fold), 2:19); %se cambia a 13
        Ytrain = myData(partition.training(fold), 1);

        Xtest = myData(partition.test(fold), 2:19);
        Ytest = myData(partition.test(fold), 1);   
        
            
        [Xtrain,mu,sigma] = zscore(Xtrain);
        Xtest = Normalize(Xtest,mu,sigma);

        numTrees = 50;
        
        model = TrainForest(numTrees, Xtrain, Ytrain);
        
        Yest = TestForest(model, Xtest);
        
        confusionMatrix = zeros(numClasses, numClasses);
        for i = 1 : size(Xtest, 1)
            confusionMatrix(Yest(i),Ytest(i)) = confusionMatrix(Yest(i), Ytest(i)) + 1;
        end
        efficiencyTest(fold) = sum(diag(confusionMatrix)) / sum(sum(confusionMatrix));
        
        impixelregion(imagesc(confusionMatrix));
    end    
    
    efficiency = mean(efficiencyTest);
    IC = std(efficiencyTest);
    text=['La eficiencia obtenida fue = ', num2str(efficiency),' +- ',num2str(IC)];
    disp(text);
    
elseif method == 3 % RNA
    
    %Y = [ones(1,50), 2*ones(1,50), 3*ones(1,50)];
%     
%     for iteraction = 1:N
%         
%         if myData(interaction, 1) == 1
%             myData(interaction, 1) = 0000001;
%         elseif myData(interaction, 1) == 2
%             myData(interaction, 1) = 0000010;
%         elseif myData(interaction, 1) == 3
%             myData(interaction, 1) = 0000100;
%         elseif myData(interaction, 1) == 4
%             myData(interaction, 1) = 0001000;
%         elseif myData(interaction, 1) == 5
%             myData(interaction, 1) = 0010000;
%         elseif myData(interaction, 1) == 6
%             myData(interaction, 1) = 0100000;
%         elseif myData(interaction, 1) == 7
%             myData(interaction, 1) = 1000000;
%         end
%         
%         
%     end
    

    %SFS
%     myData(:,6) = []; %caracteristica 5
%     myData(:,6) = []; %caracteristica 6
%     myData(:,6) = []; %caracteristica 7
%     myData(:,6) = []; %caracteristica 8
%     myData(:,6) = []; %caracteristica 9
%     myData(:,9) = []; %caracteristica 13
    
    
    [~, loc] = ismember(myData(:, 1), unique(myData(:, 1)));
    
    y_one_hot  = ind2vec(loc');
    
    y2 = full(y_one_hot);
    y_one_hot = y_one_hot';
    y2 = y2';
    
    for fold=1:repetitiones
        
        rng('default');
        partition = cvpartition(N,'Kfold',repetitiones);
        
        indices = partition.training(fold); 

        Xtrain = myData(partition.training(fold), 2:19);
        Ytrain = y2(partition.training(fold), :);

        Xtest = myData(partition.test(fold), 2:19);
        Ytest = y2(partition.test(fold), :);
        
        Ytest2 = myData(partition.test(fold), 1);
        
        [XtrainNormal,mu,sigma] = zscore(Xtrain);
        XtestNormal = (Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);
                
        net = feedforwardnet(36);
        
        net.trainParam.epochs = 100;

        net = train(net, XtrainNormal', Ytrain');

        yest = sim(net, XtestNormal');
        yest = yest';

        [~,Ydeco] = max(yest,[],2);

        yval = Ytest2;
        
        confusionMatrix = zeros(numClasses, numClasses);
        for i = 1 : size(Xtest, 1)
            confusionMatrix(Ydeco(i),Ytest2(i)) = confusionMatrix(Ydeco(i), Ytest2(i)) + 1;
        end
        efficiencyTest(fold) = sum(diag(confusionMatrix)) / sum(sum(confusionMatrix));
        
        %plot(confusionMatrix(:,1), confusionMatrix(:,2));
        
    
        %disp(sum(ydeco==yval)/length(ydeco));
    end
    
    impixelregion(imagesc(confusionMatrix));
    
    efficiency = mean(efficiencyTest);
    IC = std(efficiencyTest);
    text=['La eficiencia obtenida fue = ', num2str(efficiency),' +- ',num2str(IC)];
    disp(text);
    
%    
%     for fold=1:repetitiones
%        
%        rng('default');
%        partition = cvpartition(N,'Kfold',repetitiones);
%         
%        indices = partition.training(fold); 
% 
%         Xtrain = myData(partition.training(fold), 2:19);
%         Ytrain = myData(partition.training(fold), 1);
% 
%         Xtest = myData(partition.test(fold), 2:19);
%         Ytest = myData(partition.test(fold), 1);   
%         
%             
%         [Xtrain,mu,sigma] = zscore(Xtrain);
%         Xtest = Normalize(Xtest,mu,sigma);
%     
%     end
  
elseif method == 4 % GMM
    
   Mezclas=5;
   
   for fold=1:repetitiones

        %%% Se hace la partición de las muestras %%%
        %%%      de entrenamiento y prueba       %%%
        
        rng('default');
        partition = cvpartition(N,'Kfold',repetitiones);
        
        indices = partition.training(fold); 

        Xtrain = myData(partition.training(fold), 2:19);
        Ytrain = myData(partition.training(fold), 1);

        Xtest = myData(partition.test(fold), 2:19);
        Ytest = myData(partition.test(fold), 1);   
        
        [Xtrain,mu,sigma] = zscore(Xtrain);
        Xtest = (Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);
        
        vInd=(Ytrain == 1);
        XtrainC1 = Xtrain(vInd,:);
        if ~isempty(XtrainC1)
            Modelo1=entrenarGMM(XtrainC1,Mezclas);
        else
            error('No hay muestras de todas las clases para el entrenamiento');
        end
        
        vInd=(Ytrain == 2);
        XtrainC2 = Xtrain(vInd,:);
        if ~isempty(XtrainC2)
            Modelo2=entrenarGMM(XtrainC2,Mezclas);
        else
            error('No hay muestras de todas las clases para el entrenamiento');
        end
        
        vInd=(Ytrain == 3);
        XtrainC3 = Xtrain(vInd,:);
        if ~isempty(XtrainC3)
            Modelo3=entrenarGMM(XtrainC3,Mezclas);
        else
            error('No hay muestras de todas las clases para el entrenamiento');
        end
        
        vInd=(Ytrain == 4);
        XtrainC4 = Xtrain(vInd,:);
        if ~isempty(XtrainC4)
            Modelo4=entrenarGMM(XtrainC4,Mezclas);
        else
            error('No hay muestras de todas las clases para el entrenamiento');
        end
        
        vInd=(Ytrain == 5);
        XtrainC5 = Xtrain(vInd,:);
        if ~isempty(XtrainC5)
            Modelo5=entrenarGMM(XtrainC5,Mezclas);
        else
            error('No hay muestras de todas las clases para el entrenamiento');
        end
        
        vInd=(Ytrain == 6);
        XtrainC6 = Xtrain(vInd,:);
        if ~isempty(XtrainC6)
            Modelo6=entrenarGMM(XtrainC6,Mezclas);
        else
            error('No hay muestras de todas las clases para el entrenamiento');
        end
        
        vInd=(Ytrain == 7);
        XtrainC7 = Xtrain(vInd,:);
        if ~isempty(XtrainC7)
            Modelo7=entrenarGMM(XtrainC7,Mezclas);
        else
            error('No hay muestras de todas las clases para el entrenamiento');
        end
        
        probClase1=testGMM(Modelo1,Xtest);
        probClase2=testGMM(Modelo2,Xtest);
        probClase3=testGMM(Modelo3,Xtest);
        probClase4=testGMM(Modelo4,Xtest);
        probClase5=testGMM(Modelo5,Xtest);
        probClase6=testGMM(Modelo6,Xtest);
        probClase7=testGMM(Modelo7,Xtest);
        
        Matriz=[probClase1,probClase2,probClase3,probClase4,probClase5,probClase6,probClase7];
        
        [~,Yest] = max(Matriz,[],2);
        
        confusionMatrix = zeros(numClasses, numClasses);
        for i = 1 : size(Xtest, 1)
            confusionMatrix(Yest(i),Ytest(i)) = confusionMatrix(Yest(i),Ytest(i)) + 1;
        end
        efficiencyTest(fold) = sum(diag(confusionMatrix)) / sum(sum(confusionMatrix));
        
   end
   
   impixelregion(imagesc(confusionMatrix));
    
   efficiency = mean(efficiencyTest);
   IC = std(efficiencyTest);
   text=['(GMM) La eficiencia obtenida fue = ', num2str(efficiency),' +- ',num2str(IC)];
   disp(text);
   
elseif method == 5 % SVM
       
        %SFS
%     myData(:,6) = []; %caracteristica 5
%     myData(:,6) = []; %caracteristica 6
%     myData(:,6) = []; %caracteristica 7
%     myData(:,6) = []; %caracteristica 8
%     myData(:,6) = []; %caracteristica 9
%     myData(:,9) = []; %caracteristica 13

   boxConstraint=10;
   gamma=1;
   tipoKernel = 0;
   
   for fold=1:repetitiones

        %%% Se hace la partición de las muestras %%%
        %%%      de entrenamiento y prueba       %%%
        
        rng('default');
        partition = cvpartition(N,'Kfold',repetitiones);
        
        indices = partition.training(fold); 

        Xtrain = myData(partition.training(fold), 2:19);
        Ytrain = myData(partition.training(fold), 1);

        Xtest = myData(partition.test(fold), 2:19);
        Ytest = myData(partition.test(fold), 1);   
        
        [Xtrain,mu,sigma] = zscore(Xtrain);
        Xtest = (Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);
        
        %%% Entrenamiento de los modelos. Se usa la metodologia One vs All. %%%
        
        Ytrain1 = Ytrain;
        Ytrain1(Ytrain1~=1) = -1;
        Modelo1 = entrenarSVM(Xtrain, Ytrain1, 'c', boxConstraint, gamma, tipoKernel);
        alpha1 = Modelo1.alpha;
        b1 = Modelo1.b;
        
        Ytrain2 = Ytrain;
        Ytrain2(Ytrain2~=2) = -1;
        Ytrain2(Ytrain2==2) = 1;
        Modelo2 = entrenarSVM(Xtrain, Ytrain2, 'c', boxConstraint, gamma, tipoKernel);
        alpha2 = Modelo2.alpha;
        b2 = Modelo2.b;
                
        Ytrain3 = Ytrain;
        Ytrain3(Ytrain3~=3) = -1;
        Ytrain3(Ytrain3==3) = 1;
        Modelo3 = entrenarSVM(Xtrain, Ytrain3, 'c', boxConstraint, gamma, tipoKernel);
        alpha3 = Modelo3.alpha;
        b3 = Modelo3.b;
        
        %%% Entrenamiento de los modelos. Se usa la metodologia One vs All. %%%
        
        Ytrain4 = Ytrain;
        Ytrain4(Ytrain4~=4) = -1;
        Ytrain4(Ytrain4==4) = 1;
        Modelo4 = entrenarSVM(Xtrain, Ytrain4, 'c', boxConstraint, gamma, tipoKernel);
        alpha4 = Modelo4.alpha;
        b4 = Modelo4.b;
                
        Ytrain5 = Ytrain;
        Ytrain5(Ytrain5~=5) = -1;
        Ytrain5(Ytrain5==5) = 1;
        Modelo5 = entrenarSVM(Xtrain, Ytrain5, 'c', boxConstraint, gamma, tipoKernel);
        alpha5 = Modelo5.alpha;
        b5 = Modelo5.b;
        
        Ytrain6 = Ytrain;
        Ytrain6(Ytrain6~=6) = -1;
        Ytrain6(Ytrain6==6) = 1;
        Modelo6 = entrenarSVM(Xtrain, Ytrain6, 'c', boxConstraint, gamma, tipoKernel);
        alpha6 = Modelo6.alpha;
        b6 = Modelo6.b;
        
        Ytrain7 = Ytrain;
        Ytrain7(Ytrain7~=7) = -1;
        Ytrain7(Ytrain7==7) = 1;
        Modelo7 = entrenarSVM(Xtrain, Ytrain7, 'c', boxConstraint, gamma, tipoKernel);
        alpha7 = Modelo7.alpha;
        b7 = Modelo7.b;
        
        [Yest1, YestContinuo1] = testSVM(Modelo1, Xtest);
        [Yest2, YestContinuo2] = testSVM(Modelo2, Xtest);
        [Yest3, YestContinuo3] = testSVM(Modelo3, Xtest);
        [Yest4, YestContinuo4] = testSVM(Modelo4, Xtest);
        [Yest5, YestContinuo5] = testSVM(Modelo5, Xtest);
        [Yest6, YestContinuo6] = testSVM(Modelo6, Xtest);
        [Yest7, YestContinuo7] = testSVM(Modelo7, Xtest);
        
        if(tipoKernel == 1)
            K = kernel_matrix(Xtrain, 'lin_kernel', [], Xtest);
        else
            K = kernel_matrix(Xtrain, 'RBF_kernel', gamma, Xtest);
        end
        
        Ytemp1 = (alpha1'*K + b1)';
        Ytemp2 = (alpha2'*K + b2)';
        Ytemp3 = (alpha3'*K + b3)';
        Ytemp4 = (alpha4'*K + b4)';
        Ytemp5 = (alpha5'*K + b5)';
        Ytemp6 = (alpha6'*K + b6)';
        Ytemp7 = (alpha7'*K + b7)';
        
        YestContinuo = [YestContinuo1, YestContinuo2, YestContinuo3, YestContinuo4, YestContinuo5, YestContinuo6, YestContinuo7];
        
        Ytemp = [Ytemp1, Ytemp2, Ytemp3, Ytemp4, Ytemp5, Ytemp6, Ytemp7];
        
        [~, Yest] = max(YestContinuo, [], 2);
        [~, Yesti] = max(Ytemp, [], 2);
        
        confusionMatrix = zeros(numClasses, numClasses);
        for i = 1 : size(Xtest, 1)
            confusionMatrix(Yest(i),Ytest(i)) = confusionMatrix(Yest(i),Ytest(i)) + 1;
        end
        efficiencyTest(fold) = sum(diag(confusionMatrix)) / sum(sum(confusionMatrix));
        
   end
   
   impixelregion(imagesc(confusionMatrix));
    
   efficiency = mean(efficiencyTest);
   IC = std(efficiencyTest);
   text=['La eficiencia obtenida fue = ', num2str(efficiency),' +- ',num2str(IC)];
   disp(text);
    
   
elseif method == 6 % Pearson
           
    myMatrix = abs(corrcoef(myData));
    lista=zeros(columnas,1);
       
    for i=1:columnas-1
         for j=i:columnas-1
             if myMatrix(i,j)>=0.9 && i~=j
                 lista(j)=1;
                 %Texto=['La Característica #: ',num2str(i),' explica la característica #: ',num2str(j),' un ',num2str(myMatrix(i,j)^2),'%'];
                 Texto=['La Característica #: ',num2str(i),' explica la característica #: ',num2str(j)];
                 disp(Texto);
              end
         end
    end
    
    disp('***Coeficiente de correlación de Pearson***');
    disp('***Características candidatas a ser eliminadas***');
    for i=1:columnas
        if lista(i)==1            
            Texto=['Característica #: ',num2str(i)];
            disp(Texto);
        end
    end
    
    
elseif method == 7 % fisher
    
    disp('***Cociente discriminante de Fisher***');

     F=zeros(columnas,1);
        for i=2:columnas
            mediaI=mean(myData(:, i));
            varianzaI = var(myData(:, i));
            for j=2:columnas
                if i~=j
                    mediaJ=mean(myData(:, j));
                    varianzaJ = var(myData(:, j));
                    F(i)= F(i)+(((mediaI - mediaJ)^2)/ (varianzaI + varianzaJ));
                end
            end
        end
    
    mayor = max(F);
    for i=1:size(F)
        F(i) = F(i)/mayor;
        disp(['#', num2str(i), ' -> ', num2str(F(i))]); 
    end
   
    disp('*** Características candidatas a ser eliminadas ***'); 
    for i=1:size(F)
        if F(i)<=0.5
            disp(['La variable #', num2str(i), ' no presenta una capacidad discriminante']); 
        else
            Texto=['La Característica #: ',num2str(i),' permanece'];
            disp(Texto);
        end
    end
    
    %F
    
    %fishers = FisherFunction(X, Y);
    
    
elseif method == 8 % SFS
            
    fsAcumulado= zeros(columnas-1,1);
    iteraciones=10;
    for i=1:iteraciones
        clc;
        Texto='[';
        for j=1:i
            Texto=strcat(Texto,'*');
        end
        for j=i:iteraciones
            Texto=strcat(Texto,'_');
        end
        Texto=strcat(Texto,']');
        disp(Texto);
        fs = sequentialfs(@myCriterio,myData(:,2:19),myData(:,1));
        fsAcumulado = fsAcumulado + fs';
    end
    disp('***Selección de características***');
    disp('***Wrapper: % de importancia de cada característica***');
    fsAcumulado=fsAcumulado./iteraciones;
    fsAcumulado=fsAcumulado.*100;
    fsAcumulado
    input('\n\n(Presione Enter)');
    clc;
    disp('***Selección de características***');
    disp('***Características candidatas a ser eliminadas***');
    topeMinimo=30;
    for i=1:columnas-1
        if fsAcumulado(i)<topeMinimo            
            Texto=['Característica #: ',num2str(i)];
            disp(Texto);
        end
    end
    
    
    
elseif method == 9 % PCA
    
    tope = 85;
     
     for fold=1:10
        %%% Se hace la partición de las muestras %%%
        %%%      de entrenamiento y prueba       %%%
        rng('default');
        particion=cvpartition(N,'Kfold',10);
        indices=particion.training(fold);
        Xtrain=myData(particion.training(fold),2:19);
        Xtest=myData(particion.test(fold),2:19);
        Ytrain=myData(particion.training(fold),1);
        Ytest=myData(particion.test(fold),1);
                       
        [coefCompPrincipales,scores,covarianzaEigenValores,~,porcentajeVarianzaExplicada,~] = pca(Xtrain);
        
        numVariables = length(covarianzaEigenValores);
        
        numComponentes = 0;
        
        porcentajeVarianzaAcumulada = zeros(numVariables,1);
        puntosUmbral = ones(numVariables,1)*tope;
        ejeComponentes = 1:columnas-1;
        
        for k=1:columnas-1
            
            porcentajeVarianzaAcumulada(k) = sum(porcentajeVarianzaExplicada(1:k));
            
            if (sum(porcentajeVarianzaExplicada(1:k)) >= tope) && (numComponentes == 0)
                numComponentes = k;
            end
        end
        
        aux = Xtrain*coefCompPrincipales;
        Xtrain = aux(:,1:numComponentes);
        
        aux = Xtest*coefCompPrincipales;
        Xtest = aux(:,1:numComponentes);
        
        NumArboles=100;    
        Modelo=TrainForest(NumArboles,Xtrain,Ytrain);    
        Yest = TestForest(Modelo,Xtest);
        
        
% Inicio SVM
%         boxConstraint=10;
%         gamma=1;
%         tipoKernel = 0;
%         
%         Ytrain1 = Ytrain;
%         Ytrain1(Ytrain1~=1) = -1;
%         Modelo1 = entrenarSVM(Xtrain, Ytrain1, 'c', boxConstraint, gamma, tipoKernel);
%         alpha1 = Modelo1.alpha;
%         b1 = Modelo1.b;
%         
%         Ytrain2 = Ytrain;
%         Ytrain2(Ytrain2~=2) = -1;
%         Ytrain2(Ytrain2==2) = 1;
%         Modelo2 = entrenarSVM(Xtrain, Ytrain2, 'c', boxConstraint, gamma, tipoKernel);
%         alpha2 = Modelo2.alpha;
%         b2 = Modelo2.b;
%                 
%         Ytrain3 = Ytrain;
%         Ytrain3(Ytrain3~=3) = -1;
%         Ytrain3(Ytrain3==3) = 1;
%         Modelo3 = entrenarSVM(Xtrain, Ytrain3, 'c', boxConstraint, gamma, tipoKernel);
%         alpha3 = Modelo3.alpha;
%         b3 = Modelo3.b;
%         
%         %%% Entrenamiento de los modelos. Se usa la metodologia One vs All. %%%
%         
%         Ytrain4 = Ytrain;
%         Ytrain4(Ytrain4~=4) = -1;
%         Ytrain4(Ytrain4==4) = 1;
%         Modelo4 = entrenarSVM(Xtrain, Ytrain4, 'c', boxConstraint, gamma, tipoKernel);
%         alpha4 = Modelo4.alpha;
%         b4 = Modelo4.b;
%                 
%         Ytrain5 = Ytrain;
%         Ytrain5(Ytrain5~=5) = -1;
%         Ytrain5(Ytrain5==5) = 1;
%         Modelo5 = entrenarSVM(Xtrain, Ytrain5, 'c', boxConstraint, gamma, tipoKernel);
%         alpha5 = Modelo5.alpha;
%         b5 = Modelo5.b;
%         
%         Ytrain6 = Ytrain;
%         Ytrain6(Ytrain6~=6) = -1;
%         Ytrain6(Ytrain6==6) = 1;
%         Modelo6 = entrenarSVM(Xtrain, Ytrain6, 'c', boxConstraint, gamma, tipoKernel);
%         alpha6 = Modelo6.alpha;
%         b6 = Modelo6.b;
%         
%         Ytrain7 = Ytrain;
%         Ytrain7(Ytrain7~=7) = -1;
%         Ytrain7(Ytrain7==7) = 1;
%         Modelo7 = entrenarSVM(Xtrain, Ytrain7, 'c', boxConstraint, gamma, tipoKernel);
%         alpha7 = Modelo7.alpha;
%         b7 = Modelo7.b;
%         
%         [Yest1, YestContinuo1] = testSVM(Modelo1, Xtest);
%         [Yest2, YestContinuo2] = testSVM(Modelo2, Xtest);
%         [Yest3, YestContinuo3] = testSVM(Modelo3, Xtest);
%         [Yest4, YestContinuo4] = testSVM(Modelo4, Xtest);
%         [Yest5, YestContinuo5] = testSVM(Modelo5, Xtest);
%         [Yest6, YestContinuo6] = testSVM(Modelo6, Xtest);
%         [Yest7, YestContinuo7] = testSVM(Modelo7, Xtest);
%         
%         if(tipoKernel == 1)
%             K = kernel_matrix(Xtrain, 'lin_kernel', [], Xtest);
%         else
%             K = kernel_matrix(Xtrain, 'RBF_kernel', gamma, Xtest);
%         end
%         
%         Ytemp1 = (alpha1'*K + b1)';
%         Ytemp2 = (alpha2'*K + b2)';
%         Ytemp3 = (alpha3'*K + b3)';
%         Ytemp4 = (alpha4'*K + b4)';
%         Ytemp5 = (alpha5'*K + b5)';
%         Ytemp6 = (alpha6'*K + b6)';
%         Ytemp7 = (alpha7'*K + b7)';
%         
%         YestContinuo = [YestContinuo1, YestContinuo2, YestContinuo3, YestContinuo4, YestContinuo5, YestContinuo6, YestContinuo7];
%         
%         Ytemp = [Ytemp1, Ytemp2, Ytemp3, Ytemp4, Ytemp5, Ytemp6, Ytemp7];
%         
%         [~, Yest] = max(YestContinuo, [], 2);
%         [~, Yesti] = max(Ytemp, [], 2);
% FIN SVM
        
% Inicio RNA
%         [~, loc] = ismember(myData(:, 1), unique(myData(:, 1)));
%     
%         y_one_hot  = ind2vec(loc');
%     
%         y2 = full(y_one_hot);
%         y_one_hot = y_one_hot';
%         y2 = y2';    
%         
%         Ytest2 = myData(particion.test(fold), 1);
%         
%         [XtrainNormal,mu,sigma] = zscore(Xtrain);
%         XtestNormal = (Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);
%                 
%         net = feedforwardnet(36);
%         
%         net.trainParam.epochs = 100;
% 
%         net = train(net, XtrainNormal', Ytrain');
% 
%         yest = sim(net, XtestNormal');
%         yest = yest';
% 
%         [~,Ydeco] = max(yest,[],2);
% 
%         yval = Ytest2;
%         
%         confusionMatrix = zeros(numClasses, numClasses);
%         for i = 1 : size(Xtest, 1)
%             confusionMatrix(Ydeco(i),Ytest2(i)) = confusionMatrix(Ydeco(i), Ytest2(i)) + 1;
%         end
%         efficiencyTest(fold) = sum(diag(confusionMatrix)) / sum(sum(confusionMatrix));
% FIN RNA

        confusionMatrix = zeros(numClasses, numClasses);
        for i = 1 : size(Xtest, 1)
            confusionMatrix(Yest(i),Ytest(i)) = confusionMatrix(Yest(i),Ytest(i)) + 1;
        end
        efficiencyTest(fold) = sum(diag(confusionMatrix)) / sum(sum(confusionMatrix));
    end
    
    Eficiencia = mean(efficiencyTest);
    IC = std(efficiencyTest);
    Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(IC)];
    disp(Texto);
    
elseif method == 10 % Funciones disciminantes gaussianas
      
   disp('*** Funciones discriminantes (classify) ***');
   
   %Bootstraping
    porcentaje=N*0.7;
    rng('default');
    ind=randperm(N); %%% Se seleccionan los indices de forma aleatoria
    
    Xtrain = myData(ind(1:porcentaje), 2:19);
    Ytrain = myData(ind(1:porcentaje), 1);

    Xtest = myData(ind(porcentaje+1:end), 2:19);
    Ytest = myData(ind(porcentaje+1:end), 1);   
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%% Normalización %%%
    
    [Xtrain,mu,sigma]=zscore(Xtrain);
    Xtest=Normalize(Xtest,mu,sigma);
    
    %%%%%%%%%%%%%%%%%%%%%
    
    clasificacion = classify(Xtest, Xtrain, Ytrain);
      
    Eficiencia=(sum(clasificacion==Ytest))/length(Ytest);
    Error=1-Eficiencia;
    
    Texto=strcat('La eficiencia en prueba es: ',{' '},num2str(Eficiencia));
    disp(Texto);
 
end