function error= myCriterio( Xtrain , Ytrain , Xtest , Ytest )
   
    Yesti = classify ( Xtest , Xtrain , Ytrain );
    
%     NumArboles=100;    
%     Modelo=TrainForest(NumArboles,Xtrain,Ytrain);    
%     Yesti = TestForest(Modelo,Xtest);
%         
    error = sum( Ytest ~= Yesti ) / length( Yesti ) ;
    
end