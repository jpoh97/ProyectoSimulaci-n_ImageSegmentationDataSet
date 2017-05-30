function modelo = entrenarGMM(X,NumeroMezclas)

    inputDim=size(X,2);      %%%%% Numero de caracteristicas de las muestras
    mezclas = gmm(inputDim, NumeroMezclas, 'diag');
    options = foptions;
    options(14)=10;
    options(5)=10;
    mezclas = gmminit(mezclas, X, options);
    modelo = gmmem(mezclas, X, options);
    

end