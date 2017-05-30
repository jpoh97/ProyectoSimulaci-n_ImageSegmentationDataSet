function Modelo = entrenarSVM(X,Y,tipo,boxConstraint,sigma, tipoKernel)
    
    if (tipoKernel == 1) 
        Modelo = trainlssvm({X,Y, tipo, boxConstraint, [], 'lin_kernel'});
    else
        Modelo = trainlssvm({X,Y, tipo, boxConstraint, sigma, 'RBF_kernel'});
    end

end