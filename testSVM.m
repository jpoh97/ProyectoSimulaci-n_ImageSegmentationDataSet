function [Ytest, YestContinuo] = testSVM(Modelo,Xtest)

    [Ytest, YestContinuo] = simlssvm(Modelo,Xtest);
end