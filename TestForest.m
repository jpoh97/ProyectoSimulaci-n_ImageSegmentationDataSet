function output = TestForest(Modelo,X)

    output = predict(Modelo,X);
    output = str2double(output);
    
end