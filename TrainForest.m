function model = TrainForest(numTrees, X, Y)

    model = TreeBagger(numTrees, X, Y); 

end