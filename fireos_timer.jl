include("FIREOS.jl")
using Main.FIREOS

using Base.Threads
using DelimitedFiles
using Logging
using Distances
using Test


function time()
    Logging.disable_logging(LogLevel(1000))
    X = rand(2000,5)
    solutions = rand(1000, 2000)
    # libsvm and liblinear do not feature fixed seeds
    #clfs = Vector{String}(["svc", "logreg", "klr", "decision_tree_native", "decision_tree_sklearn", "random_forest_native", "random_forest_sklearn", "xgboost_tree", "xgboost_dart", "xgboost_linear"])
    clfs = Vector{String}(["liblinear", "xgboost_linear", "random_forest_native"])
    for i in [1,2,3]
        for clf in clfs
            println("Sequential: $clf")
            @time trained_seq = FIREOS.fireos(X, clf)
            @time FIREOS.evaluate_solutions(trained_seq, solutions, 0.0, 1.0)
            println("Parallel: $clf")
            @time trained_par = FIREOS.fireos_par(X, clf)
            @time FIREOS.evaluate_solutions_par(trained_par, solutions, 0.0, 1.0)
        end
    end
end

time()