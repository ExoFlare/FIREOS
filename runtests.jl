include("ireos.jl")
using Main.Ireos

include("ireos_par.jl")
using Main.IreosPar

using Base.Threads
using DelimitedFiles
using Logging
using BenchmarkTools
using Distances
using Test

const ITERATIONS = 10
const WINDOW_RATIO = 1.0
const TOLERANCE = 0.005
const ADAPTIVE_QUADS_ENABLED = false
const GAMMA_MAX = 1.0
const GAMMA_MIN = 0.0

#const clfs = ["svc", "logreg", "klr", "libsvm", "decision_tree_native", "decision_tree_sklearn", "random_forest_native", "random_forest_sklearn", "liblinear", "xgboost_tree", "xgboost_dart", "xgboost_linear"]

@testset "FIREOS" begin

function test_seq_and_par_equals()
    X = rand(50,5)
    solutions = rand(50, 2)
    # libsvm and liblinear do not feature fixed seeds
    clfs = Vector{String}(["svc", "logreg", "klr", "decision_tree_native", "decision_tree_sklearn", "random_forest_native", "random_forest_sklearn", "xgboost_tree", "xgboost_dart", "xgboost_linear"])
    for clf in clfs
        trained_seq = Ireos.fireos(X, clf)
        result_seq = Ireos.evaluate_solutions(trained_seq, solutions', GAMMA_MIN, GAMMA_MAX)
        trained_par = IreosPar.fireos(X, clf)
        result_par = IreosPar.evaluate_solutions(trained_par, solutions', GAMMA_MIN, GAMMA_MAX)
        @test result_seq == result_par
    end
end

function test_fixed_seed()
    X = rand(50,5)
    solutions = rand(50, 2)
    # libsvm and liblinear do not feature fixed seeds
    clfs = Vector{String}(["svc", "logreg", "klr", "decision_tree_native", "decision_tree_sklearn", "random_forest_native", "random_forest_sklearn", "xgboost_tree", "xgboost_dart", "xgboost_linear"])
    for clf in clfs
        trained_seq1 = Ireos.fireos(X, clf)
        result_seq1 = Ireos.evaluate_solutions(trained_seq1, solutions', GAMMA_MIN, GAMMA_MAX)
        trained_seq2 = Ireos.fireos(X, clf)
        result_seq2 = Ireos.evaluate_solutions(trained_seq2, solutions', GAMMA_MIN, GAMMA_MAX)

        trained_par1 = IreosPar.fireos(X, clf)
        result_par1 = IreosPar.evaluate_solutions(trained_par1, solutions', GAMMA_MIN, GAMMA_MAX)
        trained_par2 = IreosPar.fireos(X, clf)
        result_par2 = IreosPar.evaluate_solutions(trained_par2, solutions', GAMMA_MIN, GAMMA_MAX)
        @test result_seq1 == result_seq2
        @test result_par1 == result_par2
    end
end

@testset "unit_tests" begin
    @info "unit tests start"
    test_seq_and_par_equals()
    test_fixed_seed()
end

end