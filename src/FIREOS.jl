__precompile__()

module FIREOS

using Base.Threads
using DecisionTree
using Distances
using LIBLINEAR
using LIBSVM
using Random
using ScikitLearn
using StatsBase
using ThreadSafeDicts
using XGBoost

export fireos, fireos_par, evaluate_solutions, evaluate_solutions_par, normalize_solutions!, normalize_solutions_par!

@sk_import linear_model: LogisticRegression
@sk_import svm: SVC

global const VERSION = "0.1.0"

#Inlier and outlier labels
global const INLIER_CLASS = -1
global const OUTLIER_CLASS = 1

#maximum recursion depth for adaptive quadrature
global const MAX_RECURSION_DEPTH = 3

#fixed seed for experiments being reproducable
global const SEED = 123
Random.seed!(SEED)

include("fireos_lib.jl")
include("fireos_lib_par.jl")
include("fireos_common.jl")

# precompile hints
# future todo
precompile(fireos, (Matrix{Float64}, String, ))
precompile(fireos_par, (Matrix{Float64}, String, ))

end # module FIREOS