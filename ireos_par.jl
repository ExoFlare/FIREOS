module IreosPar

using Base.Threads
using Distributed
using Random
using ScikitLearn
using Distances
using ThreadSafeDicts
using LIBSVM
using LIBLINEAR
using DecisionTree
using XGBoost

@sk_import linear_model: LogisticRegression
@sk_import svm: SVC

const VERSION = "0.0.4"
const INLIER_CLASS = -1
const OUTLIER_CLASS = 1
const MAX_RECURSION_DEPTH = 3

const l = ReentrantLock()

const SEED = 123
Random.seed!(SEED)

#=struct IREOSSample
    index::UInt8
    separabilities::ImmutableDict
    num_samples::UInt8
    y::Vector{Float64}
    current_sample::Vector{Float64}
    https://visualstudiomagazine.com/articles/2020/04/29/logistic-regression.aspx
    https://www.thekerneltrip.com/machine/learning/computational-complexity-learning-algorithms/
end

struct IREOSData
    samples::Vector{IREOSSample}
    clf
    current_recursion_depth::UInt8
end=#

function ireos_par(X::AbstractMatrix{<:Number}, clf::String, gamma_min::Float64, gamma_max::Float64, tol::Float64)
    num_samples = size(X)[1]
    aucs = Vector{Float64}(undef, num_samples)
    @assert num_samples == size(X)[1]
    @info "Started Parallel IREOS with dataset of size:", size(X), " gamma_min: $gamma_min, gamma_max: $gamma_max, tol: $tol, classifier: $clf, max_recursion_depth: $MAX_RECURSION_DEPTH on ", nthreads(), "threads"
    clf_func = get_classifier_function_par(clf)
    T = Dict{Float64, AbstractMatrix{<:Number}}()
    Threads.@threads for i in 1:num_samples
        seperabilities = Dict{Float64, Float64}()
        y = fill(INLIER_CLASS, num_samples)
        y[i] = OUTLIER_CLASS
        @debug "Started IREOS calculation of sample number: $i"
        aucs[i] = adaptive_quads_par(X, y, i, gamma_min, gamma_max, tol, clf_func, seperabilities, T)
        @debug "IREOS calculation of sample number: $i successful"
        y[i] = INLIER_CLASS
    end
    return aucs
end

function adaptive_quads_par(X, y, outlier_index, a, b, tol, clf_func, seperabilities, T, current_recursion_depth = 0)
    m = (a + b) / 2
    err_all = simpson_rule_par(X, y, outlier_index, a, b, clf_func, seperabilities, T)
    err_new = simpson_rule_par(X, y, outlier_index, a, m, clf_func, seperabilities, T) + simpson_rule_par(X, y, outlier_index, m, b, clf_func, seperabilities, T)
    calculated_error = (err_all - err_new) / 15
    if tol < calculated_error && current_recursion_depth < MAX_RECURSION_DEPTH
        @debug "Iteration depth: $current_recursion_depth. Criterion not reached: $calculated_error > $tol"
        return adaptive_quads_par(X, y, outlier_index, a, m, tol / 2, clf_func, seperabilities, T, current_recursion_depth+1) + adaptive_quads_par(X, y, outlier_index, m, b, tol/2, clf_func, seperabilities, T, current_recursion_depth+1)
    else
        @debug "Termination criterion of $calculated_error < $tol OR Recursion depth $current_recursion_depth / $MAX_RECURSION_DEPTH reached."
        return err_new
    end
end

function simpson_rule_par(X, y, outlier_index, a, b, clf_func, seperabilities, T)
    h = (b - a) / 2
    for i in [a, a+h, b]
        if i in keys(seperabilities)
            continue
        end
        p_outlier = clf_func(X, y, outlier_index, i, T)
        @debug "$outlier_index: gamma $i : p-value: $p_outlier"
        seperabilities[i] = p_outlier
    end
    return (h / 3) * (seperabilities[a] + 4 * seperabilities[a + h] + seperabilities[b])
end

function get_classifier_function_par(clf::String)
    if clf == "svc"
        return get_svm_clf_par
    elseif clf == "logreg"
        return get_logreg_clf_par
    elseif clf == "klr"
        return get_klr_clf_par
    elseif clf == "libsvm"
        return get_libsvm_par
    elseif clf == "liblinear"
        return get_liblinear_par
    elseif clf == "decision_tree_native"
        return get_decision_tree_native_par
    elseif clf == "decision_tree_sklearn"
        return get_decision_tree_sklearn_par
    elseif clf == "random_forest_native"
        return get_random_forest_native_par
    elseif clf == "random_forest_sklearn"
        return get_random_forest_sklearn_par
    elseif clf == "xgboost_tree"
        return get_xgboost_tree_par
    elseif clf == "xgboost_dart"
        return get_xgboost_dart_par
    elseif clf == "xgboost_linear"
        return get_xgboost_linear_par
    else
        @error "Unknown classifier $clf"
    end
end

function get_outlier_prob(clf, current_sample)
    p_index = findfirst(isequal(OUTLIER_CLASS), clf.classes_)
    return predict_proba(clf, current_sample')[p_index]
end

function get_logreg_clf_par(X, y, outlier_index, gamma, T)
    if !haskey(T, gamma)
        @debug "Gamma: $gamma missing.. Calculating R-Matrix"
        T[gamma] = rbf_kernel(X, gamma)
    end
    lock(l)
    outlier_prob = sk_logreg(T, y, outlier_index, gamma)
    unlock(l)
    return outlier_prob
end

function sk_logreg(T, y, outlier_index, gamma)
    clf = LogisticRegression(random_state=SEED, tol=0.0095, max_iter=1000000)
    fit!(clf, T[gamma], y)
    current_sample = reshape(T[gamma][outlier_index, :] , (size(T[gamma])[2],1))
    return get_outlier_prob(clf, current_sample)
end

function get_svm_clf_par(X, y, outlier_index, gamma, T)
    #SVC cannot deal with gamma == 0
    if gamma == 0.0
        gamma = 0.0001
    end
    @debug "Thread", threadid(), "Gamma: $gamma, X: ", size(X), "y: ", size(y), "Outlier-Index:", findfirst(isequal(OUTLIER_CLASS), y)
    current_sample = reshape(X[outlier_index, :] , (size(X)[2],1))
    # lock(func, lock) did not work
    lock(l)
    outlier_prob = sk_svm(X, y, current_sample, gamma)
    unlock(l)
    return outlier_prob
end

function sk_svm(X, y, current_sample, gamma)
    clf = SVC(gamma=gamma, class_weight="balanced", probability=true, C=100, random_state=SEED, tol=0.0095, max_iter=-1)
    fit!(clf, X, y)
    return get_outlier_prob(clf, current_sample)
end

function get_klr_clf_par(X, y, outlier_index, gamma, T)
    if !haskey(T, gamma)
        @debug "Gamma: $gamma missing.. Calculating R-Matrix"
        T[gamma] = rbf_kernel(X, gamma)
    end
    lock(l)
    outlier_prob = sk_klr(T, y, outlier_index, gamma)
    unlock(l)
    return outlier_prob
end

function sk_klr(T, y, outlier_index, gamma)
    clf = LogisticRegression(class_weight="balanced", tol=0.0095, solver = "saga", C=100, max_iter=1000000, random_state=SEED)
    fit!(clf, T[gamma], y)
    current_sample = reshape(T[gamma][outlier_index, :] , (size(T[gamma])[2],1))
    return get_outlier_prob(clf, current_sample)
end

function get_libsvm_par(X, y, outlier_index, gamma, T)
    clf = svmtrain(X', y, gamma=gamma, probability=true, tolerance=tol=0.0095, cost=100.0)
    current_sample = reshape(X[outlier_index, :] , (size(X)[2],1))
    p_index = findfirst(isequal(OUTLIER_CLASS), clf.labels)
    return svmpredict(clf, current_sample)[2][p_index]
end

"""
function for predicting probabilities using liblinear
...
# Arguments
- `X`: feature matrix of input data
- `y`: target vector of input matrix
- `outlier_index`: index of current outlier
- `gamma`: not used for linear predictors
- `T`: map precomputed rbf kernel matrix of given gammas
......` 
returns probability that outlier sample is classified as outlier
"""
function get_liblinear_par(X, y, outlier_index, gamma, T)
    # First dimension of input data is features; second is instances
    model = linear_train(y, X', solver_type=Cint(7), verbose=false);
    current_sample = reshape(X[outlier_index, :] , (size(X)[2],1))
    (predicted_labels, decision_values) = linear_predict(model, current_sample, probability_estimates=true, verbose=false);
    if predicted_labels[1] == OUTLIER_CLASS
        return decision_values[1]
    elseif predicted_labels[1] == INLIER_CLASS
        return 1 - decision_values[1]
    else
        d = decision_values[1]
        @error "Invalid decision label $d"
    end
end

"""
function for predicting probabilities using native decision trees
...
# Arguments
- `X`: feature matrix of input data
- `y`: target vector of input matrix
- `outlier_index`: index of current outlier
- `gamma`: not used in decision trees
- `T`: not used in decision trees
......` 
returns probability that outlier sample is classified as outlier
"""
function get_decision_tree_native_par(X, y, outlier_index, gamma, T)
    current_sample = reshape(X[outlier_index, :] , (1,size(X)[2]))
    n_subfeatures=0
    #max_depth=2
    min_samples_leaf=1
    min_samples_split=2
    min_purity_increase=0.0
    #model = build_forest(y, X)
    probs = 0.0
    max_depth = 1000
    iteration = 1

    for depth in 1:max_depth
        model = build_tree(y, X, n_subfeatures, depth, min_samples_leaf, min_samples_split, min_purity_increase, rng = SEED)
        p_outlier = apply_tree_proba(model, current_sample, [OUTLIER_CLASS, INLIER_CLASS])[1]
        probs += p_outlier
        if p_outlier == 1.0
            break
        end
        iteration += 1
    end
    return probs / iteration
end

"""
function for predicting probabilities using decision trees in sklearn
...
# Arguments
- `X`: feature matrix of input data
- `y`: target vector of input matrix
- `outlier_index`: index of current outlier
- `gamma`: not used for decision trees
- `T`: not used for decision trees
......` 
returns probability that outlier sample is classified as outlier
"""
function get_decision_tree_sklearn_par(X, y, outlier_index, gamma, T)
    model = DecisionTreeClassifier(max_depth=2)
    DecisionTree.fit!(model, X, y)
    # pretty print of the tree, to a depth of 5 nodes (optional)
    #print_tree(model, 5)
    # apply learned model
    # get the probability of each label
    current_sample = reshape(X[outlier_index, :] , (1, size(X)[2]))
    p_index = findfirst(isequal(OUTLIER_CLASS), model.classes)
    predict_proba(model, current_sample)[p_index]
end

"""
function for predicting probabilities using native random forest
...
# Arguments
- `X`: feature matrix of input data
- `y`: target vector of input matrix
- `outlier_index`: index of current outlier
- `gamma`: not used for random forest
- `T`: not used for random forest
......` 
returns probability that outlier sample is classified as outlier
"""
function get_random_forest_native_par(X, y, outlier_index, gamma, T)
    current_sample = reshape(X[outlier_index, :] , (1,size(X)[2]))
    model = build_forest(y, X, rng=SEED)
    p_outlier = apply_forest_proba(model, current_sample, [OUTLIER_CLASS, INLIER_CLASS])[1]
    return p_outlier
end

"""
function for predicting probabilities using random forest in sklearn
...
# Arguments
- `X`: feature matrix of input data
- `y`: target vector of input matrix
- `outlier_index`: index of current outlier
- `gamma`: not used for random forest
- `T`: not used for random forest
......` 
returns probability that outlier sample is classified as outlier
"""
function get_random_forest_sklearn_par(X, y, outlier_index, gamma, T)
    model = RandomForestClassifier(n_subfeatures=-1, n_trees=10, partial_sampling=0.7, max_depth=2, min_samples_leaf=1, min_samples_split=2, min_purity_increase=0.0, rng=SEED)
    DecisionTree.fit!(model, X, y)
    # pretty print of the tree, to a depth of 5 nodes (optional)
    #print_tree(model, 5)
    # apply learned model
    # get the probability of each label
    current_sample = reshape(X[outlier_index, :] , (1,size(X)[2]))
    p_index = findfirst(isequal(OUTLIER_CLASS), model.classes)
    predict_proba(model, current_sample)[p_index]
end

"""
function for predicting probabilities using native XGBoost with tree booster
...
# Arguments
- `X`: feature matrix of input data
- `y`: target vector of input matrix
- `outlier_index`: index of current outlier
- `gamma`: not used for xgboost
- `T`: not used for xgboost
......` 
returns probability that outlier sample is classified as outlier
"""
function get_xgboost_tree_par(X, y, outlier_index, gamma, T)
    # xgboost cannot deal with values outsode [0,1) -> tempoarily transform negative labels
    replace!(y, INLIER_CLASS => 0)
    # num rounds act as number of estimators in rf
    num_rounds = 1
    # Important! xgboost uses Matrix of size (1,k) for getting one probability output, which is correct
    # A Matrix of (k, 1) on the other hand does NOT result in an error! The output is 9 probabilities for one sample, which is NOT correct!
    current_sample = reshape(X[outlier_index, :] , (1,size(X)[2]))
    # softprob transforms result into probability for multiclass classification
    # based on same principle as logistic regression -> is a generalization
    # https://stackoverflow.com/questions/36051506/difference-between-logistic-regression-and-softmax-regression
    # model = xgboost(X, num_rounds, label=y, max_depth=2, seed=SEED, objective="multi:softprob", num_class=2, silent=1, validate_parameters=true)
    # print(XGBoost.predict(model, current_sample))
    # equals reg:logistic
    # random forest parameter setting https://xgboost.readthedocs.io/en/stable/tutorials/rf.html
    model = xgboost(X, num_rounds, label=y, max_depth=2, seed=SEED, objective="binary:logistic", silent=true, eta = 1, subsample=0.8, num_parallel_tree=10, num_boost_round=3)
    # For binary classification, the output predictions are probability confidence scores in [0,1], corresponds to the probability of the label to be positive.
    # Outlier prediction acts as success event a binomial trial scenario
    return XGBoost.predict(model, current_sample)[1]
end

"""
function for predicting probabilities using native XGBoost with dart booster
https://xgboost.readthedocs.io/en/latest/tutorials/dart.html
...
# Arguments
- `X`: feature matrix of input data
- `y`: target vector of input matrix
- `outlier_index`: index of current outlier
- `gamma`: not used for xgboost
- `T`: not used for xgboost
......` 
returns probability that outlier sample is classified as outlier
"""
function get_xgboost_dart_par(X, y, outlier_index, gamma, T)
    # xgboost cannot deal with values outsode [0,1) -> tempoarily transform negative labels
    replace!(y, INLIER_CLASS => 0)
    # num rounds act as number of estimators in rf
    num_rounds = 10
    current_sample = reshape(X[outlier_index, :] , (1,size(X)[2]))
    model = xgboost(X, num_rounds, label=y, booster="dart", max_depth=2, seed=SEED, objective="binary:logistic", silent=1)
    return XGBoost.predict(model, current_sample)[1]
end

"""
function for predicting probabilities using native XGBoost with linear booster
...
# Arguments
- `X`: feature matrix of input data
- `y`: target vector of input matrix
- `outlier_index`: index of current outlier
- `gamma`: not used for xgboost
- `T`: not used for xgboost
......` 
returns probability that outlier sample is classified as outlier
"""
function get_xgboost_linear_par(X, y, outlier_index, gamma, T)
    # xgboost cannot deal with values outsode [0,1) -> tempoarily transform negative labels
    replace!(y, INLIER_CLASS => 0)
    # num rounds act as number of estimators in rf
    num_rounds = 10
    current_sample = reshape(X[outlier_index, :] , (1,size(X)[2]))
    model = xgboost(X, num_rounds, label=y, booster="gblinear", max_depth=2, seed=SEED, objective="binary:logistic", silent=1)
    return XGBoost.predict(model, current_sample)[1]
end

function evaluate_solutions_par(ireos, solutions, gamma_min, gamma_max)
    if isnothing(solutions)
        return nothing
    end
    @info "Started IREOS Evaluation:"
    results = ThreadSafeDict{Int, Float64}()
    Threads.@threads for i in 1:size(solutions)[1]
        results[i] = evaluate_solution(ireos, solutions[i,:], gamma_min, gamma_max)
    end
    return results
end

evaluate_solution(ireos, solution, gamma_min, gamma_max) = sum(ireos .* solution) / sum(solution) / (gamma_max - gamma_min)

# radial basis function: K(x, y) = exp(-gamma ||x-y||^2)
rbf_kernel(X, gamma) = exp.(-gamma * pairwise(SqEuclidean(), X, dims=1))

println("Package IreosPar loaded")

end
