module Ireos

using StatsBase
using Base.Threads
using Distributed
using ScikitLearn
using Distances
using LIBSVM
using LIBLINEAR
using DecisionTree

@sk_import linear_model: LogisticRegression
@sk_import svm: SVC

const VERSION = "0.0.5"
const INLIER_CLASS = -1
const OUTLIER_CLASS = 1
const MAX_RECURSION_DEPTH = 3

const SEED = 123



#=struct IREOSSample
    index::UInt8
    separabilities::ImmutableDict
    num_samples::UInt8
    y::Vector{Float64}
    current_sample::Vector{Float64}
end

struct IREOSData
    samples::Vector{IREOSSample}
    clf
    current_recursion_depth::UInt8
end=#

function ireos(X::AbstractMatrix{<:Number}, clf::String, gamma_min::Float64, gamma_max::Float64, tol::Float64, window_size::Union{Int32, Nothing}=nothing, adaptive_quads_enabled::Bool=true)
    window_mode = false
    num_samples = size(X)[1]
    aucs = Vector{Float64}(undef,num_samples)
    if !isnothing(window_size)
        @assert window_size <= num_samples
        window_mode = true
    end
    @info "Started IREOS with dataset of size:", size(X), "window size: $window_size, gamma_min: $gamma_min, gamma_max: $gamma_max, tol: $tol, classifier: $clf, max_recursion_depth: $MAX_RECURSION_DEPTH, adaptive_quads_enabled: $adaptive_quads_enabled"
    clf_func = get_classifier_function(clf)
    if window_mode
        y = fill(INLIER_CLASS, window_size)
    else
        y = fill(INLIER_CLASS, num_samples)
    end
    T::AbstractDict{Float64, AbstractMatrix{<:Number}} = Dict{Float64, AbstractMatrix{<:Number}}()
    for i in 1:num_samples
        seperabilities = Dict{Float64, Float64}()
        if window_mode
            outlier_index = i <= window_size ? i : window_size
            idx_start = get_start_idx(i, window_size, num_samples)
            idx_end = idx_start + window_size - 1
            data = X[idx_start:idx_end,:]
            y[outlier_index] = OUTLIER_CLASS
            @debug "windowed mode: index_start: $idx_start, index_end: $idx_end"
        else
            outlier_index = i
            data = X
            y[i] = OUTLIER_CLASS
        end
        @debug "Started IREOS calculation of sample number: $i"
        if(adaptive_quads_enabled)
            aucs[i] = adaptive_quads(data, y, outlier_index, gamma_min, gamma_max, tol, clf_func, seperabilities, T)
        else
            aucs[i] = clf_func(data, y, outlier_index, gamma_max, T)
        end
        @debug "IREOS calculation of sample number: $i successful"
        if window_mode
            y[outlier_index] = INLIER_CLASS
        else
            y[i] = INLIER_CLASS
        end
    end
    return aucs
end

function get_start_idx(current_index, window_size, size_all)
    if current_index <= window_size
        return 1
    elseif current_index > size_all - window_size
        return size_all - (window_size - 1)
    else
        return current_index - (window_size - 1)
    end
end

function adaptive_quads(X, y, outlier_index, a, b, tol, clf_func, seperabilities, T, current_recursion_depth = 0)
    m = (a + b) / 2
    err_all = simpson_rule(X, y, outlier_index, a, b, clf_func, seperabilities, T)
    err_new = simpson_rule(X, y, outlier_index, a, m, clf_func, seperabilities, T) + simpson_rule(X, y, outlier_index, m, b, clf_func, seperabilities, T)
    calculated_error = (err_all - err_new) / 15
    if tol < calculated_error && current_recursion_depth < MAX_RECURSION_DEPTH
        @debug "Iteration depth: $current_recursion_depth. Criterion not reached: $calculated_error > $tol"
        return adaptive_quads(X, y, outlier_index, a, m, tol / 2, clf_func, seperabilities, T, current_recursion_depth+1) + adaptive_quads(X, y, outlier_index, m, b, tol/2, clf_func, seperabilities, T, current_recursion_depth+1)
    else
        @debug "Termination criterion of $calculated_error < $tol OR Recursion depth $current_recursion_depth / $MAX_RECURSION_DEPTH reached."
        return err_new
    end
end

function simpson_rule(X, y, outlier_index, a, b, clf_func, seperabilities, T)
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

function get_classifier_function(clf::String)
    if clf == "svc"
        return get_svm_clf
    elseif clf == "logreg"
        return get_logreg_clf
    elseif clf == "klr"
        return get_klr_clf
    elseif clf == "libsvm"
        return get_libsvm
    elseif clf == "liblinear"
        return get_liblinear
    elseif clf == "decision_tree_native"
        return get_decision_tree_native
    elseif clf == "decision_tree_sklearn"
        return get_decision_tree_sklearn
    elseif clf == "random_forest_native"
        return get_random_forest_native
    elseif clf == "random_forest_sklearn"
        return get_random_forest_sklearn
    else
        @error "Unknown classifier $clf"
    end
end

function predict_sk_clf_proba(clf, X, y, gamma, outlier_index)
    fit!(clf, X, y)
    current_sample = reshape(X[outlier_index, :] , (size(X)[2],1))
    p_index = findfirst(isequal(OUTLIER_CLASS), clf.classes_)
    return predict_proba(clf, current_sample')[p_index]
end

function get_logreg_clf(X, y, outlier_index, gamma, T)
    clf = LogisticRegression(random_state=123, tol=0.0095, max_iter=1000000)

    if !haskey(T, gamma)
        @debug "Gamma: $gamma missing.. Calculating R-Matrix"
        T[gamma] = rbf_kernel(X, gamma)
    end
    return predict_sk_clf_proba(clf, T[gamma], y, gamma, outlier_index)
end

function get_svm_clf(X, y, outlier_index, gamma, T)
    #SVC cannot deal with gamma == 0
    if gamma == 0.0
        gamma = 0.0001
    end
    @debug "Thread", threadid(), "Gamma: $gamma, X: ", size(X), "y: ", size(y), "Outlier-Index:", findfirst(isequal(OUTLIER_CLASS), y)
    clf = SVC(gamma=gamma, class_weight="balanced", probability=true, C=100, random_state=123, tol=0.0095, max_iter=-1)
    return predict_sk_clf_proba(clf, X, y, gamma, outlier_index)
end


function get_klr_clf(X, y, outlier_index, gamma, T)
    # param set closest to the paper (liblinear returns non-zero values for gamma = 0)
    clf = LogisticRegression(class_weight="balanced", tol=0.0095, solver = "saga", C=100, max_iter=1000000, random_state=123)
    if !haskey(T, gamma)
        @debug "Gamma: $gamma missing.. Calculating R-Matrix"
        T[gamma] = rbf_kernel(X, gamma)
    end
    #self.ireos_data.set_current_sample(R[self.ireos_data.get_index()].reshape(1, -1))
    return predict_sk_clf_proba(clf, T[gamma], y, gamma, outlier_index)
end

function get_libsvm(X, y, outlier_index, gamma, T)
    clf = svmtrain(X', y, gamma=gamma, probability=true)
    current_sample = reshape(X[outlier_index, :] , (size(X)[2],1))
    p_index = findfirst(isequal(OUTLIER_CLASS), clf.labels)
    return svmpredict(clf, current_sample)[2][p_index]
end

function get_liblinear(X, y, outlier_index, gamma, T)
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

function get_decision_tree_native(X, y, outlier_index, gamma, T)
    current_sample = reshape(X[outlier_index, :] , (size(X)[2],1))
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
        p_outlier = apply_tree_proba(model, current_sample', [OUTLIER_CLASS, INLIER_CLASS])[1]
        probs += p_outlier
        if p_outlier == 1.0
            break
        end
        iteration += 1
    end
    return probs / iteration
end


function get_decision_tree_sklearn(X, y, outlier_index, gamma, T)
    model = DecisionTreeClassifier(max_depth=2)
    DecisionTree.fit!(model, X, y)
    # pretty print of the tree, to a depth of 5 nodes (optional)
    #print_tree(model, 5)
    # apply learned model
    # get the probability of each label
    current_sample = reshape(X[outlier_index, :] , (size(X)[2],1))
    p_index = findfirst(isequal(OUTLIER_CLASS), model.classes)
    predict_proba(model, current_sample')[p_index]
end

function get_random_forest_native(X, y, outlier_index, gamma, T)
    current_sample = reshape(X[outlier_index, :] , (size(X)[2],1))
    model = build_forest(y, X, -1, 10, 0.7, 2, rng=SEED)
    p_outlier = apply_forest_proba(model, current_sample', [OUTLIER_CLASS, INLIER_CLASS])[1]
    return p_outlier
end

function get_random_forest_sklearn(X, y, outlier_index, gamma, T)
    model = RandomForestClassifier(n_subfeatures=-1, n_trees=10, partial_sampling=0.7, max_depth=2, min_samples_leaf=1, min_samples_split=2, min_purity_increase=0.0, rng=SEED)
    DecisionTree.fit!(model, X, y)
    # pretty print of the tree, to a depth of 5 nodes (optional)
    #print_tree(model, 5)
    # apply learned model
    # get the probability of each label
    current_sample = reshape(X[outlier_index, :] , (size(X)[2],1))
    p_index = findfirst(isequal(OUTLIER_CLASS), model.classes)
    predict_proba(model, current_sample')[p_index]
end

function evaluate_solutions(ireos, solutions, gamma_min, gamma_max)
    @info "Started IREOS Evaluation:"
    results = Dict{Int, Float64}()
    for i in 1:size(solutions)[1]
        results[i] = evaluate_solution(ireos, solutions[i,:], gamma_min, gamma_max)
    end
    return results
end

function normalize_solutions!(solutions, norm_method)
    algorithms = solutions[2]
    if norm_method == "normalization"
        func = normalize
    elseif norm_method == "standardization"
        func = standardize
    end
    for i in 1:length(algorithms)
        solutions[1]'[i,:] = func(regularize_scores(algorithms[i], solutions[1]'[i,:]))
    end
end

function regularize_scores(alg, scores)
    if alg == "lof"
        return reg_base(1.0, scores)
    elseif alg == "ldof"
        return reg_base(0.5, scores)
    elseif alg == "abod"
        return reg_log_inverse(scores)
    # The higher, the more abnormal. Outliers tend to have higher scores. This value is available once the detector is fitted.
    elseif alg == "iforest" || alg == "ocsvm" || alg == "sdo" || alg == "hbos" || alg == "knn"
        return reg_lin(scores)
    end
end

reg_base(base, data) = broadcast(max, broadcast(-,data, base), 0)

reg_lin(data) = begin _min = minimum(data); broadcast(-, data, _min) end

reg_lin_inverse(data) = begin _max = maximum(data); broadcast(-, _max, data) end

reg_log_inverse(data, log_base = MathConstants.e) = begin _max = maximum(data); - broadcast(log, log_base, broadcast(/, data, _max)) end

normalize(data) = StatsBase.transform!(fit(UnitRangeTransform, data), data)

standardize(data) = StatsBase.transform!(fit(ZScoreTransform, data), data)

evaluate_solution(ireos, solution, gamma_min, gamma_max) = sum(ireos .* solution) / sum(solution) / (gamma_max - gamma_min)

# radial basis function: K(x, y) = exp(-gamma ||x-y||^2)
rbf_kernel(X, gamma) = exp.(-gamma * pairwise(SqEuclidean(), X, dims=1))

println("Package Ireos loaded")

end
