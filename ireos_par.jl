module IreosPar

using Base.Threads
using Distributed
using ScikitLearn
using Distances
using ThreadSafeDicts
using LIBSVM

@sk_import linear_model: LogisticRegression
@sk_import svm: SVC

const VERSION = "0.0.4"
const INLIER_CLASS = -1
const OUTLIER_CLASS = 1
const MAX_RECURSION_DEPTH = 3

const l = ReentrantLock()

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
    clf = LogisticRegression(random_state=123, tol=0.0095, max_iter=1000000)
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
    clf = SVC(gamma=gamma, class_weight="balanced", probability=true, C=100, random_state=123, tol=0.0095, max_iter=-1)
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
    clf = LogisticRegression(class_weight="balanced", tol=0.0095, solver = "saga", C=100, max_iter=1000000, random_state=123)
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

function evaluate_solutions_par(ireos, solutions, gamma_min, gamma_max)
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
