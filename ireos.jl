module Ireos

using Base.Threads
using Distributed
using ScikitLearn
using Distances
using LIBSVM
using LIBLINEAR

@sk_import linear_model: LogisticRegression
@sk_import svm: SVC

const VERSION = "0.0.4"
const INLIER_CLASS = -1
const OUTLIER_CLASS = 1
const MAX_RECURSION_DEPTH = 3



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

function ireos(X::AbstractMatrix{<:Number}, clf::String, gamma_min::Float64, gamma_max::Float64, tol::Float64, window_size::Union{Int32, Nothing}=nothing)
    aucs = Vector{Float64}()
    window_mode = false
    num_samples = size(X)[1]
    if !isnothing(window_size)
        @assert window_size <= num_samples
        window_mode = true
    end
    @info "Started IREOS with dataset of size:", size(X), "window size: $window_size, gamma_min: $gamma_min, gamma_max: $gamma_max, tol: $tol, classifier: $clf, max_recursion_depth: $MAX_RECURSION_DEPTH"
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
        push!(aucs, adaptive_quads(data, y, outlier_index, gamma_min, gamma_max, tol, clf_func, seperabilities, T))
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

function evaluate_solutions(ireos, solutions, gamma_min, gamma_max)
    @info "Started IREOS Evaluation:"
    results = Dict{Int, Float64}()
    for i in 1:size(solutions)[1]
        results[i] = evaluate_solution(ireos, solutions[i,:], gamma_min, gamma_max)
    end
    return results
end

evaluate_solution(ireos, solution, gamma_min, gamma_max) = sum(ireos .* solution) / sum(solution) / (gamma_max - gamma_min)

# radial basis function: K(x, y) = exp(-gamma ||x-y||^2)
rbf_kernel(X, gamma) = exp.(-gamma * pairwise(SqEuclidean(), X, dims=1))

println("Package Ireos loaded")

end