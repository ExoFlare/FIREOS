module Ireos

using Base.Threads
using Distributed
using ScikitLearn
using Distances

@sk_import linear_model: LogisticRegression
@sk_import svm: SVC

const VERSION = "0.0.2"
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

function ireos(X::AbstractMatrix{<:Number}, clf::String, gamma_min::Float64, gamma_max::Float64, tol::Float64)
    aucs = Vector{Float64}()
    num_samples = size(X)[1]
    @assert num_samples == size(X)[1]
    @info "Started IREOS with dataset of size:", size(X), " gamma_min: $gamma_min, gamma_max: $gamma_max, tol: $tol, classifier: $clf, max_recursion_depth: $MAX_RECURSION_DEPTH"
    clf_func = get_classifier_function(clf)
    y = fill(INLIER_CLASS, num_samples)
    T::AbstractDict{Float64, AbstractMatrix{<:Number}} = Dict{Float64, AbstractMatrix{<:Number}}()
    for i in 1:num_samples
        y[i] = OUTLIER_CLASS
        seperabilities = Dict{Float64, Float64}()
        outlier_index = findfirst(isequal(OUTLIER_CLASS), y)
        @debug "Started IREOS calculation of sample number: $outlier_index"
        push!(aucs, adaptive_quads(X, y, outlier_index, gamma_min, gamma_max, tol, clf_func, seperabilities, T))
        @debug "IREOS calculation of sample number: $outlier_index successful"
        y[i] = INLIER_CLASS
    end
    return aucs
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
        clf, current_sample = clf_func(X, y, outlier_index, i, T)
        p_index = findfirst(isequal(OUTLIER_CLASS), clf.classes_)
        p_outlier = predict_proba(clf, current_sample')[p_index]
        
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
    else
        @error "Unknown classifier $clf"
    end
end

function get_logreg_clf(X, y, outlier_index, gamma, T)
    clf = LogisticRegression(random_state=123, tol=0.0095, max_iter=1000000)

    if !haskey(T, gamma)
        @debug "Gamma: $gamma missing.. Calculating R-Matrix"
        T[gamma] = rbf_kernel(X, gamma)
    end
    #self.ireos_data.set_current_sample(R[self.ireos_data.get_index()].reshape(1, -1))
    fit!(clf, T[gamma], y)
    current_sample = reshape(T[gamma][outlier_index, :] , (size(T[gamma])[2],1))
    return clf, current_sample
end

function get_svm_clf(X, y, outlier_index, gamma, T)
    #SVC cannot deal with gamma == 0
    if gamma == 0.0
        gamma = 0.0001
    end

    clf = SVC(gamma=gamma, probability=true, C=100, random_state=123, tol=0.0095, max_iter=1000000)
    fit!(clf, X, y)
    current_sample = reshape(X[outlier_index, :] , (size(X)[2],1))
    return clf, current_sample
end

function get_klr_clf(X, y, outlier_index, gamma, T)
    # param set closest to the paper (liblinear returns non-zero values for gamma = 0)
    clf = LogisticRegression(class_weight="balanced", tol=0.0095, solver = "saga", C=100, max_iter=1000000, random_state=123)
    if !haskey(T, gamma)
        @debug "Gamma: $gamma missing.. Calculating R-Matrix"
        T[gamma] = rbf_kernel(X, gamma)
    end
    #self.ireos_data.set_current_sample(R[self.ireos_data.get_index()].reshape(1, -1))
    fit!(clf, T[gamma], y)
    current_sample = reshape(T[gamma][outlier_index, :] , (size(T[gamma])[2],1))
    return clf, current_sample
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
