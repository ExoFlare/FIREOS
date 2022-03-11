using ScikitLearn

@sk_import linear_model: LogisticRegression
@sk_import svm: SVC


const INLIER_CLASS = -1
const OUTLIER_CLASS = 1
const MAX_RECURSION_DEPTH = 10

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
    @debug("Started IREOS with dataset of size:", size(X))
    clf_func = get_classifier_function(clf)
    y = fill(INLIER_CLASS, num_samples)
    for i in 1:num_samples
        y[i] = OUTLIER_CLASS
        seperabilities = Dict{Float64, Float64}()
        push!(aucs, adaptive_quads(X, y, gamma_min, gamma_max, tol, clf_func, seperabilities))
        y[i] = INLIER_CLASS
    end
    return aucs
end

function adaptive_quads(X, y, a::Float64, b::Float64, tol::Float64, clf_func, seperabilities)
    m = (a + b) / 2
    err_all = simpson_rule(X, y, a, b, clf_func, seperabilities)
    err_new = simpson_rule(X, y, a, m, clf_func, seperabilities) + simpson_rule(X, y, m, b, clf_func, seperabilities)
    calculated_error = abs(err_all - err_new) / 15
    if tol < calculated_error && 0 < MAX_RECURSION_DEPTH
        @debug "Iteration depth: {}. Criterion not reached: $calculated_error > $tol"
        #self.current_recursion_depth += 1
        return adaptive_quads(X, y, a, m, tol / 2, clf_func, seperabilities) + adaptive_quads(X, y, m, b, tol/2, clf_func, seperabilities)
    else
        @debug "Termination criterion of $calculated_error < $tol reached."
        return err_new
    end
end

function simpson_rule(X, y, a, b, clf_func, seperabilities)
    h = (b - a) / 2
    for i in [a, a+h, b]
        if i in keys(seperabilities)
            continue
        end
        clf = clf_func(X, y, i)
        p_index = findfirst(isequal(OUTLIER_CLASS), clf.classes_)
        current_index = findfirst(isequal(OUTLIER_CLASS), y)
        current_sample = reshape(X[current_index, :] , (size(X)[2],1))
        p_outlier = predict_proba(clf, current_sample')[p_index]
          
        @debug "$current_index: gamma $i : p-value: $p_outlier"
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

function get_logreg_clf(X, y, gamma)
    clf = LogisticRegression(random_state=123, tol=0.0095, max_iter=1000000)
    R = rbf_kernel(X, gamma=gamma)
     
    #self.ireos_data.set_current_sample(R[self.ireos_data.get_index()].reshape(1, -1))
    clf.fit(R, y)
    return clf
end

function get_svm_clf(X, y, gamma)
    clf = SVC(gamma=gamma, probability=true, C=100, random_state=123, tol=0.0095, max_iter=1000000)
    fit!(clf, X, y)
    return clf
end

function get_klr_clf(X, y, gamma)
    # param set closest to the paper (liblinear returns non-zero values for gamma = 0)
    clf = LogisticRegression(class_weight="balanced", tol=0.0095, solver = "saga", C=100, max_iter=1000000, random_state=123)
    R = rbf_kernel(X, gamma=gamma)
    #self.ireos_data.set_current_sample(R[self.ireos_data.get_index()].reshape(1, -1))
    clf.fit(R, y)
    return clf
end





##########################
using DelimitedFiles
using Logging
ENV["JULIA_DEBUG"] = Main
data = readdlm("WBC_withoutdupl_norm")

const gamma_min = 0.0001
const gamma_max = 2.560381915956203
const tol = 0.005
res = @time ireos(data, "svc", gamma_min, gamma_max, tol)