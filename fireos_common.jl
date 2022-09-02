"""
File that contains common function used by both fireos and fireos_par
"""

const CLFS_WITH_GAMMA_PARAM = ["svc", "logreg", "klr", "libsvm"]
const CLFS_WITHOUT_GAMMA_PARAM = ["liblinear", "decision_tree_native", "decision_tree_sklearn", "random_forest_native", "random_forest_sklearn", "xgboost_tree", "xgboost_dart", "xgboost_linear"]

function use_window_mode(window_size::Integer, num_samples::Integer)
    if !isnothing(window_size)
        @assert window_size <= num_samples
        # apply sliding window only when !=
        if num_samples < window_size
            return true
        end
        return false
    end
    throw(ArgumentError("window size must not be null"))
end

"""
helper function for calculating the start index in a sliding window
...
# Arguments
- `current_index`: current sample
- `window_size`: window ratio
- `size_all`: overall size of the dataset
......` 
returns starting index of sliding window
"""
function get_start_idx(current_index, window_size, size_all)
    if current_index <= window_size
        return 1
    elseif current_index > size_all - window_size
        return size_all - (window_size - 1)
    else
        return current_index - (window_size - 1)
    end
end

"""
function for applying regularization described in 'Interpreting and Unifying Outlier Scores' by Kriegel et. al.
...
# Arguments
- `alg`: algorithm
- `scores`: data vector
......` 
returns regularized and normalized/standardized data
"""
function regularize_scores(alg::Union{String,SubString}, scores::Vector{<:Number})
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

function get_default_adaptive_quads_enabled_for_clf(clf::String)
    return clf in CLFS_WITH_GAMMA_PARAM
end

function get_default_gamma_max_for_clf(clf::String, data::Matrix{<:Number})
    return clf in CLFS_WITH_GAMMA_PARAM ? maximum(pairwise(Euclidean(), data, dims=1)) : 1.0
end

function get_default_gamma_min()
    return 0.0
end

calculate_class_weights(array) = begin class_map = sort(countmap(array)); length(array) ./ (length(class_map) .* values(class_map)) end

# baseline regularization
reg_base(base, data) = broadcast(max, broadcast(-,data, base), 0)

# regularization with minimum
reg_lin(data) = begin _min = minimum(data); broadcast(-, data, _min) end

# linear inversion
reg_lin_inverse(data) = begin _max = maximum(data); broadcast(-, _max, data) end

# logarithmic inversion
reg_log_inverse(data, log_base = MathConstants.e) = begin _max = maximum(data); - broadcast(log, log_base, broadcast(/, data, _max)) end

# normalization
normalize(data) = StatsBase.transform!(fit(UnitRangeTransform, data), data)

# standardization
standardize(data) = StatsBase.transform!(fit(ZScoreTransform, data), data)

# radial basis function: K(x, y) = exp(-gamma ||x-y||^2)
rbf_kernel(X, gamma) = exp.(-gamma * pairwise(SqEuclidean(), X, dims=1))

# evaluation of single solution by given probability vector
evaluate_solution(ireos, solution, gamma_min, gamma_max) = sum(ireos .* solution) / sum(solution) / (gamma_max - gamma_min)
