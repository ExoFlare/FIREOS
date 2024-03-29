"""
Main implementation of sequential IREOS
"""

function fireos(X, clf; kwargs...)
    if haskey(kwargs, :gamma_min)
        gamma_min = kwargs[:gamma_min]
    else
        gamma_min = 0.0
    end
    if haskey(kwargs, :gamma_max)
        gamma_max = kwargs[:gamma_max]
    else
        gamma_max = get_default_gamma_max_for_clf(clf, X)
    end
    if haskey(kwargs, :adaptive_quads_enabled)
        adaptive_quads_enabled = kwargs[:adaptive_quads_enabled]
    else
        adaptive_quads_enabled = get_default_adaptive_quads_enabled_for_clf(clf)
    end
    if haskey(kwargs, :tol)
        tol = kwargs[:tol]
    else
        tol = 0.005
    end
    if haskey(kwargs, :window_size)
        window_size = kwargs[:window_size]
    else
        window_size = size(X)[1]
    end
    _fireos(X, clf, gamma_min, gamma_max, tol, window_size, adaptive_quads_enabled)
end
"""
main ireos function
...
# Arguments
- `X::AbstractMatrix{<:Number}`: numerical input matrix of data having size (n x m)
- `clf::String`: internal predictor string
- `gamma_min::Float64`: minimum gamma for models using gamma hyperparameter
- `gamma_max::Float64`: maximum gamma for models using gamma hyperparameter
- `tol::Float64`: epsilon: termination condition for adaptive quadrature
- `window_size::Integer`: size of the sliding window -> n if no window
- `adaptive_quads_enabled::Bool`: true if adaptive quadrature calculation enabled else false
......` 
returns aucs::Vector{Float64}: numerical vector of separabilities having size n
"""
function _fireos(X::AbstractMatrix{<:Number}, clf::String, gamma_min::Float64, gamma_max::Float64, tol::Float64, window_size::Integer, adaptive_quads_enabled::Bool)
    num_samples = size(X)[1]
    @assert num_samples == size(X)[1]
    window_mode = use_window_mode(window_size, num_samples)

    aucs = Vector{Float64}(undef,num_samples)
    @info "Started IREOS with dataset of size:", size(X), "window size: $window_size, gamma_min: $gamma_min, gamma_max: $gamma_max, tol: $tol, classifier: $clf, max_recursion_depth: $MAX_RECURSION_DEPTH, adaptive_quads_enabled: $adaptive_quads_enabled"
    clf_func = get_classifier_function(clf)
    if window_mode
        y = fill(INLIER_CLASS, window_size)
    else
        y = fill(INLIER_CLASS, num_samples)
    end
    T = Dict{Float64, AbstractMatrix{<:Number}}()
    for i in 1:num_samples
        @debug "Started IREOS calculation of sample number: $i"
        seperabilities = Dict{Float64, Float64}()
        if window_mode
            idx_start = get_start_idx(i, window_size, num_samples)
            outlier_index = get_outlier_idx(i, window_size, num_samples)
            idx_end = idx_start + window_size - 1
            data = X[idx_start:idx_end,:]
            y[outlier_index] = OUTLIER_CLASS
        else
            outlier_index = i
            data = X
            y[i] = OUTLIER_CLASS
        end
        if(adaptive_quads_enabled)
            aucs[i] = adaptive_quads(data, y, outlier_index, gamma_min, gamma_max, tol, clf_func, seperabilities, T)
        else
            aucs[i] = clf_func(data, y, outlier_index, gamma_max, T)
        end
        @info "IREOS calculation of sample number: $i successful"
        if window_mode
            y[outlier_index] = INLIER_CLASS
        else
            y[i] = INLIER_CLASS
        end
    end
    return aucs
end

"""
function for applying adaptive quadrature and simpsons rule for calculating the area und the curve
...
# Arguments
- `X`: feature matrix of input data
- `y`: target vector of input matrix
- `outlier_index`: current index of outlier
- `a`: lower gamma value of adaptive quadrature
- `b`: higher gamma value of adaptive quadrature
- `tol`: error rate epsilon, maximum error
- `clf_func`: function of predictor to be used
- `seperabilities`: map of already calculated separabilities for given gammas
- `T`: precomputed transformed matrix or nothing
- `current_recursion_depth`: current depth of recursion, must be smaller than $MAX_RECURSION_DEPTH
......` 
returns area under the curve of separabilities of given outlier and predictor
"""
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

"""
function for calculation of Simpson's rule
...
# Arguments
- `X`: feature matrix of input data
- `y`: target vector of input matrix
- `outlier_index`: current index of outlier
- `a`: lower gamma value of adaptive quadrature
- `b`: higher gamma value of adaptive quadrature
- `clf_func`: function of predictor to be used
- `seperabilities`: map of already calculated separabilities for given gammas
- `T`: precomputed transformed matrix or nothing
......` 
returns result of simpson's rule of given interval a and b
"""
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

"""
function for getting the correnponsing classifier function by given string
...
# Arguments
- `clf`: predictor string
......` 
returns classifier function
"""
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
    elseif clf == "xgboost_tree"
        return get_xgboost_tree
    elseif clf == "xgboost_dart"
        return get_xgboost_dart
    elseif clf == "xgboost_linear"
        return get_xgboost_linear
    else
        @error "Unknown classifier $clf"
    end
end

"""
function for predicting probabilities of sklearn predictors
...
# Arguments
- `clf`: classifier function
- `X`: feature matrix of input data
- `y`: target vector of input matrix
- `gamma`: not used in sklearn predictors
- `outlier_index`: index of current outlier
......` 
returns probability that outlier sample is classified as outlier
"""
function predict_sk_clf_proba(clf, X, y, gamma, outlier_index)
    ScikitLearn.fit!(clf, X, y)
    current_sample = reshape(X[outlier_index, :] , (size(X)[2],1))
    p_index = findfirst(isequal(OUTLIER_CLASS), clf.classes_)
    return predict_proba(clf, current_sample')[p_index]
end

"""
function for predicting probabilities using logistic regression
...
# Arguments
- `X`: feature matrix of input data
- `y`: target vector of input matrix
- `outlier_index`: index of current outlier
- `gamma`: current gamma
- `T`: map precomputed rbf kernel matrix of given gammas
......` 
returns probability that outlier sample is classified as outlier
"""
function get_logreg_clf(X, y, outlier_index, gamma, T)
    clf = LogisticRegression(random_state=SEED, tol=0.0095, max_iter=1000000)

    if !haskey(T, gamma)
        @debug "Gamma: $gamma missing.. Calculating R-Matrix"
        T[gamma] = rbf_kernel(X, gamma)
    end
    return predict_sk_clf_proba(clf, T[gamma], y, gamma, outlier_index)
end

"""
function for predicting probabilities using support vector machines
...
# Arguments
- `X`: feature matrix of input data
- `y`: target vector of input matrix
- `outlier_index`: index of current outlier
- `gamma`: current gamma
- `T`: not used in SVMs
......` 
returns probability that outlier sample is classified as outlier
"""
function get_svm_clf(X, y, outlier_index, gamma, T)
    #SVC cannot deal with gamma == 0
    if gamma == 0.0
        gamma = 0.0001
    end
    @debug "Thread", threadid(), "Gamma: $gamma, X: ", size(X), "y: ", size(y), "Outlier-Index:", findfirst(isequal(OUTLIER_CLASS), y)
    clf = SVC(gamma=gamma, class_weight="balanced", probability=true, C=100, random_state=SEED, tol=0.0095, max_iter=-1)
    return predict_sk_clf_proba(clf, X, y, gamma, outlier_index)
end

"""
function for predicting probabilities using kernel logistic regression
...
# Arguments
- `X`: feature matrix of input data
- `y`: target vector of input matrix
- `outlier_index`: index of current outlier
- `gamma`: current gamma
- `T`: map precomputed rbf kernel matrix of given gammas
......` 
returns probability that outlier sample is classified as outlier
"""
function get_klr_clf(X, y, outlier_index, gamma, T)
    # param set closest to the paper (liblinear returns non-zero values for gamma = 0)
    clf = LogisticRegression(class_weight="balanced", tol=0.0095, solver = "saga", C=100, max_iter=1000000, random_state=SEED)
    if !haskey(T, gamma)
        @debug "Gamma: $gamma missing.. Calculating R-Matrix"
        T[gamma] = rbf_kernel(X, gamma)
    end
    #self.ireos_data.set_current_sample(R[self.ireos_data.get_index()].reshape(1, -1))
    return predict_sk_clf_proba(clf, T[gamma], y, gamma, outlier_index)
end

"""
function for predicting probabilities using support vector machines of library libsvm
...
# Arguments
- `X`: feature matrix of input data
- `y`: target vector of input matrix
- `outlier_index`: index of current outlier
- `gamma`: current gamma
- `T`: not used in SVMs
......` 
returns probability that outlier sample is classified as outlier
"""
function get_libsvm(X, y, outlier_index, gamma, T)
    clf = svmtrain(X', y, gamma=gamma, probability=true)
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
function get_decision_tree_native(X, y, outlier_index, gamma, T)
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
function get_decision_tree_sklearn(X, y, outlier_index, gamma, T)
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
function get_random_forest_native(X, y, outlier_index, gamma, T)
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
function get_random_forest_sklearn(X, y, outlier_index, gamma, T)
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
function get_xgboost_tree(X, y, outlier_index, gamma, T)
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
function get_xgboost_dart(X, y, outlier_index, gamma, T)
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
function get_xgboost_linear(X, y, outlier_index, gamma, T)
    # xgboost cannot deal with values outsode [0,1) -> tempoarily transform negative labels
    replace!(y, INLIER_CLASS => 0)
    # num rounds act as number of estimators in rf
    num_rounds = 10
    current_sample = reshape(X[outlier_index, :] , (1,size(X)[2]))
    #nthreads = 1 because otherwise seed is not considered
    model = xgboost(X, num_rounds, label=y, booster="gblinear", max_depth=2, seed=SEED, objective="binary:logistic", silent=1, nthread=1)
    return XGBoost.predict(model, current_sample)[1]
end

"""
function for evaluating vector of solutions. One solution mvector must consist of size n
...
# Arguments
- `ireos`: vectur of probabilities
- `solutions`: solution vector or matrix
- `gamma_min`: used minimum gamma for ireos
- `gamma_max`: used maximum gamma for ireos
......` 
returns probability that outlier sample is classified as outlier
"""
function evaluate_solutions(fireos, solutions, gamma_min, gamma_max)
    if isnothing(solutions)
        return nothing
    end
    @info "Started FIREOS Evaluation:"
    results = Dict{Int, Float64}()
    for i in 1:size(solutions)[1]
        results[i] = evaluate_solution(fireos, solutions[i,:], gamma_min, gamma_max)
    end
    return results
end

"""
function for regularizing and normalizing/standardizing solutions
...
# Arguments
- `solutions`: dataframe of solutions, solutions[1] represents data, solutions[2] different algorithms
- `norm_method`: string for normalization of standardization
......` 
returns regularized and normalized/standardized data
"""
function normalize_solutions!(solutions, norm_method)
    algorithms = solutions[2]
    if norm_method == "normalization"
        @debug "sequential normalization of dataset started"
        func = normalize
    elseif norm_method == "standardization"
        @debug "sequential standardization of dataset started"
        func = standardize
    end
    for i in eachindex(algorithms)
        solutions[1]'[i,:] = func(regularize_scores(algorithms[i], solutions[1]'[i,:]))
    end
end

println("Package Ireos loaded")
