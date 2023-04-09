"""
Main implementation of parallel IREOS
"""
const l = ReentrantLock()

function fireos_par(X, clf; kwargs...)
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
    _fireos_par(X, clf, gamma_min, gamma_max, tol, window_size, adaptive_quads_enabled)
end

"""
main multithreaded fireos function
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
function _fireos_par(X::Matrix{Float64}, clf::String, gamma_min::Float64, gamma_max::Float64, tol::Float64, window_size::Integer, adaptive_quads_enabled::Bool)
    number_of_threads = Threads.nthreads()
    num_samples = size(X)[1]
    @assert num_samples == size(X)[1]
    window_mode = use_window_mode(window_size, num_samples)

    aucs = Vector{Float64}(undef, num_samples)
    if window_mode
        y = zeros(Int8, number_of_threads, window_size)
    else
        y = zeros(Int8, number_of_threads, num_samples)
    end
    fill!(y, INLIER_CLASS)
    @info "Started Parallel IREOS with dataset of size:", size(X), "window size: $window_size, gamma_min: $gamma_min, gamma_max: $gamma_max, tol: $tol, classifier: $clf, max_recursion_depth: $MAX_RECURSION_DEPTH, adaptive_quads_enabled: $adaptive_quads_enabled, number_of_threads: $number_of_threads"
    clf_func = get_classifier_function_par(clf)
    Threads.@threads for i in 1:num_samples
        tid = Threads.threadid()
        @debug "Started IREOS calculation of sample number: $i"
        if window_mode
            outlier_index = i <= window_size ? i : window_size
            idx_start = get_start_idx(i, window_size, num_samples)
            idx_end = idx_start + window_size - 1
            y[tid, outlier_index] = OUTLIER_CLASS
            @debug "windowed mode: index_start: $idx_start, index_end: $idx_end"
            if adaptive_quads_enabled
                seperabilities = Dict{Float64, Float64}()
                T = Dict{Float64, Matrix{Float64}}()
                aucs[i] = adaptive_quads_par(X[idx_start:idx_end,:], y[tid,:], outlier_index, gamma_min, gamma_max, tol, clf_func, seperabilities, T)
            else
                aucs[i] = clf_func(X[idx_start:idx_end,:], y[tid,:], i, outlier_index, gamma_max, nothing)
            end
            y[tid, outlier_index] = INLIER_CLASS
        else
            y[tid, i] = OUTLIER_CLASS
            if adaptive_quads_enabled
                seperabilities = Dict{Float64, Float64}()
                T = Dict{Float64, Matrix{Float64}}()
                aucs[i] = adaptive_quads_par(X, y[tid,:], i, gamma_min, gamma_max, tol, clf_func, seperabilities, T)
            else
                aucs[i] = clf_func(X, y[tid,:], i, gamma_max, nothing)
            end
            y[tid, i] = INLIER_CLASS
        end
        @debug "IREOS calculation of sample number: $i successful"
    end
    return aucs
end

"""
function for getting the correnponsing classifier function by given string
...
# Arguments
- `clf`: predictor string
......` 
returns classifier function
"""
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
    outlier_prob = sk_logreg_par(T, y, outlier_index, gamma)
    unlock(l)
    return outlier_prob
end

function sk_logreg_par(T, y, outlier_index, gamma)
    clf = LogisticRegression(random_state=SEED, tol=0.0095, max_iter=1000000)
    ScikitLearn.fit!(clf, T[gamma], y)
    return get_outlier_prob(clf, reshape(T[gamma][outlier_index, :] , (size(T[gamma])[2],1)))
end

function get_svm_clf_par(X, y, outlier_index, gamma, T)
    #SVC cannot deal with gamma == 0
    if gamma == 0.0
        gamma = 0.0001
    end
    @debug "Thread", threadid(), "Gamma: $gamma, X: ", size(X), "y: ", size(y), "Outlier-Index:", findfirst(isequal(OUTLIER_CLASS), y)
    # lock(func, lock) did not work
    lock(l)
    outlier_prob = sk_svm_par(X, y, reshape(X[outlier_index, :] , (size(X)[2],1)), gamma)
    unlock(l)
    return outlier_prob
end

function sk_svm_par(X, y, current_sample, gamma)
    clf = SVC(gamma=gamma, class_weight="balanced", probability=true, C=100, random_state=SEED, tol=0.0095, max_iter=-1)
    ScikitLearn.fit!(clf, X, y)
    return get_outlier_prob(clf, current_sample)
end

function get_klr_clf_par(X, y, outlier_index, gamma, T)
    if !haskey(T, gamma)
        @debug "Gamma: $gamma missing.. Calculating R-Matrix"
        T[gamma] = rbf_kernel(X, gamma)
    end
    lock(l)
    outlier_prob = sk_klr_par(T, y, outlier_index, gamma)
    unlock(l)
    return outlier_prob
end

function sk_klr_par(T, y, outlier_index, gamma)
    clf = LogisticRegression(class_weight="balanced", tol=0.0095, solver = "saga", C=100, max_iter=1000000, random_state=SEED)
    ScikitLearn.fit!(clf, T[gamma], y)
    return get_outlier_prob(clf, reshape(T[gamma][outlier_index, :] , (size(T[gamma])[2],1)))
end

function get_libsvm_par(X, y, outlier_index, gamma, T)
    clf = svmtrain(X', y, gamma=gamma, probability=true, tolerance=tol=0.0095, cost=100.0)
    # current_sample = reshape(X[outlier_index, :] , (size(X)[2],1))
    p_index = findfirst(isequal(OUTLIER_CLASS), clf.labels)
    return svmpredict(clf, reshape(X[outlier_index, :] , (size(X)[2],1)))[2][p_index]
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
    model = linear_train(y, X', solver_type=Cint(7), verbose=false)
    (predicted_labels, decision_values) = linear_predict(model, reshape(X[outlier_index, :] , (size(X)[2],1)), probability_estimates=true, verbose=false);
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
        p_outlier = apply_tree_proba(model, reshape(X[outlier_index, :] , (1,size(X)[2])), [OUTLIER_CLASS, INLIER_CLASS])[1]
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
    p_index = findfirst(isequal(OUTLIER_CLASS), model.classes)
    predict_proba(model, reshape(X[outlier_index, :] , (1, size(X)[2])))[p_index]
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
    model = build_forest(y, X, rng=SEED)
    p_outlier = apply_forest_proba(model, reshape(X[outlier_index, :] , (1,size(X)[2])), [OUTLIER_CLASS, INLIER_CLASS])[1]
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
    #p_index = findfirst(isequal(OUTLIER_CLASS), model.classes)
    predict_proba(model, reshape(X[outlier_index, :] , (1,size(X)[2])))[findfirst(isequal(OUTLIER_CLASS), model.classes)]
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
    # current_sample = reshape(X[outlier_index, :] , (1,size(X)[2]))
    # softprob transforms result into probability for multiclass classification
    # based on same principle as logistic regression -> is a generalization
    # https://stackoverflow.com/questions/36051506/difference-between-logistic-regression-and-softmax-regression
    # model = xgboost(X, num_rounds, label=y, max_depth=2, seed=SEED, objective="multi:softprob", num_class=2, silent=1, validate_parameters=true)
    # print(XGBoost.predict(model, current_sample))
    # equals reg:logistic
    # random forest parameter setting https://xgboost.readthedocs.io/en/stable/tutorials/rf.html
    model = xgboost(X, num_rounds, label=y, max_depth=2, seed=SEED, objective="binary:logistic", silent=true, eta = 1, subsample=0.8, num_parallel_tree=10, num_boost_round=3, nthreads=Threads.nthreads())
    # For binary classification, the output predictions are probability confidence scores in [0,1], corresponds to the probability of the label to be positive.
    # Outlier prediction acts as success event a binomial trial scenario
    XGBoost.predict(model, reshape(X[outlier_index, :] , (1,size(X)[2])))[1]
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
    model = xgboost(X, num_rounds, label=y, booster="dart", max_depth=2, seed=SEED, objective="binary:logistic", silent=1)
    XGBoost.predict(model, reshape(X[outlier_index, :] , (1,size(X)[2])))[1]
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
    # nthread must be 1 here in order to keep it reproducable -> test will fail otherwise
    model = xgboost(X, num_rounds, label=y, booster="gblinear", max_depth=2, seed=SEED, objective="binary:logistic", silent=1, nthread=1)
    XGBoost.predict(model, reshape(X[outlier_index, :] , (1,size(X)[2])))[1]
end

"""
multithreaded function for evaluating vector of solutions. One solution mvector must consist of size n
...
# Arguments
- `ireos`: vectur of probabilities
- `solutions`: solution vector or matrix
- `gamma_min`: used minimum gamma for ireos
- `gamma_max`: used maximum gamma for ireos
......` 
returns probability that outlier sample is classified as outlier
"""
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

"""
multithreaded function for regularizing and normalizing/standardizing solutions
...
# Arguments
- `solutions`: dataframe of solutions, solutions[1] represents data, solutions[2] different algorithms
- `norm_method`: string for normalization of standardization
......` 
returns regularized and normalized/standardized data
"""
function normalize_solutions_par!(solutions, norm_method)
    algorithms = solutions[2]
    if norm_method == "normalization"
        @debug "parallel normalization of dataset started"
        func = normalize
    elseif norm_method == "standardization"
        @debug "parallel standardization of dataset started"
        func = standardize
    end
    Threads.@threads for i in eachindex(algorithms)
        solutions[1]'[i,:] = func(regularize_scores(algorithms[i], solutions[1]'[i,:]))
    end
end

println("Package IreosPar loaded")