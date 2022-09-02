# create empty results dataframe with predefined column names for aggregated scores
create_empty_result_df() = DataFrame(dataset = [], clf = String[], adaptive_quads_enabled = Bool[], window_ratio = Float64[], is_parallel = Bool[], time = Float64[]
, ireos_sdo = Float64[], ireos_abod = Float64[], ireos_hbos = Float64[], ireos_iforest = Float64[], ireos_knn = Float64[], ireos_lof = Float64[], ireos_ocsvm = Float64[])

# push row with values with fallback
push_row!(df, current_dataset_name, clf, adaptive_quads_enabled_mode, window_ratio, is_parallel, time, results) = push!(df, (current_dataset_name, clf, adaptive_quads_enabled_mode, isnothing(window_ratio) ? 1.0 : window_ratio, is_parallel, time
, !isnothing(results) && !isnothing(results[1]) ? results[1] : -1.0, !isnothing(results) && length(results) > 1 && !isnothing(results[2]) ? results[2] : -1.0
, !isnothing(results) && length(results) > 2 && !isnothing(results[3]) ? results[3] : -1.0, !isnothing(results) && length(results) > 3 && !isnothing(results[4]) ? results[4] : -1.0
, !isnothing(results) && length(results) > 4 && !isnothing(results[5]) ? results[5] : -1.0, !isnothing(results) && length(results) > 5 && !isnothing(results[6]) ? results[6] : -1.0
, !isnothing(results) && length(results) > 6 && !isnothing(results[7]) ? results[7] : -1.0))

# function for executing one calculation of sequential ireos ( calculating parameters -> training ireos -> evaluating solutions)
function execute_sequential_experiment(data, solutions, clf, gamma_min, gamma_max, tol, window_ratio, adaptive_quads_enabled)
    if isnothing(window_ratio)
        window_size = nothing
    else
        window_size = round(Int32, window_ratio * size(data)[1])
    end
    params = create_params(gamma_min, gamma_max, tol, window_size, adaptive_quads_enabled)
    trained = FIREOS.fireos(data, clf; params...)
    eval_gamma_min = !isnothing(gamma_min) ? gamma_min : FIREOS.get_default_gamma_min()
    eval_gamma_max = !isnothing(gamma_max) ? gamma_max : FIREOS.get_default_gamma_max_for_clf(clf, data)
    return create_result(FIREOS.evaluate_solutions, trained, solutions, eval_gamma_min, eval_gamma_max)
end

# function for executing one calculation of parallel ireos ( calculating parameters -> training ireos -> evaluating solutions)
function execute_parallel_experiment(data, solutions, clf, gamma_min, gamma_max, tol, window_ratio, adaptive_quads_enabled)
    if isnothing(window_ratio)
        window_size = nothing
    else
        window_size = round(Int32, window_ratio * size(data)[1])
    end
    params = create_params(gamma_min, gamma_max, tol, window_size, adaptive_quads_enabled)
    trained = FIREOS.fireos_par(data, clf; params...)
    eval_gamma_min = !isnothing(gamma_min) ? gamma_min : FIREOS.get_default_gamma_min()
    eval_gamma_max = !isnothing(gamma_max) ? gamma_max : FIREOS.get_default_gamma_max_for_clf(clf, data)
    return create_result(FIREOS.evaluate_solutions_par, trained, solutions, eval_gamma_min, eval_gamma_max)
end

function create_params(gamma_min, gamma_max, tol, window_ratio, adaptive_quads_enabled)
    params = Dict()
    if !isnothing(gamma_min)
        params[:gamma_min] = gamma_min
    end
    if !isnothing(gamma_max)
        params[:gamma_max] = gamma_max
    end
    if !isnothing(tol)
        params[:tol] = tol
    end
    if !isnothing(window_ratio)
        params[:window_ratio] = window_ratio
    end
    if !isnothing(adaptive_quads_enabled)
        params[:adaptive_quads_enabled] = adaptive_quads_enabled
    end
    return params
end

# helper function for nullsave evaluation of solutions 
create_result(ireos_evaluation_func, trained, solutions, gamma_min, gamma_max) = isnothing(solutions) ? nothing : 
ireos_evaluation_func(trained, solutions[1]', gamma_min, gamma_max) , trained / (gamma_max - gamma_min)