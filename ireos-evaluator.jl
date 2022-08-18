"""
Main evaluator class. This class tests all implemented predictors and sampling methods and windows using 80 datasets, writing all 
intermediate results into different folders.
"""

include("ireos.jl")
using Main.Ireos

include("ireos_par.jl")
using Main.IreosPar

include("compare_results.jl")

using CSV
using Base.Threads
using DelimitedFiles
using Logging
using BenchmarkTools
using Distances
using DataFrames

const ITERATIONS = 1

function run()
     
    #ENV["JULIA_DEBUG"] = Main
    ENV["JULIA_INFO"] = Main

    global persist_intermediate_result = true

    global data_dir = "data/"
    global scorings_dir = "scores/"
    global scaled_dir = scorings_dir * "scaled/"
    global results_dir = "results/"
    global ireos_dir = results_dir * "ireos/"
    global internal_validation_indices_dir = "internal_validation_indices/"

    global names = ["complex_1", "complex_2", "complex_3", "complex_4", "complex_5", "complex_6", "complex_7", "complex_8", "complex_9", "complex_10",
	"complex_11", "complex_12", "complex_13", "complex_14", "complex_15", "complex_16",
    "complex_17", "complex_18", "complex_19", "complex_20", "high-noise_1", "high-noise_2", "high-noise_3",
	"high-noise_4", "high-noise_5", "high-noise_6", "high-noise_7", "high-noise_8", "high-noise_9", 
	"high-noise_10", "high-noise_11", "high-noise_12", "high-noise_13", 
    "high-noise_14", "high-noise_15", "high-noise_16", "high-noise_17", "high-noise_18", "high-noise_19", "high-noise_20", 
    "dens-diff_1", "dens-diff_2", "dens-diff_3", "dens-diff_4", "dens-diff_5", "dens-diff_6", "dens-diff_7", "dens-diff_8",
	"dens-diff_9", "dens-diff_10", "dens-diff_11", "dens-diff_12", "dens-diff_13", "dens-diff_14", "dens-diff_15", "dens-diff_16", 
    "dens-diff_17", "dens-diff_18", "dens-diff_19", "dens-diff_20",
	"low-noise_1", "low-noise_2", "low-noise_3", "low-noise_4", "low-noise_5", "low-noise_6", "low-noise_7", "low-noise_8", "low-noise_9",
	"low-noise_10", "low-noise_11", "low-noise_12", "low-noise_13", "low-noise_14", "low-noise_15", "low-noise_16", 
    "low-noise_17", "low-noise_18", "low-noise_19", "low-noise_20", "separated_20"]

    #global names = ["my_test"]
    #global names = ["complex_11"]

    clfs = ["decision_tree_native", "decision_tree_sklearn", "random_forest_native", "random_forest_sklearn", "liblinear", "xgboost_tree", "xgboost_dart", "xgboost_linear"]
    clfs_par = ["libsvm"]

    #algs = ["sdo","abod", "hbos", "iforest", "knn", "lof", "ocsvm"]

    window_modes =[0.1, 0.5, nothing]

    #adaptive modes only feature algorithms without alpha -> disable adaptive_quads_modes
    adaptive_quads_modes =[false]

    global gamma_min = 0.0
    global tol = 0.005

    global gamma_max = -1.0

    # Normalization transforms all scores [0,1], is used in the original IREOS paper
    global norm_method = "normalization"

    for d in 1:length(names)
        current_dataset_name = names[d]
        result_df = create_empty_result_df()
        # check if dataset already calculated
        result_file_name = results_dir * internal_validation_indices_dir * current_dataset_name * ".csv"
        if isfile(result_file_name)
            @info "IREOS for Dataset: $current_dataset_name already calculated. Skipping.."
            continue
        end
        # read dataset
        global data = readdlm(data_dir * current_dataset_name,',', Float64, '\n')
        global num_cols = size(data)[2]
        # drop target column
        data = data[:, (1:end) .!= num_cols]
        
        # read solutions
        global solutions = nothing
        if isfile(scaled_dir * current_dataset_name * ".csv")
            @info "Scaled scorings for Dataset: $current_dataset_name already calculated. Reading scaled scores directly from file.."
            solutions = readdlm(scaled_dir * current_dataset_name * ".csv",',', Float64, '\n', header=true)
        else
            if isfile(scorings_dir * current_dataset_name * ".csv")
                @info "Scaled scorings for Dataset: $current_dataset_name missing. Scaling scores.."
                solutions = readdlm(scorings_dir * current_dataset_name * ".csv",',', Float64, '\n', header=true)
                Ireos.normalize_solutions!(solutions, norm_method)
                writedlm(scaled_dir * current_dataset_name * ".csv", vcat(solutions[2], solutions[1]), ",")
            else
                @warn "Scaled scorings for Dataset: $current_dataset_name failed. No scores found."
            end
        end

        for i in 1:ITERATIONS
            println("\n--------------------------------------------SEQUENTIAL--------------------------------------------\n")
            for clf in clfs
                for adaptive_quads_enabled_mode in adaptive_quads_modes
                    for window_mode in window_modes
                        ireos_file_name = ireos_dir * current_dataset_name * "-" * clf * "-" * string(window_mode) * "-" * string(adaptive_quads_enabled_mode) * "-sequential.csv"
                        time = @elapsed begin 
                            # train sequential ireos
                            results, trained = execute_sequential_experiment(data, solutions, clf, gamma_min, tol, isnothing(window_mode) ? 1.0 : window_mode, adaptive_quads_enabled_mode)
                        end
                        println(time)
                        # push row in aggregated results dataframe
                        if !isnothing(results)
                            push_row!(result_df, current_dataset_name, clf, adaptive_quads_enabled_mode, window_mode, false, time, results)
                        else 
                            @warn "Sequential results for Dataset: $current_dataset_name is nothing!"
                        end
                        # write trained ireos probability vector to file
                        if persist_intermediate_result
                            writedlm(ireos_file_name , trained)
                        end
                    end
                end
            end
            println("\n--------------------------------------------PARALLEL--------------------------------------------\n")
            for clf in clfs_par
                ireos_file_name = ireos_dir * current_dataset_name * "-" * clf * "-parallel.csv"
                gamma_max = calculate_gamma_max(data, clf, true)
                time = @elapsed begin 
                    # train parallel irels
                    results, trained = execute_parallel_experiment(data, solutions, clf, gamma_min, gamma_max, tol)
                end
                println(time)
                # push row in aggregated results dataframe
                if !isnothing(results)
                    push_row!(result_df, current_dataset_name, clf, true, 1.0 , true, time, results)
                else
                    @warn "Parallel results for Dataset: $current_dataset_name is nothing!"
                end
                # write trained ireos probability vector to file
                if persist_intermediate_result
                    writedlm(ireos_file_name , trained)
                end
            end
        end
        #Uncomment for saving evaluated solutions
        CSV.write(result_file_name, result_df)
    end
end

# create empty results dataframe with predefined column names for aggregated scores
create_empty_result_df() = DataFrame(dataset = [], clf = String[], adaptive_quads_enabled = Bool[], window_ratio = Float64[], is_parallel = Bool[], time = Float64[]
, ireos_sdo = Float64[], ireos_abod = Float64[], ireos_hbos = Float64[], ireos_iforest = Float64[], ireos_knn = Float64[], ireos_lof = Float64[], ireos_ocsvm = Float64[])

# push row with values with fallback
push_row!(df, current_dataset_name, clf, adaptive_quads_enabled_mode, window_mode, is_parallel, time, results) = push!(df, (current_dataset_name, clf, adaptive_quads_enabled_mode, isnothing(window_mode) ? 1.0 : window_mode, is_parallel, time
, !isnothing(results[1]) ? results[1] : -1.0, length(results) > 1 && !isnothing(results[2]) ? results[2] : -1.0
, length(results) > 2 && !isnothing(results[3]) ? results[3] : -1.0, length(results) > 3 && !isnothing(results[4]) ? results[4] : -1.0
, length(results) > 4 && !isnothing(results[5]) ? results[5] : -1.0, length(results) > 5 && !isnothing(results[6]) ? results[6] : -1.0
, length(results) > 6 && !isnothing(results[7]) ? results[7] : -1.0))

# calculate gamma_max for given classifier
function calculate_gamma_max(X, clf, adaptive_quads_enabled)
    if clf in ["liblinear", "decision_tree_native", "decision_tree_sklearn", "random_forest_native", "random_forest_sklearn", "xgboost_tree", "xgboost_dart", "xgboost_linear"]
        return 1.0
    elseif adaptive_quads_enabled
        # rowwise distance
        return maximum(pairwise(Euclidean(), data, dims=1))
    else
        # default gamma is 1 / num_dims
        return 1/ size(X)[2]
    end
end

# function for executing one calculation of sequential ireos ( calculating parameters -> training ireos -> evaluating solutions)
function execute_sequential_experiment(data, solutions, clf, gamma_min, tol, window_ratio, adaptive_quads_enabled)
    gamma_max = calculate_gamma_max(data, clf, adaptive_quads_enabled)
    if isnothing(window_ratio)
        window_size = nothing
    else
        window_size = round(Int32, window_ratio * size(data)[1])
    end
    trained = Ireos.ireos(data, clf, gamma_min, gamma_max, tol, window_size, adaptive_quads_enabled)
    eval_gamma = gamma_max
    if !adaptive_quads_enabled
        eval_gamma = 1.0
    end
    return create_result(Ireos.evaluate_solutions, trained, solutions, gamma_min, eval_gamma)
end

# function for executing one calculation of parallel ireos ( calculating parameters -> training ireos -> evaluating solutions)
function execute_parallel_experiment(data, solutions, clf, gamma_min, gamma_max, tol)
    trained = IreosPar.ireos_par(data, clf, gamma_min, gamma_max, tol)
    return create_result(IreosPar.evaluate_solutions_par, trained, solutions, gamma_min, gamma_max)
end

# helper function for nullsave evaluation of solutions 
create_result(ireos_evaluation_func, trained, solutions, gamma_min, gamma_max) = isnothing(solutions) ? nothing : 
ireos_evaluation_func(trained, solutions[1]', gamma_min, gamma_max) , trained / (gamma_max - gamma_min)

# change default for `seconds` to 2.5
#println(nworkers())
println(Threads.nthreads())
run()
#rmprocs(4)