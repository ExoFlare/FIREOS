"""
Main evaluator class. This class tests all implemented predictors and sampling methods and windows using 80 datasets, writing all 
intermediate results into different folders.
"""

include("FIREOS.jl")
using .FIREOS

include("fireos_utils.jl")

using CSV
using Base.Threads
using DelimitedFiles
using Logging
using Distances
using DataFrames

const ITERATIONS = 1

function run()
     
    #ENV["JULIA_DEBUG"] = Main
    ENV["JULIA_INFO"] = Main

    persist_intermediate_result = true

    data_dir = "data/"
    scorings_dir = "scores/"
    scaled_dir = scorings_dir * "scaled/"
    results_dir = "results/"
    ireos_dir = results_dir * "ireos/"
    internal_validation_indices_dir = "internal_validation_indices/"

    #=global names = ["complex_1", "complex_2", "complex_3", "complex_4", "complex_5", "complex_6", "complex_7", "complex_8", "complex_9", "complex_10",
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
    "low-noise_17", "low-noise_18", "low-noise_19", "low-noise_20", "separated_20"]=#

    names = ["complex_1", "complex_2", "complex_3", "complex_4", "complex_5", "complex_6", "complex_7", "complex_8", "complex_9", "complex_10",
	"complex_11", "complex_12", "complex_13", "complex_14", "complex_15", "complex_16",
    "complex_17", "complex_18", "complex_19", "complex_20", "high-noise_1", "high-noise_2", "high-noise_3",
	"high-noise_4", "high-noise_5", "high-noise_6", "high-noise_7", "high-noise_8", "high-noise_9", 
	"high-noise_10", "high-noise_11", "high-noise_12", "high-noise_13", 
    "high-noise_14", "high-noise_15", "high-noise_16", "high-noise_17", "high-noise_18", "high-noise_19", "high-noise_20",
	"low-noise_1", "low-noise_2", "low-noise_3", "low-noise_4", "low-noise_5", "low-noise_6", "low-noise_7", "low-noise_8", "low-noise_9",
	"low-noise_10", "low-noise_11", "low-noise_12", "low-noise_13", "low-noise_14", "low-noise_15", "low-noise_16", 
    "low-noise_17", "low-noise_18", "low-noise_19", "low-noise_20"]

    #=names = ["basic2d_1.csv", "basic2d_10.csv", "basic2d_11.csv", "basic2d_12.csv", "basic2d_13.csv", "basic2d_14.csv", "basic2d_15.csv", "basic2d_16.csv", 
    "basic2d_17.csv", "basic2d_18.csv", "basic2d_19.csv", "basic2d_2.csv", "basic2d_20.csv", "basic2d_3.csv", "basic2d_4.csv", "basic2d_5.csv", "basic2d_6.csv", 
    "basic2d_7.csv", "basic2d_8.csv", "basic2d_9.csv"]=#


    #clfs = ["svc", "logreg", "klr", "libsvm" "decision_tree_native", "decision_tree_sklearn", "random_forest_native", "random_forest_sklearn", "liblinear", "xgboost_tree", "xgboost_dart", "xgboost_linear"]
    clfs = ["decision_tree_native", "decision_tree_sklearn", "random_forest_native", "random_forest_sklearn", "liblinear", "xgboost_tree", "xgboost_dart", "xgboost_linear"]

    window_modes =[0.1, 0.5, nothing]

    # Normalization transforms all scores [0,1], is used in the original IREOS paper
    norm_method = "normalization"

    for d in eachindex(names)
        current_dataset_name = names[d]
        result_df = create_empty_result_df()
        # check if dataset already calculated
        result_file_name = results_dir * internal_validation_indices_dir * current_dataset_name * ".csv"
        if isfile(result_file_name)
            @info "IREOS for Dataset: $current_dataset_name already calculated. Skipping.."
            continue
        end
        # read dataset
        data = readdlm(data_dir * current_dataset_name,',', Float64, '\n')
        num_cols = size(data)[2]
        # drop target column
        data = data[:, (1:end) .!= num_cols]
        
        # read solutions
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
                solutions = nothing
            end
        end

        for i in 1:ITERATIONS
            for clf in clfs
                for window_mode in window_modes
                    println("\n--------------------------------------------SEQUENTIAL--------------------------------------------\n")
                    ireos_file_name_seq = ireos_dir * current_dataset_name * "-" * clf * "-" * string(window_mode) * "-sequential-" * string(i) * ".csv"
                    time = @elapsed begin 
                        # train sequential ireos
                        @time results, trained = execute_sequential_experiment(data, solutions, clf, nothing, nothing, nothing, window_mode, nothing)
                    end
                    if !isnothing(results)
                        push_row!(result_df, current_dataset_name, clf, FIREOS.get_default_adaptive_quads_enabled_for_clf(clf), window_mode, false, time, results)
                    else 
                        @warn "Sequential results for Dataset: $current_dataset_name is nothing!"
                    end
                    println(time)
                    if persist_intermediate_result
                        writedlm(ireos_file_name_seq , trained)
                    end
                    println("\n--------------------------------------------PARALLEL--------------------------------------------\n")
                    ireos_file_name_par = ireos_dir * current_dataset_name * "-" * clf * "-" * string(window_mode) * "-parallel-" * string(i) * ".csv"
                    time = @elapsed begin 
                        # train parallel irels
                        @time results, trained = execute_parallel_experiment(data, solutions, clf, nothing, nothing, nothing, window_mode, nothing)
                    end
                    # push row in aggregated results dataframe
                    if !isnothing(results)
                        push_row!(result_df, current_dataset_name, clf, true, window_mode , true, time, results)
                    else
                        @warn "Parallel results for Dataset: $current_dataset_name is nothing!"
                    end
                    println(time)
                    # write trained ireos probability vector to file
                    if persist_intermediate_result
                        writedlm(ireos_file_name_par , trained)
                    end
                end
            end
        end
        #Uncomment for saving evaluated solutions
        CSV.write(result_file_name, result_df)
    end
end

# change default for `seconds` to 2.5
#println(nworkers())
println(Threads.nthreads())
run()
#rmprocs(4)