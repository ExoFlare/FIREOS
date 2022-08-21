include("ireos.jl")
using Main.Ireos

include("ireos_par.jl")
using Main.IreosPar

using Base.Threads
using CSV
using DelimitedFiles
using Logging
using DataFrames
using Distributed
using ScikitLearn
using Distances
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--data", "-d"
            help = "data input file(s)"
            arg_type = String
            required = true
            nargs = '+'
        "--clf", "-c"
            help = "classifier(s) calling sequential (main) fireos interface\nFeatured classifiers:\n Decision Trees: <decision_tree_native> <decision_tree_sklearn>\nRandom Forests: <random_forest_native> <random_forest_sklearn>\n<liblinear>\nXGBoost: <xgboost_tree> <xgboost_dart> <xgboost_linear>"
            arg_type = String
            required = true
            nargs = '+'
        "--clf-par", "-p"
            help = "classifier(s) calling parallel fireos interface\nFeatures classifiers:\nSupport Vector Machines: <svc> <libsvm>\nLogistic Regression: <logreg> <klr>"
            arg_type = String
            nargs = '*'
        "--solution", "-s"
            help = "scoring(s) file(s)"
            arg_type = String
            nargs = '*'
        "--scaling-method"
            help = "method of scaling solution files. Scaled solution files are stored with datasetname + _scaled.csv.\n Supported scaling methods: <normalization> <standardization>"
            arg_type = String
            default = nothing
        "--gamma-max", "-g"
            help = "maximum gamma value (only relevant for classifiers using a gamma parameter)"
            arg_type = Float64
            default = 1.0
        "--tol", "-t"
            help = "maximum error rate (only relevant for classifiers using adaptive quadrature)"
            arg_type = Float64
            default = 0.005
        "--window-ratio", "-w"
            help = "window ratio of sliding window"
            arg_type = Float64
        "--adaptive-quads-enabled"
            help = "enable adaptive quadrature manually"
            action = :store_true
        "--persist-trained-ireos"
            help = "persist trained ireos file. Trained files are stored with datasetname + _trained.csv"
            action = :store_true
        "--drop-last-column"
            help = "drop last column of input data (might be target column which is not considered in internal validation indices)"
            action = :store_true
        "--show-debug-logs"
            help = "enable debug messages"
            action = :store_true
    end

    return parse_args(s)
end

function run()
     
    # parsing program arguments
    parsed_args = parse_commandline()

    # log level
    if parsed_args["show-debug-logs"]
        ENV["JULIA_DEBUG"] = Main
        println("Parsed args:")
        for (arg,val) in parsed_args
            println("  $arg  =>  $val")
        end
    else
        ENV["JULIA_INFO"] = Main
    end

    # suffixes for result file names
    global trained_suffix = "_trained"
    global evaluated_suffix = "_evaluated"
    global scaled_suffix = "_scaled"

    # main and mandatory parameters
    global names = parsed_args["data"]
    global clfs = parsed_args["clf"]
    global clfs_par = parsed_args["clf-par"]
    global solution_names = parsed_args["solution"]
    global scaling_method = parsed_args["scaling-method"]

    # additional ireos parameters and settings
    global gamma_max = parsed_args["gamma-max"]
    global tol = parsed_args["tol"]
    global persist_trained_ireos = parsed_args["persist-trained-ireos"]
    global window_ratio = parsed_args["window-ratio"]
    global adaptive_quads_enabled = parsed_args["adaptive-quads-enabled"]
    global drop_last_solumn = parsed_args["drop-last-column"]

    global gamma_min = 0.0

    @debug "Printing parameters: \ndata: $names\nclfs: $clfs\nclfs_par: $clfs_par\nsolution: $solution_names\nscaling_method: $scaling_method\ngamma_max: $gamma_max\npersist_trained_ireos: $persist_trained_ireos\nwindow_ratio: $window_ratio\nadaptive_quads_enabled: $adaptive_quads_enabled"
    if !(size(names) == size(solution_names))
        @error "Number of datasets and solutions must be equals! Aborting calculation.."
        return
    end

    for d in 1:lastindex(names)
        current_dataset_name = names[d]
        result_df = create_empty_result_df()
        # check if dataset already calculated
        result_file_name = current_dataset_name * evaluated_suffix * ".csv"
        if isfile(result_file_name)
            @info "IREOS for Dataset: $current_dataset_name already calculated. Skipping.."
            continue
        end
        # read dataset
        global data = readdlm(current_dataset_name,',', Float64, '\n')
        global num_cols = size(data)[2]
        # drop target column
        if drop_last_solumn
            data = data[:, (1:end) .!= num_cols]
        end
        
        # read solutions
        
        global solutions = nothing
        if !isnothing(solution_names) && !isempty(solution_names) && isfile(solution_names[d])
            current_solution_name = solution_names[d]
            @info "Reading solution: $current_solution_name"
            solutions = readdlm(current_solution_name,',', Float64, '\n', header=true)
            if !isnothing(scaling_method)
                @info "Scaling scores of file $current_solution_name with scaling method $scaling_method"
                Ireos.normalize_solutions!(solutions, scaling_method)
                writedlm(current_solution_name * scaled_suffix * ".csv", vcat(solutions[2], solutions[1]), ",")
            end
        else
            @warn "No scorings for dataset $current_dataset_name found. Ireos proceeds with training."
        end

        for clf in clfs
            ireos_file_name = current_dataset_name * "-" * clf * trained_suffix * "-sequential.csv"
            time = @elapsed begin 
                # train sequential ireos
                results, trained = execute_sequential_experiment(data, solutions, clf, gamma_min, tol, isnothing(window_ratio) ? 1.0 : window_ratio, adaptive_quads_enabled)
            end
            # push row in aggregated results dataframe
            push_row!(result_df, current_dataset_name, clf, adaptive_quads_enabled, window_ratio, false, time, results)

            # write trained ireos probability vector to file
            if persist_trained_ireos
                writedlm(ireos_file_name , trained)
            end
        end
        for clf in clfs_par
            ireos_file_name = current_dataset_name * "-" * clf * trained_suffix * "-parallel.csv"
            gamma_max = calculate_gamma_max(data, clf, true)
            time = @elapsed begin 
                # train parallel irels
                results, trained = execute_parallel_experiment(data, solutions, clf, gamma_min, gamma_max, tol)
            end
            # push row in aggregated results dataframe
            push_row!(result_df, current_dataset_name, clf, true, 1.0 , true, time, results)

            # write trained ireos probability vector to file
            if persist_trained_ireos
                writedlm(ireos_file_name , trained)
            end
        end
        #Uncomment for saving evaluated solutions
        CSV.write(result_file_name, result_df)
        @info "Fireos calculation of dataset $current_dataset_name finished. Results are stored in $result_file_name."
    end
end

# create empty results dataframe with predefined column names for aggregated scores
create_empty_result_df() = DataFrame(dataset = [], clf = String[], adaptive_quads_enabled = Bool[], window_ratio = Float64[], is_parallel = Bool[], time = Float64[]
, ireos_sdo = Float64[], ireos_abod = Float64[], ireos_hbos = Float64[], ireos_iforest = Float64[], ireos_knn = Float64[], ireos_lof = Float64[], ireos_ocsvm = Float64[])

# push row with values with fallback
push_row!(df, current_dataset_name, clf, adaptive_quads_enabled_mode, window_ratio, is_parallel, time, results) = push!(df, (current_dataset_name, clf, adaptive_quads_enabled_mode, isnothing(window_ratio) ? 1.0 : window_ratio, is_parallel, time
, !isnothing(results) && !isnothing(results[1]) ? results[1] : -1.0, !isnothing(results) && length(results) > 1 && !isnothing(results[2]) ? results[2] : -1.0
, !isnothing(results) && length(results) > 2 && !isnothing(results[3]) ? results[3] : -1.0, !isnothing(results) && length(results) > 3 && !isnothing(results[4]) ? results[4] : -1.0
, !isnothing(results) && length(results) > 4 && !isnothing(results[5]) ? results[5] : -1.0, !isnothing(results) && length(results) > 5 && !isnothing(results[6]) ? results[6] : -1.0
, !isnothing(results) && length(results) > 6 && !isnothing(results[7]) ? results[7] : -1.0))

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


println("Number of parallel Threads: " * string(Threads.nthreads()))
run()