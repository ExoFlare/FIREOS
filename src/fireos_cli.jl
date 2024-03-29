include("FIREOS.jl")
using .FIREOS

include("fireos_utils.jl")

using Base.Threads
using CSV
using DelimitedFiles
using Logging
using DataFrames
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
            help = "classifier(s) calling (main) fireos interface\nFeatured classifiers:\nSupport Vector Machines: <svc> <logreg> <klr> <libsvm>\n Decision Trees: <decision_tree_native> <decision_tree_sklearn>\nRandom Forests: <random_forest_native> <random_forest_sklearn>\n<liblinear>\nXGBoost: <xgboost_tree> <xgboost_dart> <xgboost_linear>"
            arg_type = String
            required = true
            nargs = '+'
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
        "--use-parallel", "-p"
            help = "Using mulitthreaded fireos interface"
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

    # main fireos parameters
    global names = parsed_args["data"]
    global clfs = parsed_args["clf"]
    global solution_names = parsed_args["solution"]
    global scaling_method = parsed_args["scaling-method"]
    global use_parallel = parsed_args["use-parallel"]

    # additional ireos parameters and settings
    global gamma_max = parsed_args["gamma-max"]
    global tol = parsed_args["tol"]
    global persist_trained_ireos = parsed_args["persist-trained-ireos"]
    global window_ratio = parsed_args["window-ratio"]
    global adaptive_quads_enabled = parsed_args["adaptive-quads-enabled"]
    global drop_last_solumn = parsed_args["drop-last-column"]

    global gamma_min = 0.0

    @debug "Printing parameters: \ndata: $names\nclfs: $clfs\nsolution: $solution_names\nscaling_method: $scaling_method\ngamma_max: $gamma_max\nuse_parallel: $use_parallel\npersist_trained_ireos: $persist_trained_ireos\nwindow_ratio: $window_ratio\nadaptive_quads_enabled: $adaptive_quads_enabled"
    if (length(solution_names) > 0) && (!(size(names) == size(solution_names)))
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
                if !use_parallel
                    FIREOS.normalize_solutions!(solutions, scaling_method)
                else
                    FIREOS.normalize_solutions_par!(solutions, scaling_method)
                end
                writedlm(current_solution_name * scaled_suffix * ".csv", vcat(solutions[2], solutions[1]), ",")
            end
        else
            @warn "No scorings for dataset $current_dataset_name found. Ireos proceeds with training."
        end

        for clf in clfs
            if !use_parallel
                ireos_file_name = current_dataset_name * "-" * clf * trained_suffix * "-sequential.csv"
                time = @elapsed begin 
                    # train sequential ireos
                    results, trained = execute_sequential_experiment(data, solutions, clf, gamma_min, gamma_max, tol, isnothing(window_ratio) ? 1.0 : window_ratio, adaptive_quads_enabled)
                end
                # push row in aggregated results dataframe
                push_row!(result_df, current_dataset_name, clf, adaptive_quads_enabled, window_ratio, false, time, results)
            else
                ireos_file_name = current_dataset_name * "-" * clf * trained_suffix * "-parallel.csv"
                time = @elapsed begin 
                    # train parallel irels
                    results, trained = execute_parallel_experiment(data, solutions, clf, gamma_min, gamma_max, tol, isnothing(window_ratio) ? 1.0 : window_ratio, adaptive_quads_enabled)
                end
                # push row in aggregated results dataframe
                push_row!(result_df, current_dataset_name, clf, adaptive_quads_enabled, window_ratio, true, time, results)
            end
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




println("Number of parallel Threads: " * string(Threads.nthreads()))
run()