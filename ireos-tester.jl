include("ireos.jl")
using Main.Ireos

include("ireos_par.jl")
using Main.IreosPar

using DelimitedFiles
using Logging
using BenchmarkTools
using Distances
using Distributed

function run()
    #ENV["JULIA_DEBUG"] = Main
    ENV["JULIA_INFO"] = Main

    benchmark_result_file_name = "benchmarks/Benchmark_IREOS_" * Ireos.VERSION * ".json"
    
    suite = BenchmarkGroup()
    global data_files = ["data/WBC_withoutdupl_norm", "data/mytest.txt", "data/dens-diff_12", "data/high-noise_8"]
    global solution_files = ["data/WBC_withoutdupl_norm_solutions.csv", "data/mytest_solutions.txt", "data/dens-diff_12_solutions.csv", "data/high-noise_8_solutions.csv"]

    #global data_files = ["data/WBC_withoutdupl_norm"]
    #global solution_files = ["data/WBC_withoutdupl_norm_solutions.csv"]
    clfs = ["svc", "logreg", "klr", "libsvm"]
    clfs_par = ["libsvm"]

    global gamma_min = 0.0
    global tol = 0.005

    @assert length(data_files) == length(solution_files)
    for i in 1:length(data_files)
        global data = readdlm(data_files[i])
        global num_rows = size(data)[1]
        global solutions = readdlm(solution_files[i],',', Float64, '\n')
        #rowwise distance
        global gamma_max = maximum(pairwise(Euclidean(), data, dims=1))
        suite[data_files[i]] = BenchmarkGroup()
        println("\n--------------------------------------------Sequencial--------------------------------------------\n")
        """for j in 1:length(clfs)
            global clf = clfs[j]
            suite[data_files[i]][clf] = @time begin
            res = Ireos.ireos(data, clf, gamma_min, gamma_max, tol)
            res2 = Ireos.evaluate_solutions(res, solutions, gamma_min, gamma_max)
            end
        end
        println("\n--------------------------------------------PARALLEL--------------------------------------------\n")
        for j in 1:length(clfs_par)
            global clf_par = clfs_par[j]
            @time begin
            res_par = IreosPar.ireos_par(data, clf_par, gamma_min, gamma_max, tol)
            res2_par = IreosPar.evaluate_solutions_par(res_par, solutions, gamma_min, gamma_max)
            end
        end"""
        window_perc = 0.3
        println("\n--------------------------------------------WINDOWED--------------------------------------------\n")
        for j in 1:length(clfs)
            global clf = clfs[j]
            @time begin
            res3 = Ireos.ireos(data, clf, gamma_min, gamma_max, tol, round(Int32, num_rows*window_perc))
            res4 = Ireos.evaluate_solutions(res3, solutions, gamma_min, gamma_max)
            end
        end
    end
    BenchmarkTools.save(benchmark_result_file_name, suite)
end

# change default for `seconds` to 2.5
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 600
#addprocs(4)
run()
