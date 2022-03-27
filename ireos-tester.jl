include("ireos.jl")
using Main.Ireos

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
    clfs = ["svc", "logreg", "klr"]

    global gamma_min = 0.0
    global tol = 0.005

    @assert length(data_files) == length(solution_files)
    for i in 1:length(data_files)
        global data = readdlm(data_files[i])
        global solutions = readdlm(solution_files[i],',', Float64, '\n')
        #rowwise distance
        global gamma_max = maximum(pairwise(Euclidean(), data, dims=1))
        suite[data_files[i]] = BenchmarkGroup()
        for j in 1:length(clfs)
            global clf = clfs[j]
            suite[data_files[i]][clf] = #@benchmark begin
            res = Ireos.ireos(data, clf, gamma_min, gamma_max, tol)
            res2 = Ireos.evaluate_solutions(res, solutions, gamma_min, gamma_max)
            #end
        end
    end
    BenchmarkTools.save(benchmark_result_file_name, suite)
end

# change default for `seconds` to 2.5
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 600
#addprocs(4)
run()
