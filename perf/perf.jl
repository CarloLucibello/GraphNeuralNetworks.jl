using Flux, GraphNeuralNetworks, Graphs, BenchmarkTools, CUDA
using DataFrames, Statistics, JLD2, SparseArrays
CUDA.device!(2)
CUDA.allowscalar(false)

BenchmarkTools.ratio(::Missing, x) = Inf
BenchmarkTools.ratio(x, ::Missing) = 0.0
BenchmarkTools.ratio(::Missing, ::Missing) = missing

function run_single_benchmark(N, c, D, CONV; gtype = :lg)
    data = erdos_renyi(N, c / (N - 1), seed = 17)
    X = randn(Float32, D, N)

    g = GNNGraph(data; ndata = X, graph_type = gtype)
    g_gpu = g |> gpu

    m = CONV(D => D)
    ps = Flux.params(m)

    m_gpu = m |> gpu
    ps_gpu = Flux.params(m_gpu)

    res = Dict()

    res["CPU_FWD"] = @benchmark $m($g)
    res["CPU_GRAD"] = @benchmark gradient(() -> sum($m($g).ndata.x), $ps)

    try
        res["GPU_FWD"] = @benchmark CUDA.@sync($m_gpu($g_gpu)) teardown=(GC.gc(); CUDA.reclaim())
    catch
        res["GPU_FWD"] = missing
    end

    try
        res["GPU_GRAD"] = @benchmark CUDA.@sync(gradient(() -> sum($m_gpu($g_gpu).ndata.x),
                                                         $ps_gpu)) teardown=(GC.gc(); CUDA.reclaim())
    catch
        res["GPU_GRAD"] = missing
    end

    return res
end

"""
    run_benchmarks(;
        Ns = [10, 100, 1000, 10000],
        c = 6,
        D = 100,
        layers = [GCNConv, GraphConv, GATConv]
        )

Benchmark GNN layers on Erdos-Renyi random graphs 
with average degree `c`. Benchmarks are performed for each graph size in the list `Ns`.
`D` is the number of node features.
"""
function run_benchmarks(;
                        Ns = [10, 100, 1000, 10000],
                        c = 6,
                        D = 100,
                        layers = [GCNConv, GATConv],
                        gtypes = [:coo, :sparse, :dense])
    df = DataFrame(N = Int[], c = Float64[], layer = String[], gtype = Symbol[],
                   time_cpu = Any[], time_gpu = Any[]) |> allowmissing

    for gtype in gtypes
        for N in Ns
            println("## GRAPH_TYPE = $gtype  N = $N")
            for CONV in layers
                res = run_single_benchmark(N, c, D, CONV; gtype)
                row = (; layer = "$CONV",
                       N = N,
                       c = c,
                       gtype = gtype,
                       time_cpu = ismissing(res["CPU"]) ? missing : median(res["CPU"]),
                       time_gpu = ismissing(res["GPU"]) ? missing : median(res["GPU"]))
                push!(df, row)
            end
        end
    end

    df.gpu_to_cpu = ratio.(df.time_gpu, df.time_cpu)
    sort!(df, [:layer, :N, :c, :gtype])
    return df
end

# df = run_benchmarks()
# for g in groupby(df, :layer); println(g, "\n"); end

# @save "perf/perf_master_20210803_carlo.jld2" dfmaster=df
## or
# @save "perf/perf_pr.jld2" dfpr=df

function compare(dfpr, dfmaster; on = [:N, :c, :gtype, :layer])
    df = outerjoin(dfpr, dfmaster; on = on, makeunique = true,
                   renamecols = :_pr => :_master)
    df.pr_to_master_cpu = ratio.(df.time_cpu_pr, df.time_cpu_master)
    df.pr_to_master_gpu = ratio.(df.time_gpu_pr, df.time_gpu_master)
    return df[:, [:N, :c, :gtype, :layer, :pr_to_master_cpu, :pr_to_master_gpu]]
end

# @load "perf/perf_pr.jld2" dfpr
# @load "perf/perf_master.jld2" dfmaster
# compare(dfpr, dfmaster)
