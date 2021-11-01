using Flux, GraphNeuralNetworks, Graphs, BenchmarkTools, CUDA
using DataFrames, Statistics, JLD2, SparseArrays
using Unitful
# CUDA.device!(2)
CUDA.allowscalar(false)

function getres(res, str)
    ismissing(res[str]) && return missing 
    t = median(res[str]).time
    if t < 1e3
        t * u"ns"
    elseif t < 1e6
        t / 1e3 * u"μs"
    elseif t < 1e9
        t / 1e6 * u"ms"
    else
        t / 1e9 * u"s"
    end
end

function run_single_benchmark(N, c, D, CONV; gtype=:lg)
    X = randn(Float32, D, N)

    data = erdos_renyi(N, c / (N-1), seed=17)
    g = GNNGraph(data; ndata=X, graph_type=gtype)
    
    # g = rand_graph(N, c*N; ndata=X, graph_type=gtype)
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
        res["GPU_GRAD"] = @benchmark CUDA.@sync(gradient(() -> sum($m_gpu($g_gpu).ndata.x), $ps_gpu)) teardown=(GC.gc(); CUDA.reclaim())
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

Benchmark GNN layers on Erdos-Renyi ranomd graphs 
with average degree `c`. Benchmarks are perfomed for each graph size in the list `Ns`.
`D` is the number of node features.
"""
function run_benchmarks(; 
        Ns = [10, 100, 1000, 10000],
        c = 6,
        D = 100,
        layers = [GCNConv, GATConv],
        gtypes = [:coo],
        )

    df = DataFrame(N=Int[], c=Int[], layer=String[], gtype=Symbol[], 
                   time_fwd_cpu=Any[], time_fwd_gpu=Any[],
                   time_grad_cpu=Any[], time_grad_gpu=Any[])
    
    for gtype in gtypes
        for N in Ns
            println("## GRAPH_TYPE = $gtype  N = $N")           
            for CONV in layers
                res = run_single_benchmark(N, c, D, CONV; gtype)
                row = (;layer = "$CONV", 
                        N = N,
                        c = c,
                        gtype = gtype, 
                        time_fwd_cpu = getres(res, "CPU_FWD"),
                        time_fwd_gpu = getres(res, "GPU_FWD"),
                        time_grad_cpu = getres(res, "CPU_GRAD"),
                        time_grad_gpu = getres(res, "GPU_GRAD"),
                    )
                push!(df, row)
                println(row)
            end
        end
    end

    df.grad_gpu_to_cpu = NoUnits.(df.time_grad_gpu ./ df.time_grad_cpu)
    sort!(df, [:layer, :N, :c, :gtype])
    return df
end

df = run_benchmarks()
for g in groupby(df, :layer); println(g, "\n"); end

# @save "master_2021_11_01_arrakis.jld2" dfmaster=df
## or
# @save "pr.jld2" dfpr=df


function compare(dfpr, dfmaster; on=[:N, :c, :gtype, :layer])
    df = outerjoin(dfpr, dfmaster; on=on, makeunique=true, renamecols = :_pr => :_master)
    df.pr_to_master_cpu = df.time_cpu_pr ./ df.time_cpu_master
    df.pr_to_master_gpu = df.time_gpu_pr ./ df.time_gpu_master 
    return df[:,[:N, :c, :gtype, :layer, :pr_to_master_cpu, :pr_to_master_gpu]]
end

# @load "perf/perf_pr.jld2" dfpr
# @load "perf/perf_master.jld2" dfmaster
# compare(dfpr, dfmaster)
