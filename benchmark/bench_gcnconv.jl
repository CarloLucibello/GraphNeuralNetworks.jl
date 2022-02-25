module BenchGCNConv
using GraphNeuralNetworks
using Random, Statistics, SparseArrays
using BenchmarkTools
using Flux, Graphs


function get_single_benchmark(sub, N, c, D, CONV; gtype=:lg)
    data = erdos_renyi(N, c / (N-1), seed=17)
    X = randn(Float32, D, N)
    
    g = GNNGraph(data; ndata=X, graph_type=gtype)
    g_gpu = g |> gpu    
    
    m = CONV(D => D)
    ps = Flux.params(m)
    
    m_gpu = m |> gpu
    ps_gpu = Flux.params(m_gpu)


    sub["CPU_FWD"] = @benchmarkable $m($g)
    sub["CPU_GRAD"] = @benchmarkable gradient(() -> sum($m($g).ndata.x), $ps)
    
    # try
    #     sub["GPU_FWD"] = @benchmark CUDA.@sync($m_gpu($g_gpu)) teardown=(GC.gc(); CUDA.reclaim())
    # catch
    #     sub["GPU_FWD"] = missing
    # end

    # try
    #     sub["GPU_GRAD"] = @benchmark CUDA.@sync(gradient(() -> sum($m_gpu($g_gpu).ndata.x), $ps_gpu)) teardown=(GC.gc(); CUDA.reclaim())
    # catch
    #     # sub["GPU_GRAD"] = missing
    # end

    return sub
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
function get_benchmarks(; 
        # Ns = [10, 100, 1000, 10000],
        Ns = [10, 100],
        c = 6,
        D = 100,
        # layers = [GCNConv, GATConv],
        # gtypes = [:coo, :sparse, :dense],
        layers = [GCNConv,],
        gtypes = [:coo,],
        )

        SUITE = BenchmarkGroup()
    
    for gtype in gtypes
        for N in Ns
            for CONV in layers
                sub = SUITE["$(gtype)_$(N)_$(CONV)"] = BenchmarkGroup()
                get_single_benchmark(sub, N, c, D, CONV; gtype)
            end
        end
    end
    return SUITE
end

SUITE = get_benchmarks()

end

BenchGCNConv.SUITE