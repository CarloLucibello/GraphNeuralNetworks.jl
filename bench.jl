##
using GraphNeuralNetworks
using CUDA
using Flux

N = 10000
M = 1
I = 10

function test_mem(n, make_data)
    for i in 1:I
        g = make_data()
        x = n(g |> gpu)
        # b_g = Flux.batch([g for i in 1:M]) |> gpu
        # x = n(b_g)
        CUDA.memory_status()
    end
end

GC.gc(); CUDA.reclaim(); GC.gc();
CUDA.memory_status()

println("GNN:")
make_data() = GNNGraph(collect(1:N-1), collect(2:N), num_nodes = N, ndata = rand(1, N))
n = GNNChain(Dense(1, 1000), Dense(1000, 1)) |> gpu
# n = GCNConv(1 => 1000) |> gpu

CUDA.@time test_mem(n, make_data)


# GC.gc(); CUDA.reclaim(); GC.gc();
# println("\n\nNN:")
# make_data() = rand(3*N,1)
# n = Chain(Dense(3*N, 1000), Dense(1000, 1)) |> gpu
# CUDA.@time test_mem(n, make_data)

println("################################")