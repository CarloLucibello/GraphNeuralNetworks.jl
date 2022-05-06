using GraphNeuralNetworks, CUDA, Flux

N = 10000
I = 10

function test_mem(n, data)
    for i in 1:I
        y = n(data)
        CUDA.memory_status()
    end
end


GC.gc(); CUDA.reclaim();
println("GNN, memory filling")
g = GNNGraph(collect(1:N-1), collect(2:N), num_nodes = N, ndata = rand(Float32, 1, N)) |> gpu
gnnchain = GNNChain(Dense(1, 1000), Dense(1000, 1)) |> gpu
CUDA.@time test_mem(gnnchain, g)

GC.gc(); CUDA.reclaim();
println("\n\nNN equivalent to GNN, memory filling")
x = g.ndata.x
chain = Chain(gnnchain.layers...)
@assert gnnchain(g).ndata.x ≈ chain(x) 
## same results with these
# x = rand(1, N) |> gpu
# chain = Chain(Dense(1, 1000), Dense(1000, 1)) |> gpu
CUDA.@time test_mem(chain, x)

GC.gc(); CUDA.reclaim();
println("\n\nNN 1,  same memory")
data = rand(N, 1) |> gpu
n = Chain(Dense(N, 1000), Dense(1000, 1)) |> gpu
CUDA.@time test_mem(n, data)

println("\n\nNN 2, memory filling:")
data = rand(N, N) |> gpu
n = Chain(Dense(N, 1000), Dense(1000, 1)) |> gpu
CUDA.@time test_mem(n, data)
