using GraphNeuralNetworks, Flux
using BenchmarkTools
using Test

function unbatchold(g::GNNGraph)
    return [getgraph(g, i) for i in 1:g.num_graphs]
end

n = 100
c = 6
ngraphs = 128
gs = [rand_graph(n, c*n, ndata=rand(64, n), edata=rand(64, c*n)) for _ in 1:ngraphs]
gall = Flux.batch(gs)

@btime Flux.batch($gs);
# 9.858 ms (12857 allocations: 47.69 MiB)
@btime Flux.unbatch($gall);
# 335.357 ms (17801 allocations: 50.69 MiB) # master
# 10.017 ms (11807 allocations: 46.94 MiB) # THIS PR 
@btime unbatchold($gall);

gradient(gall -> sum(Flux.unbatch(gall)[1].ndata.x), gall)

