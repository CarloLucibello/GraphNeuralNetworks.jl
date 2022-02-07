using GraphNeuralNetworks # WHY YELLOW LINE? (MISSING REFERENCE)
using Flux, Zygote
using Test

s = [1,2,3]
t = [2,3,1]
w = [0.1,0.1,0.2]
g = GNNGraph(s, t, w)
A = adjacency_matrix(g)



gw = gradient(w) do w
    g = GNNGraph((s, t, w), graph_type=GRAPH_T)
    sum(degree(g, edge_weight=false))
end[1]
@test gw === nothing

gw = gradient(w) do w
    g = GNNGraph((s, t, w), graph_type=GRAPH_T)
    sum(degree(g, edge_weight=true))
end[1]
@test gw isa Vector{Float64}