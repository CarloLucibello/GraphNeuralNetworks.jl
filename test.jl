using GraphNeuralNetworks
using Flux, Zygote
s = [1,2,3]
t = [2,3,1]
w = [0.1,0.1,0.2]
g = GNNGraph(s, t, w)
A = adjacency_matrix(g)
gradient(g -> sum(adjacency_matrix(g)), g)
