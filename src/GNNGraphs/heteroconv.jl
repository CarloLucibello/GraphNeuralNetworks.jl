using GraphNeuralNetworks
using Random, Statistics

g = rand_heterograph((:user => 10, :movie => 20), (:user, :rate, :movie) => 30)
# g.ndata[:user].x = randn(Float32,3, 10)
d = 3
X = Dict()
for (node_t, node_num) in g.num_nodes
    X[node_t] =  randn(Float32, d, node_num)
end

g[(:user, :rate, :movie)]
