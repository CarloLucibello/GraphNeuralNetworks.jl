using GraphNeuralNetworks, Flux

chain = GNNChain(GraphConv(2=>2))

params, restructure = Flux.destructure(chain)

restructure(params)