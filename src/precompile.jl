
function workflow1()
    nnodes, d = 10, 6
    ngraphs = 5
    g = Flux.batch([rand_graph(nnodes, 3*nnodes) for i in 1:ngraphs])
    x = rand(Float32, d, g.num_nodes)
    model = GNNChain(GCNConv(d => d, relu), 
                    GraphConv(d => d, tanh),
                    GATv2Conv(d => d ÷ 2, relu, heads=2), 
                    GlobalPool(max), 
                    Dense(d, 1))
    y = model(g, x)
    grad = gradient(m -> sum(m(g, x)), model)[1]
end

workflow1()