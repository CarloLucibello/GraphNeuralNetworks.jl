using GraphNeuralNetworks, Graphs

d, n = 3, 5
g = rand_bipartite_heterograph(n, n, 15)
g[:A].x = rand(Float32, d, n)
g[:B].x = rand(Float32, d, n)

model = HeteroGraphConv([(:A,:to,:B) => GraphConv(d => d), 
                         (:B,:to,:A) => GraphConv(d => d)])
x = (A = g[:A].x, B = g[:B].x)
y = model(g, x)