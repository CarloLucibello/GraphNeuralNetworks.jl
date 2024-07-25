@testset "GraphConv" begin
    rng = MersenneTwister(1234)
    g = rand_graph(10, 20, ndata= rand(Float32, 3, 10))
    l = GraphConv(3 => 5, relu)
    ps = LuxCore.initialparameters(rng, l)
    st = LuxCore.initialstates(rng, l)
end