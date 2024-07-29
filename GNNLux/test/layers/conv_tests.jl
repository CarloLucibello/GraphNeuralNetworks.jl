@testitem "layers/conv" setup=[SharedTestSetup] begin
    rng = StableRNG(1234)
    g = rand_graph(10, 40, seed=1234)
    x = randn(rng, Float32, 3, 10)

    @testset "GCNConv" begin
        l = GCNConv(3 => 5, relu)
        test_lux_layer(rng, l, g, x, outputsize=(5,))
    end

    @testset "ChebConv" begin
        l = ChebConv(3 => 5, 2)
        test_lux_layer(rng, l, g, x, outputsize=(5,))
    end

    @testset "GraphConv" begin
        l = GraphConv(3 => 5, relu)
        test_lux_layer(rng, l, g, x, outputsize=(5,))
    end

    @testset "AGNNConv" begin
        l = AGNNConv(init_beta=1.0f0)
        test_lux_layer(rng, l, g, x, sizey=(3,10))
    end

    @testset "EdgeConv" begin
        nn = Chain(Dense(6 => 5, relu), Dense(5 => 5))
        l = EdgeConv(nn, aggr = +)
        test_lux_layer(rng, l, g, x, sizey=(5,10), container=true)
    end

    @testset  "CGConv" begin
        l = CGConv(3 => 3, residual = true)
        test_lux_layer(rng, l, g, x, outputsize=(3,), container=true)
    end
end
