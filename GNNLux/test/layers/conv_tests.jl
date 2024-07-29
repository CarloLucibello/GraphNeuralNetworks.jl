@testitem "layers/conv" setup=[SharedTestSetup] begin
    rng = StableRNG(1234)
    g = rand_graph(10, 40, seed=1234)
    in_dims = 3
    out_dims = 5
    x = randn(rng, Float32, in_dims, 10)

    @testset "GCNConv" begin
        l = GCNConv(in_dims => out_dims, relu)
        test_lux_layer(rng, l, g, x, outputsize=(out_dims,))
    end

    @testset "ChebConv" begin
        l = ChebConv(in_dims => out_dims, 2)
        test_lux_layer(rng, l, g, x, outputsize=(out_dims,))
    end

    @testset "GraphConv" begin
        l = GraphConv(in_dims => out_dims, relu)
        test_lux_layer(rng, l, g, x, outputsize=(out_dims,))
    end

    @testset "AGNNConv" begin
        l = AGNNConv(init_beta=1.0f0)
        test_lux_layer(rng, l, g, x, sizey=(in_dims, 10))
    end

    @testset "EdgeConv" begin
        nn = Chain(Dense(2*in_dims => 5, relu), Dense(5 => out_dims))
        l = EdgeConv(nn, aggr = +)
        test_lux_layer(rng, l, g, x, sizey=(out_dims,10), container=true)
    end

    @testset  "CGConv" begin
        l = CGConv(in_dims => in_dims, residual = true)
        test_lux_layer(rng, l, g, x, outputsize=(in_dims,), container=true)
    end

    @testset "DConv" begin
        l = DConv(in_dims => out_dims, 2)
        test_lux_layer(rng, l, g, x, outputsize=(5,))
    end

    @testset "EGNNConv" begin
        hin = 6
        hout = 7
        hidden = 8
        l = EGNNConv(hin => hout, hidden)
        ps = LuxCore.initialparameters(rng, l)
        st = LuxCore.initialstates(rng, l)
        h = randn(rng, Float32, hin, g.num_nodes)
        (hnew, xnew), stnew = l(g, h, x, ps, st)
        @test size(hnew) == (hout, g.num_nodes)
        @test size(xnew) == (in_dims, g.num_nodes)
    end

    @testset "GATConv" begin
        x = randn(rng, Float32, 6, 10)
        l = GATConv(6 => 8, heads=2)
        test_lux_layer(rng, l, g, x, outputsize=(8,))
    end
end

