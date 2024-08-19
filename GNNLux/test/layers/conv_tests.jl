@testitem "layers/conv" setup=[SharedTestSetup] begin
    rng = StableRNG(1234)
    edim = 10
    g = rand_graph(rng, 10, 40)
    in_dims = 3
    out_dims = 5
    x = randn(rng, Float32, in_dims, 10)

    g = GNNGraph(g, edata = rand(Float32, edim, g.num_edges)) 

    @testset "GCNConv" begin
        l = GCNConv(in_dims => out_dims, tanh)
        test_lux_layer(rng, l, g, x, outputsize=(out_dims,))
    end

    @testset "ChebConv" begin
        l = ChebConv(in_dims => out_dims, 2)
        test_lux_layer(rng, l, g, x, outputsize=(out_dims,))
    end

    @testset "GraphConv" begin
        l = GraphConv(in_dims => out_dims, tanh)
        test_lux_layer(rng, l, g, x, outputsize=(out_dims,))
    end

    @testset "AGNNConv" begin
        l = AGNNConv(init_beta=1.0f0)
        test_lux_layer(rng, l, g, x, sizey=(in_dims, 10))
    end

    @testset "EdgeConv" begin
        nn = Chain(Dense(2*in_dims => 2, tanh), Dense(2 => out_dims))
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
        test_lux_layer(rng, l, g, x, outputsize=(16,))

        l = GATConv(6 => 8, heads=2, concat=false, dropout=0.5)
        test_lux_layer(rng, l, g, x, outputsize=(8,))

        #TODO test edge
    end

    @testset "GATv2Conv" begin
        x = randn(rng, Float32, 6, 10)

        l = GATv2Conv(6 => 8, heads=2)
        test_lux_layer(rng, l, g, x, outputsize=(16,))

        l = GATv2Conv(6 => 8, heads=2, concat=false, dropout=0.5)
        test_lux_layer(rng, l, g, x, outputsize=(8,))

        #TODO test edge
    end

    @testset "SGConv" begin
        l = SGConv(in_dims => out_dims, 2)
        test_lux_layer(rng, l, g, x, outputsize=(out_dims,))
    end

    @testset "GatedGraphConv" begin
        l = GatedGraphConv(in_dims, 3)
        test_lux_layer(rng, l, g, x, outputsize=(in_dims,))
    end

    @testset "GINConv" begin
        nn = Chain(Dense(in_dims => out_dims, relu), Dense(out_dims => out_dims))
        l = GINConv(nn, 0.5)
        test_lux_layer(rng, l, g, x, sizey=(out_dims,g.num_nodes), container=true)
    end

    @testset "NNConv" begin
        edim = 10
        nn = Dense(edim, out_dims * in_dims)
        l = NNConv(in_dims => out_dims, nn, tanh, aggr = +)
        test_lux_layer(rng, l, g, x, sizey=(out_dims, g.num_nodes), container=true)
    end
end
