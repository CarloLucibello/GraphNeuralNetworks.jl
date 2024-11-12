RTOL_LOW = 1e-2
RTOL_HIGH = 1e-5
ATOL_LOW = 1e-3

@testset "GCNConv" begin
    l = GCNConv(D_IN => D_OUT)
    for g in test_graphs
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (D_OUT, g.num_nodes))
    end

    l = GCNConv(D_IN => D_OUT, tanh, bias = false)
    for g in test_graphs
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (D_OUT, g.num_nodes))
    end

    l = GCNConv(D_IN => D_OUT, add_self_loops = false)
    test_layer(l, g1, rtol = RTOL_HIGH, outsize = (D_OUT, g1.num_nodes))

    @testset "edge weights & custom normalization" begin
        s = [2, 3, 1, 3, 1, 2]
        t = [1, 1, 2, 2, 3, 3]
        w = Float32[1, 2, 3, 4, 5, 6]
        g = GNNGraph((s, t, w), ndata = ones(Float32, 1, 3), graph_type = GRAPH_T)
        x = g.ndata.x
        custom_norm_fn(d) = 1 ./ sqrt.(d)  
        l = GCNConv(1 => 1, add_self_loops = false, use_edge_weight = true)
        l.weight .= 1
        d = degree(g, dir = :in, edge_weight = true)
        y = l(g, x)
        @test y[1, 1] ≈ w[1] / √(d[1] * d[2]) + w[2] / √(d[1] * d[3])
        @test y[1, 2] ≈ w[3] / √(d[2] * d[1]) + w[4] / √(d[2] * d[3])
        @test y ≈ l(g, x, w; norm_fn = custom_norm_fn) # checking without custom

        # test gradient with respect to edge weights
        w = rand(Float32, 6)
        x = rand(Float32, 1, 3)
        g = GNNGraph((s, t, w), ndata = x, graph_type = GRAPH_T, edata = w)
        l = GCNConv(1 => 1, add_self_loops = false, use_edge_weight = true)
        @test gradient(w -> sum(l(g, x, w)), w)[1] isa AbstractVector{Float32}   # redundant test but more explicit
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (1, g.num_nodes), test_gpu = false)
    end

    @testset "conv_weight" begin
         l = GraphNeuralNetworks.GCNConv(D_IN => D_OUT)
        w = zeros(Float32, D_OUT, D_IN)
        g1 = GNNGraph(TEST_GRAPHS[1], ndata = ones(Float32, D_IN, 4))
        @test l(g1, g1.ndata.x, conv_weight = w) == zeros(Float32, D_OUT, 4)
        a = rand(Float32, D_IN, 4)
        g2 = GNNGraph(TEST_GRAPHS[1], ndata = a)
        @test l(g2, g2.ndata.x, conv_weight = w) == w * a
    end
end

@testset "ChebConv" begin
    k = 2
    l = ChebConv(D_IN => D_OUT, k)
    @test size(l.weight) == (D_OUT, D_IN, k)
    @test size(l.bias) == (D_OUT,)
    @test l.k == k
    for g in test_graphs
        g = add_self_loops(g)
        test_layer(l, g, rtol = RTOL_HIGH, test_gpu = TEST_GPU,
                    outsize = (D_OUT, g.num_nodes))
    end

    @testset "bias=false" begin
        @test length(Flux.params(ChebConv(2 => 3, 3))) == 2
        @test length(Flux.params(ChebConv(2 => 3, 3, bias = false))) == 1
    end
end

@testset "GraphConv" begin
    l = GraphConv(D_IN => D_OUT)
    for g in test_graphs
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (D_OUT, g.num_nodes))
    end

    l = GraphConv(D_IN => D_OUT, tanh, bias = false, aggr = mean)
    for g in test_graphs
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (D_OUT, g.num_nodes))
    end

    @testset "bias=false" begin
        @test length(Flux.params(GraphConv(2 => 3))) == 3
        @test length(Flux.params(GraphConv(2 => 3, bias = false))) == 2
    end
end

@testset "GATConv" begin
    for heads in (1, 2), concat in (true, false)
        l = GATConv(D_IN => D_OUT; heads, concat, dropout=0)
        for g in test_graphs
            test_layer(l, g, rtol = RTOL_LOW,
                        exclude_grad_fields = [:negative_slope, :dropout],
                        outsize = (concat ? heads * D_OUT : D_OUT,
                                    g.num_nodes))
        end
    end

    @testset "edge features" begin
        ein = 3
        l = GATConv((D_IN, ein) => D_OUT, add_self_loops = false, dropout=0)
        g = GNNGraph(g1, edata = rand(Float32, ein, g1.num_edges))
        test_layer(l, g, rtol = RTOL_LOW,
                    exclude_grad_fields = [:negative_slope, :dropout],
                    outsize = (D_OUT, g.num_nodes))
    end

    @testset "num params" begin
        l = GATConv(2 => 3, add_self_loops = false)
        @test length(Flux.params(l)) == 3
        l = GATConv((2, 4) => 3, add_self_loops = false)
        @test length(Flux.params(l)) == 4
        l = GATConv((2, 4) => 3, add_self_loops = false, bias = false)
        @test length(Flux.params(l)) == 3
    end
end

@testset "GATv2Conv" begin
    for heads in (1, 2), concat in (true, false)
        l = GATv2Conv(D_IN => D_OUT, tanh; heads, concat, dropout=0)
        for g in test_graphs
            test_layer(l, g, rtol = RTOL_LOW, atol=ATOL_LOW,
                        exclude_grad_fields = [:negative_slope, :dropout],
                        outsize = (concat ? heads * D_OUT : D_OUT,
                                    g.num_nodes))
        end
    end

    @testset "edge features" begin
        ein = 3
        l = GATv2Conv((D_IN, ein) => D_OUT, add_self_loops = false, dropout=0)
        g = GNNGraph(g1, edata = rand(Float32, ein, g1.num_edges))
        test_layer(l, g, rtol = RTOL_LOW, atol=ATOL_LOW,
                    exclude_grad_fields = [:negative_slope, :dropout],
                    outsize = (D_OUT, g.num_nodes))
    end

    @testset "num params" begin
        l = GATv2Conv(2 => 3, add_self_loops = false)
        @test length(Flux.params(l)) == 5
        l = GATv2Conv((2, 4) => 3, add_self_loops = false)
        @test length(Flux.params(l)) == 6
        l = GATv2Conv((2, 4) => 3, add_self_loops = false, bias = false)
        @test length(Flux.params(l)) == 4
    end
end

@testset "GatedGraphConv" begin
    num_layers = 3
    l = GatedGraphConv(D_OUT, num_layers)
    @test size(l.weight) == (D_OUT, D_OUT, num_layers)

    for g in test_graphs
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (D_OUT, g.num_nodes))
    end
end

@testset "EdgeConv" begin
    l = EdgeConv(Dense(2 * D_IN, D_OUT), aggr = +)
    for g in test_graphs
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (D_OUT, g.num_nodes))
    end
end

@testset "GINConv" begin
    nn = Dense(D_IN, D_OUT)

    l = GINConv(nn, 0.01f0, aggr = mean)
    for g in test_graphs
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (D_OUT, g.num_nodes))
    end

    @test !in(:eps, Flux.trainable(l))
end

@testset "NNConv" begin
    edim = 10
    nn = Dense(edim, D_OUT * D_IN)

    l = NNConv(D_IN => D_OUT, nn, tanh, bias = true, aggr = +)
    for g in test_graphs
        g = GNNGraph(g, edata = rand(Float32, edim, g.num_edges))
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (D_OUT, g.num_nodes))
    end
end

@testset "SAGEConv" begin
    l = SAGEConv(D_IN => D_OUT)
    @test l.aggr == mean

    l = SAGEConv(D_IN => D_OUT, tanh, bias = false, aggr = +)
    for g in test_graphs
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (D_OUT, g.num_nodes))
    end
end

@testset "ResGatedGraphConv" begin
    l = ResGatedGraphConv(D_IN => D_OUT, tanh, bias = true)
    for g in test_graphs
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (D_OUT, g.num_nodes))
    end
end

@testset "CGConv" begin
    edim = 10
    l = CGConv((D_IN, edim) => D_OUT, tanh, residual = false, bias = true)
    for g in test_graphs
        g = GNNGraph(g, edata = rand(Float32, edim, g.num_edges))
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (D_OUT, g.num_nodes))
    end

    # no edge features
    l1 = CGConv(D_IN => D_OUT, tanh, residual = false, bias = true)
    @test l1(g1, g1.ndata.x) == l1(g1).ndata.x
    @test l1(g1, g1.ndata.x, nothing) == l1(g1).ndata.x
end

@testset "AGNNConv" begin
    l = AGNNConv(trainable=false, add_self_loops=false)
    @test l.β == [1.0f0]
    @test l.add_self_loops == false
    @test l.trainable == false
    Flux.trainable(l) == (;)

    l = AGNNConv(init_beta=2.0f0)
    @test l.β == [2.0f0]
    @test l.add_self_loops == true
    @test l.trainable == true 
    Flux.trainable(l) == (; β = [1f0])
    for g in test_graphs
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (D_IN, g.num_nodes))
    end
end

@testset "MEGNetConv" begin
    l = MEGNetConv(D_IN => D_OUT, aggr = +)
    for g in test_graphs
        g = GNNGraph(g, edata = rand(Float32, D_IN, g.num_edges))
        test_layer(l, g, rtol = RTOL_LOW,
                    outtype = :node_edge,
                    outsize = ((D_OUT, g.num_nodes), (D_OUT, g.num_edges)))
    end
end

@testset "GMMConv" begin
    ein_channel = 10
    K = 5
    l = GMMConv((D_IN, ein_channel) => D_OUT, K = K)
    for g in test_graphs
        g = GNNGraph(g, edata = rand(Float32, ein_channel, g.num_edges))
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (D_OUT, g.num_nodes))
    end
end

@testset "SGConv" begin
    K = [1, 2, 3] # for different number of hops       
    for k in K
        l = SGConv(D_IN => D_OUT, k, add_self_loops = true)
        for g in test_graphs
            test_layer(l, g, rtol = RTOL_HIGH, outsize = (D_OUT, g.num_nodes))
        end

        l = SGConv(D_IN => D_OUT, k, add_self_loops = true)
        for g in test_graphs
            test_layer(l, g, rtol = RTOL_HIGH, outsize = (D_OUT, g.num_nodes))
        end
    end
end

@testset "TAGConv" begin
    K = [1, 2, 3]
    for k in K
        l = TAGConv(D_IN => D_OUT, k, add_self_loops = true)
        for g in test_graphs
            test_layer(l, g, rtol = RTOL_HIGH, outsize = (D_OUT, g.num_nodes))
        end

        l = TAGConv(D_IN => D_OUT, k, add_self_loops = true)
        for g in test_graphs
            test_layer(l, g, rtol = RTOL_HIGH, outsize = (D_OUT, g.num_nodes))
        end
    end
end

@testset "EGNNConv" begin
    hin = 5
    hout = 5
    hidden = 5
    l = EGNNConv(hin => hout, hidden)
    g = rand_graph(10, 20, graph_type = GRAPH_T)
    x = rand(Float32, D_IN, g.num_nodes)
    h = randn(Float32, hin, g.num_nodes)
    hnew, xnew = l(g, h, x)
    @test size(hnew) == (hout, g.num_nodes)
    @test size(xnew) == (D_IN, g.num_nodes)
end

@testset "TransformerConv" begin
    ein = 2
    heads = 3
    # used like in Kool et al., 2019
    l = TransformerConv(D_IN * heads => D_IN; heads, add_self_loops = true,
                        root_weight = false, ff_channels = 10, skip_connection = true,
                        batch_norm = false)
    # batch_norm=false here for tests to pass; true in paper
    for g in TEST_GRAPHS
        g = GNNGraph(g, ndata = rand(Float32, D_IN * heads, g.num_nodes), graph_type = GRAPH_T)
        test_layer(l, g, rtol = RTOL_LOW,
                    exclude_grad_fields = [:negative_slope],
                    outsize = (D_IN * heads, g.num_nodes))
    end
    # used like in Shi et al., 2021 
    l = TransformerConv((D_IN, ein) => D_IN; heads, gating = true,
                        bias_qkv = true)
    for g in test_graphs
        g = GNNGraph(g, edata = rand(Float32, ein, g.num_edges))
        test_layer(l, g, rtol = RTOL_LOW,
                    exclude_grad_fields = [:negative_slope],
                    outsize = (D_IN * heads, g.num_nodes))
    end
    # test averaging heads
    l = TransformerConv(D_IN => D_IN; heads, concat = false,
                        bias_root = false,
                        root_weight = false)
    for g in test_graphs
        test_layer(l, g, rtol = RTOL_LOW,
                    exclude_grad_fields = [:negative_slope],
                    outsize = (D_IN, g.num_nodes))
    end
end

@testset "DConv" begin
    K = [1, 2, 3] # for different number of hops       
    for k in K
        l = DConv(D_IN => D_OUT, k)
        for g in test_graphs
            test_layer(l, g, rtol = RTOL_HIGH, outsize = (D_OUT, g.num_nodes))
        end
    end
end