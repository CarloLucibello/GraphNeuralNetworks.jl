RTOL_LOW = 1e-2
RTOL_HIGH = 1e-5
ATOL_LOW = 1e-3

in_channel = 3
out_channel = 5
N = 4
T = Float32

adj1 = [0 1 0 1
        1 0 1 0
        0 1 0 1
        1 0 1 0]

g1 = GNNGraph(adj1,
                ndata = rand(T, in_channel, N),
                graph_type = GRAPH_T)

adj_single_vertex = [0 0 0 1
                        0 0 0 0
                        0 0 0 1
                        1 0 1 0]

g_single_vertex = GNNGraph(adj_single_vertex,
                            ndata = rand(T, in_channel, N),
                            graph_type = GRAPH_T)

test_graphs = [g1, g_single_vertex]

@testset "GCNConv" begin
    l = GCNConv(in_channel => out_channel)
    for g in test_graphs
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (out_channel, g.num_nodes))
    end

    l = GCNConv(in_channel => out_channel, tanh, bias = false)
    for g in test_graphs
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (out_channel, g.num_nodes))
    end

    l = GCNConv(in_channel => out_channel, add_self_loops = false)
    test_layer(l, g1, rtol = RTOL_HIGH, outsize = (out_channel, g1.num_nodes))

    @testset "edge weights & custom normalization" begin
        s = [2, 3, 1, 3, 1, 2]
        t = [1, 1, 2, 2, 3, 3]
        w = T[1, 2, 3, 4, 5, 6]
        g = GNNGraph((s, t, w), ndata = ones(T, 1, 3), graph_type = GRAPH_T)
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
        w = rand(T, 6)
        x = rand(T, 1, 3)
        g = GNNGraph((s, t, w), ndata = x, graph_type = GRAPH_T, edata = w)
        l = GCNConv(1 => 1, add_self_loops = false, use_edge_weight = true)
        @test gradient(w -> sum(l(g, x, w)), w)[1] isa AbstractVector{T}   # redundant test but more explicit
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (1, g.num_nodes), test_gpu = false)
    end

    @testset "conv_weight" begin
         l = GraphNeuralNetworks.GCNConv(in_channel => out_channel)
        w = zeros(T, out_channel, in_channel)
        g1 = GNNGraph(adj1, ndata = ones(T, in_channel, N))
        @test l(g1, g1.ndata.x, conv_weight = w) == zeros(T, out_channel, N)
        a = rand(T, in_channel, N)
        g2 = GNNGraph(adj1, ndata = a)
        @test l(g2, g2.ndata.x, conv_weight = w) == w * a
    end
end

@testset "ChebConv" begin
    k = 2
    l = ChebConv(in_channel => out_channel, k)
    @test size(l.weight) == (out_channel, in_channel, k)
    @test size(l.bias) == (out_channel,)
    @test l.k == k
    for g in test_graphs
        g = add_self_loops(g)
        test_layer(l, g, rtol = RTOL_HIGH, test_gpu = TEST_GPU,
                    outsize = (out_channel, g.num_nodes))
    end

    @testset "bias=false" begin
        @test length(Flux.params(ChebConv(2 => 3, 3))) == 2
        @test length(Flux.params(ChebConv(2 => 3, 3, bias = false))) == 1
    end
end

@testset "GraphConv" begin
    l = GraphConv(in_channel => out_channel)
    for g in test_graphs
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (out_channel, g.num_nodes))
    end

    l = GraphConv(in_channel => out_channel, tanh, bias = false, aggr = mean)
    for g in test_graphs
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (out_channel, g.num_nodes))
    end

    @testset "bias=false" begin
        @test length(Flux.params(GraphConv(2 => 3))) == 3
        @test length(Flux.params(GraphConv(2 => 3, bias = false))) == 2
    end
end

@testset "GATConv" begin
    for heads in (1, 2), concat in (true, false)
        l = GATConv(in_channel => out_channel; heads, concat, dropout=0)
        for g in test_graphs
            test_layer(l, g, rtol = RTOL_LOW,
                        exclude_grad_fields = [:negative_slope, :dropout],
                        outsize = (concat ? heads * out_channel : out_channel,
                                    g.num_nodes))
        end
    end

    @testset "edge features" begin
        ein = 3
        l = GATConv((in_channel, ein) => out_channel, add_self_loops = false, dropout=0)
        g = GNNGraph(g1, edata = rand(T, ein, g1.num_edges))
        test_layer(l, g, rtol = RTOL_LOW,
                    exclude_grad_fields = [:negative_slope, :dropout],
                    outsize = (out_channel, g.num_nodes))
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
        l = GATv2Conv(in_channel => out_channel, tanh; heads, concat, dropout=0)
        for g in test_graphs
            test_layer(l, g, rtol = RTOL_LOW, atol=ATOL_LOW,
                        exclude_grad_fields = [:negative_slope, :dropout],
                        outsize = (concat ? heads * out_channel : out_channel,
                                    g.num_nodes))
        end
    end

    @testset "edge features" begin
        ein = 3
        l = GATv2Conv((in_channel, ein) => out_channel, add_self_loops = false, dropout=0)
        g = GNNGraph(g1, edata = rand(T, ein, g1.num_edges))
        test_layer(l, g, rtol = RTOL_LOW, atol=ATOL_LOW,
                    exclude_grad_fields = [:negative_slope, :dropout],
                    outsize = (out_channel, g.num_nodes))
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
    l = GatedGraphConv(out_channel, num_layers)
    @test size(l.weight) == (out_channel, out_channel, num_layers)

    for g in test_graphs
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (out_channel, g.num_nodes))
    end
end

@testset "EdgeConv" begin
    l = EdgeConv(Dense(2 * in_channel, out_channel), aggr = +)
    for g in test_graphs
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (out_channel, g.num_nodes))
    end
end

@testset "GINConv" begin
    nn = Dense(in_channel, out_channel)

    l = GINConv(nn, 0.01f0, aggr = mean)
    for g in test_graphs
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (out_channel, g.num_nodes))
    end

    @test !in(:eps, Flux.trainable(l))
end

@testset "NNConv" begin
    edim = 10
    nn = Dense(edim, out_channel * in_channel)

    l = NNConv(in_channel => out_channel, nn, tanh, bias = true, aggr = +)
    for g in test_graphs
        g = GNNGraph(g, edata = rand(T, edim, g.num_edges))
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (out_channel, g.num_nodes))
    end
end

@testset "SAGEConv" begin
    l = SAGEConv(in_channel => out_channel)
    @test l.aggr == mean

    l = SAGEConv(in_channel => out_channel, tanh, bias = false, aggr = +)
    for g in test_graphs
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (out_channel, g.num_nodes))
    end
end

@testset "ResGatedGraphConv" begin
    l = ResGatedGraphConv(in_channel => out_channel, tanh, bias = true)
    for g in test_graphs
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (out_channel, g.num_nodes))
    end
end

@testset "CGConv" begin
    edim = 10
    l = CGConv((in_channel, edim) => out_channel, tanh, residual = false, bias = true)
    for g in test_graphs
        g = GNNGraph(g, edata = rand(T, edim, g.num_edges))
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (out_channel, g.num_nodes))
    end

    # no edge features
    l1 = CGConv(in_channel => out_channel, tanh, residual = false, bias = true)
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
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (in_channel, g.num_nodes))
    end
end

@testset "MEGNetConv" begin
    l = MEGNetConv(in_channel => out_channel, aggr = +)
    for g in test_graphs
        g = GNNGraph(g, edata = rand(T, in_channel, g.num_edges))
        test_layer(l, g, rtol = RTOL_LOW,
                    outtype = :node_edge,
                    outsize = ((out_channel, g.num_nodes), (out_channel, g.num_edges)))
    end
end

@testset "GMMConv" begin
    ein_channel = 10
    K = 5
    l = GMMConv((in_channel, ein_channel) => out_channel, K = K)
    for g in test_graphs
        g = GNNGraph(g, edata = rand(Float32, ein_channel, g.num_edges))
        test_layer(l, g, rtol = RTOL_HIGH, outsize = (out_channel, g.num_nodes))
    end
end

@testset "SGConv" begin
    K = [1, 2, 3] # for different number of hops       
    for k in K
        l = SGConv(in_channel => out_channel, k, add_self_loops = true)
        for g in test_graphs
            test_layer(l, g, rtol = RTOL_HIGH, outsize = (out_channel, g.num_nodes))
        end

        l = SGConv(in_channel => out_channel, k, add_self_loops = true)
        for g in test_graphs
            test_layer(l, g, rtol = RTOL_HIGH, outsize = (out_channel, g.num_nodes))
        end
    end
end

@testset "TAGConv" begin
    K = [1, 2, 3]
    for k in K
        l = TAGConv(in_channel => out_channel, k, add_self_loops = true)
        for g in test_graphs
            test_layer(l, g, rtol = RTOL_HIGH, outsize = (out_channel, g.num_nodes))
        end

        l = TAGConv(in_channel => out_channel, k, add_self_loops = true)
        for g in test_graphs
            test_layer(l, g, rtol = RTOL_HIGH, outsize = (out_channel, g.num_nodes))
        end
    end
end

@testset "EGNNConv" begin
    hin = 5
    hout = 5
    hidden = 5
    l = EGNNConv(hin => hout, hidden)
    g = rand_graph(10, 20, graph_type = GRAPH_T)
    x = rand(T, in_channel, g.num_nodes)
    h = randn(T, hin, g.num_nodes)
    hnew, xnew = l(g, h, x)
    @test size(hnew) == (hout, g.num_nodes)
    @test size(xnew) == (in_channel, g.num_nodes)
end

@testset "TransformerConv" begin
    ein = 2
    heads = 3
    # used like in Kool et al., 2019
    l = TransformerConv(in_channel * heads => in_channel; heads, add_self_loops = true,
                        root_weight = false, ff_channels = 10, skip_connection = true,
                        batch_norm = false)
    # batch_norm=false here for tests to pass; true in paper
    for adj in [adj1, adj_single_vertex]
        g = GNNGraph(adj, ndata = rand(T, in_channel * heads, size(adj, 1)),
                        graph_type = GRAPH_T)
        test_layer(l, g, rtol = RTOL_LOW,
                    exclude_grad_fields = [:negative_slope],
                    outsize = (in_channel * heads, g.num_nodes))
    end
    # used like in Shi et al., 2021 
    l = TransformerConv((in_channel, ein) => in_channel; heads, gating = true,
                        bias_qkv = true)
    for g in test_graphs
        g = GNNGraph(g, edata = rand(T, ein, g.num_edges))
        test_layer(l, g, rtol = RTOL_LOW,
                    exclude_grad_fields = [:negative_slope],
                    outsize = (in_channel * heads, g.num_nodes))
    end
    # test averaging heads
    l = TransformerConv(in_channel => in_channel; heads, concat = false,
                        bias_root = false,
                        root_weight = false)
    for g in test_graphs
        test_layer(l, g, rtol = RTOL_LOW,
                    exclude_grad_fields = [:negative_slope],
                    outsize = (in_channel, g.num_nodes))
    end
end

@testset "DConv" begin
    K = [1, 2, 3] # for different number of hops       
    for k in K
        l = DConv(in_channel => out_channel, k)
        for g in test_graphs
            test_layer(l, g, rtol = RTOL_HIGH, outsize = (out_channel, g.num_nodes))
        end
    end
end