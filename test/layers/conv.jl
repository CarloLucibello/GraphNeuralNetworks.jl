@testset "Conv Layers" begin
    in_channel = 3
    out_channel = 5
    N = 4
    T = Float32

    adj1 =  [0 1 0 1
            1 0 1 0
            0 1 0 1
            1 0 1 0]
    
    g1 = GNNGraph(adj1, 
            ndata=rand(T, in_channel, N), 
            graph_type=GRAPH_T)
        
    adj_single_vertex =  [0 0 0 1
                          0 0 0 0
                          0 0 0 1
                          1 0 1 0]
    
    g_single_vertex = GNNGraph(adj_single_vertex, 
                                ndata=rand(T, in_channel, N), 
                                graph_type=GRAPH_T)    

    test_graphs = [g1, g_single_vertex]

    @testset "GCNConv" begin
        l = GCNConv(in_channel => out_channel)
        for g in test_graphs
            test_layer(l, g, rtol=1e-5, outsize=(out_channel, g.num_nodes))
        end

        l = GCNConv(in_channel => out_channel, tanh, bias=false)
        for g in test_graphs
            test_layer(l, g, rtol=1e-5, outsize=(out_channel, g.num_nodes))
        end

        l = GCNConv(in_channel => out_channel, add_self_loops=false)
        test_layer(l, g1, rtol=1e-5, outsize=(out_channel, g1.num_nodes))
    end

    @testset "ChebConv" begin
        k = 3
        l = ChebConv(in_channel => out_channel, k)
        @test size(l.weight) == (out_channel, in_channel, k)
        @test size(l.bias) == (out_channel,)
        @test l.k == k
        for g in test_graphs
            g = add_self_loops(g)
            test_layer(l, g, rtol=1e-5, test_gpu=false, outsize=(out_channel, g.num_nodes))
            if TEST_GPU
                @test_broken test_layer(l, g, rtol=1e-5, test_gpu=true, outsize=(out_channel, g.num_nodes))
            end              
        end
        
        @testset "bias=false" begin
            @test length(Flux.params(ChebConv(2=>3, 3))) == 2
            @test length(Flux.params(ChebConv(2=>3, 3, bias=false))) == 1
        end
    end

    @testset "GraphConv" begin
        l = GraphConv(in_channel => out_channel)
        for g in test_graphs
            test_layer(l, g, rtol=1e-5, outsize=(out_channel, g.num_nodes))
        end

        l = GraphConv(in_channel => out_channel, relu, bias=false, aggr=mean)
        for g in test_graphs
            test_layer(l, g, rtol=1e-5, outsize=(out_channel, g.num_nodes))
        end
        
        @testset "bias=false" begin
            @test length(Flux.params(GraphConv(2=>3))) == 3
            @test length(Flux.params(GraphConv(2=>3, bias=false))) == 2
        end
    end

    @testset "GATConv" begin

        for heads in (1, 2), concat in (true, false)
            l = GATConv(in_channel => out_channel; heads, concat)
            for g in test_graphs
                test_layer(l, g, rtol=1e-4,
                    outsize=(concat ? heads*out_channel : out_channel, g.num_nodes))
            end
        end

        @testset "bias=false" begin
            @test length(Flux.params(GATConv(2=>3))) == 3
            @test length(Flux.params(GATConv(2=>3, bias=false))) == 2
        end
    end

    @testset "GatedGraphConv" begin
        num_layers = 3
        l = GatedGraphConv(out_channel, num_layers)
        @test size(l.weight) == (out_channel, out_channel, num_layers)

        for g in test_graphs
            test_layer(l, g, rtol=1e-5, outsize=(out_channel, g.num_nodes)) 
        end
    end

    @testset "EdgeConv" begin
        l = EdgeConv(Dense(2*in_channel, out_channel), aggr=+)
        for g in test_graphs
            test_layer(l, g, rtol=1e-5, outsize=(out_channel, g.num_nodes))
        end
    end

    @testset "GINConv" begin
        nn = Dense(in_channel, out_channel)
        
        l = GINConv(nn, 0.01f0, aggr=mean)
        for g in test_graphs
            test_layer(l, g, rtol=1e-5, outsize=(out_channel, g.num_nodes)) 
        end
    
        @test !in(:eps, Flux.trainable(l))
    end

    @testset "NNConv" begin
        edim = 10
        nn = Dense(edim, out_channel * in_channel)
        
        l = NNConv(in_channel => out_channel, nn, tanh, bias=true, aggr=+)
        for g in test_graphs
            g = GNNGraph(g, edata=rand(T, edim, g.num_edges))
            test_layer(l, g, rtol=1e-5, outsize=(out_channel, g.num_nodes)) 
        end
    end

    @testset "SAGEConv" begin
        l = SAGEConv(in_channel => out_channel)
        @test l.aggr == mean

        l = SAGEConv(in_channel => out_channel, tanh, bias=false, aggr=+)
        for g in test_graphs
            test_layer(l, g, rtol=1e-5, outsize=(out_channel, g.num_nodes)) 
        end
    end


    @testset "ResGatedGraphConv" begin
        l = ResGatedGraphConv(in_channel => out_channel, tanh, bias=true)
        for g in test_graphs
            test_layer(l, g, rtol=1e-5, outsize=(out_channel, g.num_nodes))
        end
    end


    @testset "CGConv" begin
        edim = 10
        l = CGConv((in_channel, edim) => out_channel, tanh, residual=false, bias=true)
        for g in test_graphs
            g = GNNGraph(g, edata=rand(T, edim, g.num_edges))
            test_layer(l, g, rtol=1e-5, outsize=(out_channel, g.num_nodes)) 
        end
    end


    @testset "AGNNConv" begin
        l = AGNNConv()
        l.β == [1f0]
        for g in test_graphs
            test_layer(l, g, rtol=1e-5, outsize=(in_channel, g.num_nodes)) 
        end
    end
end
