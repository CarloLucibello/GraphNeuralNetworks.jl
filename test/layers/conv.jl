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
            gradtest(l, g)
        end

        # l = GCNConv(in_channel => out_channel, relu, bias=false)
        # for g in test_graphs
        #     gradtest(l, g)
        # end
    end


    @testset "ChebConv" begin
        k = 6
        l = ChebConv(in_channel => out_channel, k)
        @test size(l.weight) == (out_channel, in_channel, k)
        @test size(l.bias) == (out_channel,)
        @test l.k == k
        for g in test_graphs
            gradtest(l, g, rtol=1) #TODO broken
        end
        
        @testset "bias=false" begin
            @test length(Flux.params(ChebConv(2=>3, 3))) == 2
            @test length(Flux.params(ChebConv(2=>3, 3, bias=false))) == 1
        end
    end

    @testset "GraphConv" begin
        l = GraphConv(in_channel => out_channel)
        for g in test_graphs
            gradtest(l, g, rtol=1e-3)
        end

        # l = GraphConv(in_channel => out_channel, relu, bias=false)
        # for g in test_graphs
        #     gradtest(l, g)
        # end
        
        @testset "bias=false" begin
            @test length(Flux.params(GraphConv(2=>3))) == 3
            @test length(Flux.params(GraphConv(2=>3, bias=false))) == 2
        end
    end

    @testset "GATConv" begin

        for heads = [1, 2], concat = [true, false]
            l = GATConv(in_channel => out_channel; heads, concat)
            for g in test_graphs
                gradtest(l, g, atol=1e-3, rtol=1e-2) #TODO 
            end
    
            # l = GraphConv(in_channel => out_channel, relu, bias=false)
            # for g in test_graphs
            #     gradtest(l, g)
            # end
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
            gradtest(l, g, atol=1e-5, rtol=1e-5) 
        end
    end

    @testset "EdgeConv" begin
        l = EdgeConv(Dense(2*in_channel, out_channel))
        for g in test_graphs
            gradtest(l, g, atol=1e-5, rtol=1e-5) 
        end
    end

    @testset "GINConv" begin
        nn = Dense(in_channel, out_channel)
        eps = 0.001f0
        l = GINConv(nn, eps=eps)
        for g in test_graphs
            gradtest(l, g, atol=1e-5, rtol=1e-5) 
        end
    
        @test !in(:eps, Flux.trainable(l))
    end
end
