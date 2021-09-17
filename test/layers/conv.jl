@testset "conv" begin
    in_channel = 3
    out_channel = 5
    N = 4
    T = Float32
    adj =  [0 1 0 1
            1 0 1 0
            0 1 0 1
            1 0 1 0]
    
    g = GNNGraph(adj)
        
    adj_single_vertex =  [0 0 0 1
                          0 0 0 0
                          0 0 0 1
                          1 0 1 0]
    
    g_single_vertex = GNNGraph(adj_single_vertex, graph_type=GRAPH_T)    

    @testset "GCNConv" begin
        X = rand(T, in_channel, N)
        Xt = transpose(rand(T, N, in_channel))
    
        l = GCNConv(in_channel=>out_channel)
        @test size(l.weight) == (out_channel, in_channel)
        @test size(l.bias) == (out_channel,)
        
        g = GNNGraph(adj, ndata=X, graph_type=GRAPH_T)
        g_ = l(g)
        @test node_features(g_) isa Matrix{T}
        @test size(node_features(g_)) == (out_channel, N)
        @test_throws MethodError l(X)
        
        # Test with transposed features
        gt = GNNGraph(adj, ndata=Xt, graph_type=GRAPH_T)
        gt_ = l(gt)
        @test node_features(gt_) isa Matrix{T}
        @test size(node_features(gt_)) == (out_channel, N)

        gs = Zygote.gradient(x -> sum(node_features(l(x))), g)[1]
        @test size(gs.ndata.x) == size(X)

        gs = Zygote.gradient(model -> sum(node_features(model(g))), l)[1]
        @test size(gs.weight) == size(l.weight)
        @test size(gs.bias) == size(l.bias)
        
        @testset "bias=false" begin
            @test length(Flux.params(GCNConv(2=>3))) == 2
            @test length(Flux.params(GCNConv(2=>3, bias=false))) == 1
        end
    end


    @testset "ChebConv" begin
        k = 6
        X = rand(T, in_channel, N)
        Xt = transpose(rand(T, N, in_channel))
       
        l = ChebConv(in_channel=>out_channel, k)
        @test size(l.weight) == (out_channel, in_channel, k)
        @test size(l.bias) == (out_channel,)
        @test l.k == k
        
        g = GNNGraph(adj, ndata=X, graph_type=GRAPH_T)
        g_ = l(g)
        @test node_features(g_) isa Matrix{T}
        @test size(node_features(g_)) == (out_channel, N)
        @test_throws MethodError l(X)

        # Test with transposed features
        gt = GNNGraph(adj, ndata=Xt, graph_type=GRAPH_T)
        gt_ = l(gt)
        @test node_features(g_) isa Matrix{T}
        @test size(node_features(gt_)) == (out_channel, N)

        gs = Zygote.gradient(x -> sum(node_features(l(x))), g)[1]
        @test size(gs.ndata.x) == size(X)

        gs = Zygote.gradient(model -> sum(node_features(model(g))), l)[1]
        @test size(gs.weight) == size(l.weight)
        @test size(gs.bias) == size(l.bias)

        @testset "bias=false" begin
            @test length(Flux.params(ChebConv(2=>3, 3))) == 2
            @test length(Flux.params(ChebConv(2=>3, 3, bias=false))) == 1
        end
    end

    @testset "GraphConv" begin
        X = rand(T, in_channel, N)
        Xt = transpose(rand(T, N, in_channel))
        

        l = GraphConv(in_channel=>out_channel)
        @test size(l.weight1) == (out_channel, in_channel)
        @test size(l.weight2) == (out_channel, in_channel)
        @test size(l.bias) == (out_channel,)

        g = GNNGraph(adj, ndata=X, graph_type=GRAPH_T)
        g_ = l(g)
        @test node_features(g_) isa Matrix{T}
        @test size(node_features(g_)) == (out_channel, N)
        @test_throws MethodError l(X)

        # Test with transposed features
        gt = GNNGraph(adj, ndata=Xt, graph_type=GRAPH_T)
        gt_ = l(gt)
        @test node_features(gt_) isa Matrix{T}
        @test size(node_features(gt_)) == (out_channel, N)

        gs = Zygote.gradient(g -> sum(node_features(l(g))), g)[1]
        @test size(gs.ndata.x) == size(X)

        gs = Zygote.gradient(model -> sum(node_features(model(g))), l)[1]
        @test size(gs.weight1) == size(l.weight1)
        @test size(gs.weight2) == size(l.weight2)
        @test size(gs.bias) == size(l.bias)

        @testset "bias=false" begin
            @test length(Flux.params(GraphConv(2=>3))) == 3
            @test length(Flux.params(GraphConv(2=>3, bias=false))) == 2
        end
    end

    @testset "GATConv" begin

        X = rand(T, in_channel, N)
        Xt = transpose(rand(T, N, in_channel))

        for heads = [1, 2], concat = [true, false], adj_gat in [adj, adj_single_vertex]
            g_gat = GNNGraph(adj_gat, ndata=X, graph_type=GRAPH_T)
            gat = GATConv(in_channel=>out_channel, heads=heads, concat=concat)
            @test size(gat.weight) == (out_channel * heads, in_channel)
            @test size(gat.bias) == (out_channel * heads,)
            @test size(gat.a) == (2*out_channel, heads)
            @test length(Flux.trainable(gat)) == 3

            g_ = gat(g_gat)
            Y = node_features(g_)
            @test Y isa Matrix{T}
            @test size(Y) == (concat ? (out_channel*heads, N) : (out_channel, N))
            @test_throws MethodError gat(X)

            # Test with transposed features
            gt = GNNGraph(adj_gat, ndata=Xt, graph_type=GRAPH_T)
            gt_ = gat(gt)
            @test node_features(g_) isa Matrix{T}
            @test size(node_features(gt_)) == (concat ? (out_channel*heads, N) : (out_channel, N))

            gs = Zygote.gradient(g -> sum(node_features(gat(g))), g_gat)[1]
            @test size(gs.ndata.x) == size(X)

            gs = Zygote.gradient(model -> sum(node_features(model(g_gat))), gat)[1]
            @test size(gs.weight) == size(gat.weight)
            @test size(gs.bias) == size(gat.bias)
            @test size(gs.a) == size(gat.a)
        end

        @testset "bias=false" begin
            @test length(Flux.params(GATConv(2=>3))) == 3
            @test length(Flux.params(GATConv(2=>3, bias=false))) == 2
        end
    end

    @testset "GatedGraphConv" begin
        num_layers = 3
        X = rand(T, in_channel, N)
        Xt = transpose(rand(T, N, in_channel))
        ggc = GatedGraphConv(out_channel, num_layers)
        @test size(ggc.weight) == (out_channel, out_channel, num_layers)

        g = GNNGraph(adj, ndata=X, graph_type=GRAPH_T)
        g_ = ggc(g)
        @test node_features(g_) isa Matrix{T}
        @test size(node_features(g_)) == (out_channel, N)
        @test_throws MethodError ggc(X)

        # Test with transposed features
        gt = GNNGraph(adj, ndata=Xt, graph_type=GRAPH_T)
        gt_ = ggc(gt)
        @test node_features(gt_) isa Matrix{T}
        @test size(node_features(gt_)) == (out_channel, N)

        gs = Zygote.gradient(x -> sum(node_features(ggc(x))), g)[1]
        @test size(gs.ndata.x) == size(X)

        gs = Zygote.gradient(model -> sum(node_features(model(g))), ggc)[1]
        @test size(gs.weight) == size(ggc.weight)
    end

    @testset "EdgeConv" begin
        X = rand(T, in_channel, N)
        Xt = transpose(rand(T, N, in_channel))
        ec = EdgeConv(Dense(2*in_channel, out_channel))

        g = GNNGraph(adj, ndata=X, graph_type=GRAPH_T)
        g_ = ec(g)
        @test node_features(g_) isa Matrix{T} 
        @test size(node_features(g_)) == (out_channel, N)
        @test_throws MethodError ec(X)

        # Test with transposed features
        gt = GNNGraph(adj, ndata=Xt, graph_type=GRAPH_T)
        gt_ = ec(gt)
        @test node_features(gt_) isa Matrix{T}
        @test size(node_features(gt_)) == (out_channel, N)

        gs = Zygote.gradient(x -> sum(node_features(ec(x))), g)[1]
        @test size(gs.ndata.x) == size(X)

        gs = Zygote.gradient(model -> sum(node_features(model(g))), ec)[1]
        @test size(gs.nn.weight) == size(ec.nn.weight)
        @test size(gs.nn.bias) == size(ec.nn.bias)
    end

    @testset "GINConv" begin
        X = rand(T, in_channel, N)
        nn = Dense(in_channel, out_channel)
        eps = 0.001f0
        g = GNNGraph(adj, ndata=X) 
        
        l = GINConv(nn, eps=eps)
        @test l.nn === nn
        
        g_ = l(g)
        @test size(node_features(g_)) == (out_channel, N)

        gs = Zygote.gradient(g -> sum(node_features(l(g))), g)[1]
        @test size(gs.ndata.x) == size(X)

        gs = Zygote.gradient(model -> sum(node_features(model(g))), l)[1]
        @test size(gs.nn.weight) == size(l.nn.weight)
        @test size(gs.nn.bias) == size(l.nn.bias)
        
        @test !in(:eps, Flux.trainable(l))
    end
end
