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
    
        gc = GCNConv(in_channel=>out_channel)
        @test size(gc.weight) == (out_channel, in_channel)
        @test size(gc.bias) == (out_channel,)
        
        g = GNNGraph(adj, nf=X, graph_type=GRAPH_T)
        g_ = gc(g)
        @test node_feature(g_) isa Matrix{T}
        @test size(node_feature(g_)) == (out_channel, N)
        @test_throws MethodError gc(X)
        
        # Test with transposed features
        gt = GNNGraph(adj, nf=Xt, graph_type=GRAPH_T)
        gt_ = gc(gt)
        @test node_feature(gt_) isa Matrix{T}
        @test size(node_feature(gt_)) == (out_channel, N)

        gs = Zygote.gradient(x -> sum(node_feature(gc(x))), g)[1]
        @test size(gs.nf) == size(X)

        gs = Zygote.gradient(model -> sum(node_feature(model(g))), gc)[1]
        @test size(gs.weight) == size(gc.weight)
        @test size(gs.bias) == size(gc.bias)
        
        @testset "bias=false" begin
            @test length(Flux.params(GCNConv(2=>3))) == 2
            @test length(Flux.params(GCNConv(2=>3, bias=false))) == 1
        end
    end


    @testset "ChebConv" begin
        k = 6
        X = rand(T, in_channel, N)
        Xt = transpose(rand(T, N, in_channel))
       
        cc = ChebConv(in_channel=>out_channel, k)
        @test size(cc.weight) == (out_channel, in_channel, k)
        @test size(cc.bias) == (out_channel,)
        @test cc.k == k
        
        g = GNNGraph(adj, nf=X, graph_type=GRAPH_T)
        g_ = cc(g)
        @test node_feature(g_) isa Matrix{T}
        @test size(node_feature(g_)) == (out_channel, N)
        @test_throws MethodError cc(X)

        # Test with transposed features
        gt = GNNGraph(adj, nf=Xt, graph_type=GRAPH_T)
        gt_ = cc(gt)
        @test node_feature(g_) isa Matrix{T}
        @test size(node_feature(gt_)) == (out_channel, N)

        gs = Zygote.gradient(x -> sum(node_feature(cc(x))), g)[1]
        @test size(gs.nf) == size(X)

        gs = Zygote.gradient(model -> sum(node_feature(model(g))), cc)[1]
        @test size(gs.weight) == size(cc.weight)
        @test size(gs.bias) == size(cc.bias)

        @testset "bias=false" begin
            @test length(Flux.params(ChebConv(2=>3, 3))) == 2
            @test length(Flux.params(ChebConv(2=>3, 3, bias=false))) == 1
        end
    end

    @testset "GraphConv" begin
        X = rand(T, in_channel, N)
        Xt = transpose(rand(T, N, in_channel))
        

        gc = GraphConv(in_channel=>out_channel)
        @test size(gc.weight1) == (out_channel, in_channel)
        @test size(gc.weight2) == (out_channel, in_channel)
        @test size(gc.bias) == (out_channel,)

        g = GNNGraph(adj, nf=X, graph_type=GRAPH_T)
        g_ = gc(g)
        @test node_feature(g_) isa Matrix{T}
        @test size(node_feature(g_)) == (out_channel, N)
        @test_throws MethodError gc(X)

        # Test with transposed features
        gt = GNNGraph(adj, nf=Xt, graph_type=GRAPH_T)
        gt_ = gc(gt)
        @test node_feature(gt_) isa Matrix{T}
        @test size(node_feature(gt_)) == (out_channel, N)

        gs = Zygote.gradient(x -> sum(node_feature(gc(x))), g)[1]
        @test size(gs.nf) == size(X)

        gs = Zygote.gradient(model -> sum(node_feature(model(g))), gc)[1]
        @test size(gs.weight1) == size(gc.weight1)
        @test size(gs.weight2) == size(gc.weight2)
        @test size(gs.bias) == size(gc.bias)

        @testset "bias=false" begin
            @test length(Flux.params(GraphConv(2=>3))) == 3
            @test length(Flux.params(GraphConv(2=>3, bias=false))) == 2
        end
    end

    @testset "GATConv" begin

        X = rand(T, in_channel, N)
        Xt = transpose(rand(T, N, in_channel))

        for heads = [1, 2], concat = [true, false], adj_gat in [adj, adj_single_vertex]
            g_gat = GNNGraph(adj_gat, nf=X, graph_type=GRAPH_T)
            gat = GATConv(in_channel=>out_channel, heads=heads, concat=concat)
            @test size(gat.weight) == (out_channel * heads, in_channel)
            @test size(gat.bias) == (out_channel * heads,)
            @test size(gat.a) == (2*out_channel, heads)

            g_ = gat(g_gat)
            Y = node_feature(g_)
            @test Y isa Matrix{T}
            @test size(Y) == (concat ? (out_channel*heads, N) : (out_channel, N))
            @test_throws MethodError gat(X)

            # Test with transposed features
            gt = GNNGraph(adj_gat, nf=Xt, graph_type=GRAPH_T)
            gt_ = gat(gt)
            @test node_feature(g_) isa Matrix{T}
            @test size(node_feature(gt_)) == (concat ? (out_channel*heads, N) : (out_channel, N))

            gs = Zygote.gradient(x -> sum(node_feature(gat(x))), g_gat)[1]
            @test size(gs.nf) == size(X)

            gs = Zygote.gradient(model -> sum(node_feature(model(g_gat))), gat)[1]
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

        g = GNNGraph(adj, nf=X, graph_type=GRAPH_T)
        g_ = ggc(g)
        @test node_feature(g_) isa Matrix{T}
        @test size(node_feature(g_)) == (out_channel, N)
        @test_throws MethodError ggc(X)

        # Test with transposed features
        gt = GNNGraph(adj, nf=Xt, graph_type=GRAPH_T)
        gt_ = ggc(gt)
        @test node_feature(gt_) isa Matrix{T}
        @test size(node_feature(gt_)) == (out_channel, N)

        gs = Zygote.gradient(x -> sum(node_feature(ggc(x))), g)[1]
        @test size(gs.nf) == size(X)

        gs = Zygote.gradient(model -> sum(node_feature(model(g))), ggc)[1]
        @test size(gs.weight) == size(ggc.weight)
    end

    @testset "EdgeConv" begin
        X = rand(T, in_channel, N)
        Xt = transpose(rand(T, N, in_channel))
        ec = EdgeConv(Dense(2*in_channel, out_channel))

        g = GNNGraph(adj, nf=X, graph_type=GRAPH_T)
        g_ = ec(g)
        @test node_feature(g_) isa Matrix{T} 
        @test size(node_feature(g_)) == (out_channel, N)
        @test_throws MethodError ec(X)

        # Test with transposed features
        gt = GNNGraph(adj, nf=Xt, graph_type=GRAPH_T)
        gt_ = ec(gt)
        @test node_feature(gt_) isa Matrix{T}
        @test size(node_feature(gt_)) == (out_channel, N)

        gs = Zygote.gradient(x -> sum(node_feature(ec(x))), g)[1]
        @test size(gs.nf) == size(X)

        gs = Zygote.gradient(model -> sum(node_feature(model(g))), ec)[1]
        @test size(gs.nn.weight) == size(ec.nn.weight)
        @test size(gs.nn.bias) == size(ec.nn.bias)
    end

    @testset "GINConv" begin
        X = rand(T, in_channel, N)
        nn = Dense(in_channel, out_channel)
        eps = 0.001f0
        g = GNNGraph(adj, nf=X) 
        
        gc = GINConv(nn, eps=eps)
        @test gc.nn === nn
        
        g_ = gc(g)
        @test size(node_feature(g_)) == (out_channel, N)

        gs = Zygote.gradient(g -> sum(node_feature(gc(g))), g)[1]
        @test size(gs.nf) == size(X)

        gs = Zygote.gradient(model -> sum(node_feature(model(g))), gc)[1]
        @test size(gs.nn.weight) == size(gc.nn.weight)
        @test size(gs.nn.bias) == size(gc.nn.bias)
        
        @test !in(:eps, Flux.trainable(gc))
    end
end
