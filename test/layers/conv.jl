@testset "conv" begin
    in_channel = 3
    out_channel = 5
    N = 4
    T = Float32
    adj =  [0 1 0 1
            1 0 1 0
            0 1 0 1
            1 0 1 0]
    
    fg = FeaturedGraph(adj)
        
    adj_single_vertex =  [0 0 0 1
                          0 0 0 0
                          0 0 0 1
                          1 0 1 0]
    
    fg_single_vertex = FeaturedGraph(adj_single_vertex, graph_type=GRAPH_T)    

    @testset "GCNConv" begin
        X = rand(T, in_channel, N)
        Xt = transpose(rand(T, N, in_channel))
    
        gc = GCNConv(in_channel=>out_channel)
        @test size(gc.weight) == (out_channel, in_channel)
        @test size(gc.bias) == (out_channel,)
        
        fg = FeaturedGraph(adj, nf=X, graph_type=GRAPH_T)
        fg_ = gc(fg)
        @test node_feature(fg_) isa Matrix{T}
        @test size(node_feature(fg_)) == (out_channel, N)
        @test_throws MethodError gc(X)
        
        # Test with transposed features
        fgt = FeaturedGraph(adj, nf=Xt, graph_type=GRAPH_T)
        fgt_ = gc(fgt)
        @test node_feature(fgt_) isa Matrix{T}
        @test size(node_feature(fgt_)) == (out_channel, N)

        g = Zygote.gradient(x -> sum(node_feature(gc(x))), fg)[1]
        @test size(g.nf) == size(X)

        g = Zygote.gradient(model -> sum(node_feature(model(fg))), gc)[1]
        @test size(g.weight) == size(gc.weight)
        @test size(g.bias) == size(gc.bias)
        
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
        
        fg = FeaturedGraph(adj, nf=X, graph_type=GRAPH_T)
        fg_ = cc(fg)
        @test node_feature(fg_) isa Matrix{T}
        @test size(node_feature(fg_)) == (out_channel, N)
        @test_throws MethodError cc(X)

        # Test with transposed features
        fgt = FeaturedGraph(adj, nf=Xt, graph_type=GRAPH_T)
        fgt_ = cc(fgt)
        @test node_feature(fg_) isa Matrix{T}
        @test size(node_feature(fgt_)) == (out_channel, N)

        g = Zygote.gradient(x -> sum(node_feature(cc(x))), fg)[1]
        @test size(g.nf) == size(X)

        g = Zygote.gradient(model -> sum(node_feature(model(fg))), cc)[1]
        @test size(g.weight) == size(cc.weight)
        @test size(g.bias) == size(cc.bias)

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

        fg = FeaturedGraph(adj, nf=X, graph_type=GRAPH_T)
        fg_ = gc(fg)
        @test node_feature(fg_) isa Matrix{T}
        @test size(node_feature(fg_)) == (out_channel, N)
        @test_throws MethodError gc(X)

        # Test with transposed features
        fgt = FeaturedGraph(adj, nf=Xt, graph_type=GRAPH_T)
        fgt_ = gc(fgt)
        @test node_feature(fgt_) isa Matrix{T}
        @test size(node_feature(fgt_)) == (out_channel, N)

        g = Zygote.gradient(x -> sum(node_feature(gc(x))), fg)[1]
        @test size(g.nf) == size(X)

        g = Zygote.gradient(model -> sum(node_feature(model(fg))), gc)[1]
        @test size(g.weight1) == size(gc.weight1)
        @test size(g.weight2) == size(gc.weight2)
        @test size(g.bias) == size(gc.bias)

        @testset "bias=false" begin
            @test length(Flux.params(GraphConv(2=>3))) == 3
            @test length(Flux.params(GraphConv(2=>3, bias=false))) == 2
        end
    end

    @testset "GATConv" begin

        X = rand(T, in_channel, N)
        Xt = transpose(rand(T, N, in_channel))

        for heads = [1, 2], concat = [true, false], adj_gat in [adj, adj_single_vertex]
            fg_gat = FeaturedGraph(adj_gat, nf=X, graph_type=GRAPH_T)
            gat = GATConv(in_channel=>out_channel, heads=heads, concat=concat)
            @test size(gat.weight) == (out_channel * heads, in_channel)
            @test size(gat.bias) == (out_channel * heads,)
            @test size(gat.a) == (2*out_channel, heads)

            fg_ = gat(fg_gat)
            Y = node_feature(fg_)
            @test Y isa Matrix{T}
            @test size(Y) == (concat ? (out_channel*heads, N) : (out_channel, N))
            @test_throws MethodError gat(X)

            # Test with transposed features
            fgt = FeaturedGraph(adj_gat, nf=Xt, graph_type=GRAPH_T)
            fgt_ = gat(fgt)
            @test node_feature(fg_) isa Matrix{T}
            @test size(node_feature(fgt_)) == (concat ? (out_channel*heads, N) : (out_channel, N))

            g = Zygote.gradient(x -> sum(node_feature(gat(x))), fg_gat)[1]
            @test size(g.nf) == size(X)

            g = Zygote.gradient(model -> sum(node_feature(model(fg_gat))), gat)[1]
            @test size(g.weight) == size(gat.weight)
            @test size(g.bias) == size(gat.bias)
            @test size(g.a) == size(gat.a)
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

        fg = FeaturedGraph(adj, nf=X, graph_type=GRAPH_T)
        fg_ = ggc(fg)
        @test node_feature(fg_) isa Matrix{T}
        @test size(node_feature(fg_)) == (out_channel, N)
        @test_throws MethodError ggc(X)

        # Test with transposed features
        fgt = FeaturedGraph(adj, nf=Xt, graph_type=GRAPH_T)
        fgt_ = ggc(fgt)
        @test node_feature(fgt_) isa Matrix{T}
        @test size(node_feature(fgt_)) == (out_channel, N)

        g = Zygote.gradient(x -> sum(node_feature(ggc(x))), fg)[1]
        @test size(g.nf) == size(X)

        g = Zygote.gradient(model -> sum(node_feature(model(fg))), ggc)[1]
        @test size(g.weight) == size(ggc.weight)
    end

    @testset "EdgeConv" begin
        X = rand(T, in_channel, N)
        Xt = transpose(rand(T, N, in_channel))
        ec = EdgeConv(Dense(2*in_channel, out_channel))

        fg = FeaturedGraph(adj, nf=X, graph_type=GRAPH_T)
        fg_ = ec(fg)
        @test node_feature(fg_) isa Matrix{T} 
        @test size(node_feature(fg_)) == (out_channel, N)
        @test_throws MethodError ec(X)

        # Test with transposed features
        fgt = FeaturedGraph(adj, nf=Xt, graph_type=GRAPH_T)
        fgt_ = ec(fgt)
        @test node_feature(fgt_) isa Matrix{T}
        @test size(node_feature(fgt_)) == (out_channel, N)

        g = Zygote.gradient(x -> sum(node_feature(ec(x))), fg)[1]
        @test size(g.nf) == size(X)

        g = Zygote.gradient(model -> sum(node_feature(model(fg))), ec)[1]
        @test size(g.nn.weight) == size(ec.nn.weight)
        @test size(g.nn.bias) == size(ec.nn.bias)
    end

    @testset "GINConv" begin
        X = rand(T, in_channel, N)
        nn = Dense(in_channel, out_channel)
        eps = 0.001f0
        fg = FeaturedGraph(adj, nf=X) 
        
        gc = GINConv(nn, eps=eps)
        @test gc.nn === nn
        
        fg_ = gc(fg)
        @test size(node_feature(fg_)) == (out_channel, N)

        g = Zygote.gradient(fg -> sum(node_feature(gc(fg))), fg)[1]
        @test size(g.nf) == size(X)

        g = Zygote.gradient(model -> sum(node_feature(model(fg))), gc)[1]
        @test size(g.nn.weight) == size(gc.nn.weight)
        @test size(g.nn.bias) == size(gc.nn.bias)
        
        @test !in(:eps, Flux.trainable(gc))
    end
end
