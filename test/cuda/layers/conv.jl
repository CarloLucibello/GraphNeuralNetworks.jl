@testset "cuda/conv" begin
    in_channel = 3
    out_channel = 5
    N = 4
    adj =  [0 1 0 1
            1 0 1 0
            0 1 0 1
            1 0 1 0]

    fg = FeaturedGraph(adj, graph_type=GRAPH_T) |> gpu

    @testset "GCNConv" begin
        gc = GCNConv(in_channel=>out_channel) |> gpu
        @test size(gc.weight) == (out_channel, in_channel)
        @test size(gc.bias) == (out_channel,)
        @test adjacency_matrix(gc.fg |> cpu) == adj

        X = rand(in_channel, N) |> gpu
        Y = gc(fg, X)
        @test size(Y) == (out_channel, N)

        g = Zygote.gradient(x -> sum(gc(x)), X)[1]
        @test size(g) == size(X)

        g = Zygote.gradient(model -> sum(model(X)), gc)[1]
        @test size(g.weight) == size(gc.weight)
        @test size(g.bias) == size(gc.bias)
    end


    @testset "ChebConv" begin
        k = 6
        cc = ChebConv(in_channel=>out_channel, k) |> gpu
        @test size(cc.weight) == (out_channel, in_channel, k)
        @test size(cc.bias) == (out_channel,)
        @test adjacency_matrix(cc.fg |> cpu) == adj
        @test cc.k == k
        
        @test_broken begin 
            X = rand(in_channel, N) |> gpu
            Y = cc(fg, X)
            @test size(Y) == (out_channel, N)

            g = Zygote.gradient(x -> sum(cc(x)), X)[1]
            @test size(g) == size(X)

            g = Zygote.gradient(model -> sum(model(fg, X)), cc)[1]
            @test size(g.weight) == size(cc.weight)
            @test size(g.bias) == size(cc.bias)

            true
        end
    end

    @testset "GraphConv" begin
        gc = GraphConv(in_channel=>out_channel) |> gpu
        @test size(gc.weight1) == (out_channel, in_channel)
        @test size(gc.weight2) == (out_channel, in_channel)
        @test size(gc.bias) == (out_channel,)

        X = rand(in_channel, N) |> gpu
        Y = gc(fg, X)
        @test size(Y) == (out_channel, N)

        g = Zygote.gradient(x -> sum(gc(fg, x)), X)[1]
        @test size(g) == size(X)

        g = Zygote.gradient(model -> sum(model(fg, X)), gc)[1]
        @test size(g.weight1) == size(gc.weight1)
        @test size(g.weight2) == size(gc.weight2)
        @test size(g.bias) == size(gc.bias)
    end

    @testset "GATConv" begin
        gat = GATConv(in_channel=>out_channel) |> gpu
        @test size(gat.weight) == (out_channel, in_channel)
        @test size(gat.bias) == (out_channel,)

        X = rand(in_channel, N) |> gpu
        Y = gat(fg, X)
        @test size(Y) == (out_channel, N)

        g = Zygote.gradient(x -> sum(gat(fg, x)), X)[1]
        @test size(g) == size(X)

        g = Zygote.gradient(model -> sum(model(fg, X)), gat)[1]
        @test size(g.weight) == size(gat.weight)
        @test size(g.bias) == size(gat.bias)
        @test size(g.a) == size(gat.a)
    end

    @testset "GatedGraphConv" begin
        num_layers = 3
        ggc = GatedGraphConv(out_channel, num_layers) |> gpu
        @test size(ggc.weight) == (out_channel, out_channel, num_layers)

        X = rand(in_channel, N) |> gpu
        Y = ggc(fg, X)
        @test size(Y) == (out_channel, N)

        g = Zygote.gradient(x -> sum(ggc(fg, x)), X)[1]
        @test size(g) == size(X)

        g = Zygote.gradient(model -> sum(model(fg, X)), ggc)[1]
        @test size(g.weight) == size(ggc.weight)
    end

    @testset "EdgeConv" begin
        ec = EdgeConv(Dense(2*in_channel, out_channel)) |> gpu
        X = rand(in_channel, N) |> gpu
        Y = ec(fg, X)
        @test size(Y) == (out_channel, N)

        g = Zygote.gradient(x -> sum(ec(fg, x)), X)[1]
        @test size(g) == size(X)

        g = Zygote.gradient(model -> sum(model(fg, X)), ec)[1]
        @test size(g.nn.weight) == size(ec.nn.weight)
        @test size(g.nn.bias) == size(ec.nn.bias)
    end
end
