@testset "cuda/conv" begin
    in_channel = 3
    out_channel = 5
    N = 4
    adj =  [0 1 0 1
            1 0 1 0
            0 1 0 1
            1 0 1 0]

    fg = FeaturedGraph(adj, graph_type=GRAPH_T)
    X = rand(Float32, in_channel, N)
   
    @testset "GCNConv" begin
        m = GCNConv(in_channel => out_channel)
        gpugradtest(m, fg, X)
    end

    @testset "ChebConv" begin
        k = 6
        m = ChebConv(in_channel => out_channel, k)
        @test_broken gpugradtest(m, fg, X)
    end

    @testset "GraphConv" begin
        m = GraphConv(in_channel => out_channel)
        gpugradtest(m, fg, X)
    end

    @testset "GATConv" begin
        m = GATConv(in_channel => out_channel)
        gpugradtest(m, fg, X)
    end

    @testset "GINConv" begin
        m = GINConv(Dense(in_channel, out_channel), eps=0.1f0)
        gpugradtest(m, fg, X)
    end

    @testset "GatedGraphConv" begin
        num_layers = 3
        m = GatedGraphConv(out_channel, num_layers)
        gpugradtest(m, fg, X)
    end

    @testset "EdgeConv" begin
        m = EdgeConv(Dense(2*in_channel, out_channel))
        gpugradtest(m, fg, X)
    end
end
