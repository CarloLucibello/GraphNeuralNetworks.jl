@testset "Utils" begin
    De, Dx = 3, 2
    g = Flux.batch([GNNGraph(erdos_renyi(10, 30),
                             ndata = rand(Dx, 10),
                             edata = rand(De, 30),
                             graph_type = GRAPH_T) for i in 1:5])
    x = g.ndata.x
    e = g.edata.e

    @testset "reduce_nodes" begin
        r = reduce_nodes(mean, g, x)
        @test size(r) == (Dx, g.num_graphs)
        @test r[:, 2] ≈ mean(getgraph(g, 2).ndata.x, dims = 2)
    end

    @testset "reduce_edges" begin
        r = reduce_edges(mean, g, e)
        @test size(r) == (De, g.num_graphs)
        @test r[:, 2] ≈ mean(getgraph(g, 2).edata.e, dims = 2)
    end

    @testset "softmax_nodes" begin
        r = softmax_nodes(g, x)
        @test size(r) == size(x)
        @test r[:, 1:10] ≈ softmax(getgraph(g, 1).ndata.x, dims = 2)
    end

    @testset "softmax_edges" begin
        r = softmax_edges(g, e)
        @test size(r) == size(e)
        @test r[:, 1:60] ≈ softmax(getgraph(g, 1).edata.e, dims = 2)
    end

    @testset "broadcast_nodes" begin
        z = rand(4, g.num_graphs)
        r = broadcast_nodes(g, z)
        @test size(r) == (4, g.num_nodes)
        @test r[:, 1] ≈ z[:, 1]
        @test r[:, 10] ≈ z[:, 1]
        @test r[:, 11] ≈ z[:, 2]
    end

    @testset "broadcast_edges" begin
        z = rand(4, g.num_graphs)
        r = broadcast_edges(g, z)
        @test size(r) == (4, g.num_edges)
        @test r[:, 1] ≈ z[:, 1]
        @test r[:, 60] ≈ z[:, 1]
        @test r[:, 61] ≈ z[:, 2]
    end

    @testset "softmax_edge_neighbors" begin
        s = [1, 2, 3, 4]
        t = [5, 5, 6, 6]
        g2 = GNNGraph(s, t)
        e2 = randn(Float32, 3, g2.num_edges)
        z = softmax_edge_neighbors(g2, e2)
        @test size(z) == size(e2)
        @test z[:, 1:2] ≈ NNlib.softmax(e2[:, 1:2], dims = 2)
        @test z[:, 3:4] ≈ NNlib.softmax(e2[:, 3:4], dims = 2)
    end

    @testset "topk_nodes" begin
        A = [1.0 5.0 9.0; 2.0 6.0 10.0; 3.0 7.0 11.0; 4.0 8.0 12.0]
        B = [0.318907 0.189981 0.991791;
             0.547022 0.977349 0.680538;
             0.921823 0.35132 0.494715;
             0.451793 0.00704976 0.0189275]
        g1 = rand_graph(3, 6, ndata = (x = A,))
        g2 = rand_graph(3, 6, ndata = B)
        output1 = topk_nodes(g1, :x, 2)
        output2 = topk_nodes(g2, :x, 1, sortby = 2)
        @test output1 == [9.0 5.0;
                          10.0 6.0;
                          11.0 7.0;
                          12.0 8.0]
        @test output2 == [0.189981;
                          0.977349;
                          0.35132;
                          0.00704976;;]
        g = Flux.batch([g1, g2])
        output3 = topk_nodes(g, :x, 2; sortby = 4)
        @test output3 == [9.0 5.0 0.318907 0.991791;
               10.0 6.0 0.547022 0.680538;
               11.0 7.0 0.921823 0.494715;
               12.0 8.0 0.451793 0.0189275]
    end

    @testset "topk_edges" begin
        A = [0.157163 0.561874 0.886584 0.0475203 0.72576 0.815986;
             0.852048 0.974619 0.0345627 0.874303 0.614322 0.113491]
        g1 = rand_graph(5, 6, edata = (x = A,))
        output1 = topk_edges(g1, :x, 2)
        @test output1 == [0.886584 0.815986;
                          0.974619 0.874303]
    end
end
