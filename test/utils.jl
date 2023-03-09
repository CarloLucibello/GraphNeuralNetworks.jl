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
        A = [0.0297 0.8307 0.9140 0.6702 0.3346;
             0.5901 0.3030 0.9280 0.6893 0.7997;
             0.0880 0.6515 0.4451 0.7507 0.5297;
             0.5171 0.6379 0.2695 0.8954 0.5197]
        B = [0.3168 0.3174 0.5303 0.0804 0.3808;
             0.1752 0.9105 0.5692 0.8489 0.0539;
             0.1931 0.4954 0.3455 0.3934 0.0857;
             0.5065 0.5182 0.5418 0.1520 0.3872]
        C = [0.0297 0.0297 0.8307 0.9140 0.6702 0.3346;
             0.5901 0.5901 0.3030 0.9280 0.6893 0.7997;
             0.0880 0.0880 0.6515 0.4451 0.7507 0.5297;
             0.5171 0.5171 0.6379 0.2695 0.8954 0.5197]
        g1 = rand_graph(5, 6, ndata = (w = A,))
        g2 = rand_graph(5, 6, ndata = (w = B,))
        g3 = rand_graph(5, 6, edata = (e = C,))
        g = Flux.batch([g1, g2])
        output1 = topk_nodes(g1, :w, 3)
        output2 = topk_nodes(g1, :w, 3; sortby = 5)
        output3 = topk_edges(g3, :e, 3; sortby = 6)
        output_batch = topk_nodes(g, :w, 3; sortby = 5)
        correctout1 = [0.5901 0.8307 0.9280 0.8954 0.7997;
                       0.5171 0.6515 0.9140 0.7507 0.5297;
                       0.0880 0.6379 0.4451 0.6893 0.5197]
        correctout2 = [0.5901 0.3030 0.9280 0.6893 0.7997;
                       0.0880 0.6515 0.4451 0.7507 0.5297;
                       0.5171 0.6379 0.2695 0.8954 0.5197]
        correctout3 = [0.5901 0.5901 0.3030 0.9280 0.6893 0.7997;
                       0.0880 0.0880 0.6515 0.4451 0.7507 0.5297;
                       0.5171 0.5171 0.6379 0.2695 0.8954 0.5197]
        correctout_batch = [0.5901 0.3030 0.9280 0.6893 0.7997 0.5065 0.5182 0.5418 0.1520 0.3872;
                            0.0880 0.6515 0.4451 0.7507 0.5297 0.3168 0.3174 0.5303 0.0804 0.3808;
                            0.5171 0.6379 0.2695 0.8954 0.5197 0.1931 0.4954 0.3455 0.3934 0.0857]
        @test output1 == correctout1
        @test output2 == correctout2
        @test output3 == correctout3
        @test output_batch == correctout_batch
    end
end
