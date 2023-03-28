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

@testset "topk_feature" begin
    A = [0.0297 0.5901 0.088 0.5171;
         0.8307 0.303 0.6515 0.6379;
         0.914 0.928 0.4451 0.2695;
         0.6702 0.6893 0.7507 0.8954;
         0.3346 0.7997 0.5297 0.5197]
    B = [0.3168 0.1323 0.1752 0.1931 0.5065;
         0.3174 0.2766 0.9105 0.4954 0.5182;
         0.5303 0.4318 0.5692 0.3455 0.5418;
         0.0804 0.6114 0.8489 0.3934 0.152;
         0.3808 0.1458 0.0539 0.0857 0.3872]
    g1 = rand_graph(4, 2, ndata = (x = A,))
    g2 = rand_graph(5, 4, ndata = B)
    g = Flux.batch([g1, g2])
    output1 = topk_feature(g, g.ndata.x, 3)
    @test output1[1][:, :, 1] == [0.5901 0.5171 0.088;
           0.8307 0.6515 0.6379;
           0.928 0.914 0.4451;
           0.8954 0.7507 0.6893;
           0.7997 0.5297 0.5197]
    @test output1[1][:, :, 2] == [0.5065 0.3168 0.1931;
           0.9105 0.5182 0.4954;
           0.5692 0.5418 0.5303;
           0.8489 0.6114 0.3934;
           0.3872 0.3808 0.1458]
    @test output1[2][:, :, 1] == [2 4 3;
                                  1 3 4;
                                  2 1 3;
                                  4 3 2;
                                  2 3 4]
    @test output1[2][:, :, 2] == [5 1 4;
                                  3 5 4;
                                  3 5 1;
                                  3 2 4;
                                  5 1 2]
    output2 = topk_feature(g, g.ndata.x, 2; sortby = 5)
    @test output2[1][:, :, 1] == [0.5901 0.088
           0.303 0.6515;
           0.928 0.4451;
           0.6893 0.7507;
           0.7997 0.5297]
    @test output2[2][:, :, 1] == [2; 3;;]
end
