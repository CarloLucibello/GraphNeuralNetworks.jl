@testset "Utils" begin
    De, Dx = 3, 2
    g = Flux.batch([GNNGraph(erdos_renyi(10, 30), ndata=rand(Dx, 10), edata=rand(De, 30)) for i=1:5]) 
    x = g.ndata.x
    e = g.edata.e

    @testset "readout_nodes" begin
        r = readout_nodes(g, x, mean)
        @test size(r) == (Dx, g.num_graphs)
        @test r[:,2] ≈ mean(getgraph(g, 2).ndata.x, dims=2)
    end

    @testset "readout_edges" begin
        r = readout_edges(g, e, mean)
        @test size(r) == (De, g.num_graphs)
        @test r[:,2] ≈ mean(getgraph(g, 2).edata.e, dims=2)
    end

    @testset "softmax_nodes" begin
        r = softmax_nodes(g, x)
        @test size(r) == size(x)
        @test r[:,1:10] ≈ softmax(getgraph(g, 1).ndata.x, dims=2)
    end

    @testset "softmax_edges" begin
        r = softmax_edges(g, e)
        @test size(r) == size(e)
        @test r[:,1:60] ≈ softmax(getgraph(g, 1).edata.e, dims=2)
    end
end