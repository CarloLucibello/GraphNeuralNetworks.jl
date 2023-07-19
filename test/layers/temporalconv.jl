in_channel = 3
out_channel = 5
N = 4
T = Float32

g1 = GNNGraph(rand_graph(N,8),
                ndata = rand(T, in_channel, N),
                graph_type = :sparse)

@testset "TGCNCell" begin
    tgcn = TGCNCell(in_channel => out_channel)
    h, x̃ = tgcn(tgcn.state0, g1, g1.ndata.x)
    @test size(h) == (out_channel, N)
    @test size(x̃) == (out_channel, N)
    @test h == x̃
end