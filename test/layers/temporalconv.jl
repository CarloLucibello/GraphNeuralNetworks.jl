in_channel = 3
out_channel = 5
N = 4
T = Float32

g1 = GNNGraph(rand_graph(N,8),
                ndata = rand(T, in_channel, N),
                graph_type = :sparse)

@testset "TGCNCell" begin
    tgcn = GraphNeuralNetworks.TGCNCell(in_channel => out_channel)
    h, x̃ = tgcn(tgcn.state0, g1, g1.ndata.x)
    @test size(h) == (out_channel, N)
    @test size(x̃) == (out_channel, N)
    @test h == x̃
end

@testset "TGCN" begin
    tgcn = TGCN(in_channel => out_channel)
    @test size(Flux.gradient(x -> sum(tgcn(g1, x)), g1.ndata.x)[1]) == (in_channel, N)
    model = GNNChain(TGCN(in_channel => out_channel), Dense(out_channel, 1))
    @test size(model(g1, g1.ndata.x)) == (1, N)
    @test model(g1) isa GNNGraph            
end

@testset "A3TGCN" begin
    a3tgcn = A3TGCN(in_channel => out_channel)
    @test size(Flux.gradient(x -> sum(a3tgcn(g1, x)), g1.ndata.x)[1]) == (in_channel, N)
    model = GNNChain(A3TGCN(in_channel => out_channel), Dense(out_channel, 1))
    @test size(model(g1, g1.ndata.x)) == (1, N)
    @test model(g1) isa GNNGraph            
end

@testset "TemporalGraphConv" begin
    snapshots = [rand_graph(20,40; ndata = rand(3,20)), rand_graph(20,14; ndata = rand(3,20)), rand_graph(20,20; ndata = rand(3,20))]
    tsg = TemporalSnapshotsGNNGraph(snapshots)
    
    AGNN = GraphNeuralNetworks.TemporalGraphConv(AGNNConv())
    @test size(Flux.gradient(x -> sum(sum(AGNN(tsg, x))), tsg.ndata.x)[1][1]) == (3, 20)
    @test tsg1 = AGNN(tsg) isa TemporalSnapshotsGNNGraph
    @test size(AGNN(tsg).snapshots[1].ndata.x) == (3, 20)

    GCN = GraphNeuralNetworks.TemporalGraphConv(GCNConv(3=>5))
    @test size(Flux.gradient(x -> sum(sum(GCN(tsg, x))), tsg.ndata.x)[1][1]) == (3, 20)
    @test tsg1 = GCN(tsg) isa TemporalSnapshotsGNNGraph
    @test size(GCN(tsg).snapshots[1].ndata.x) == (5, 20)
end;