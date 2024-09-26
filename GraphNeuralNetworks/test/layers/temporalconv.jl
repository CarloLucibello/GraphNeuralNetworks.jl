in_channel = 3
out_channel = 5
N = 4
S = 5
T = Float32

g1 = GNNGraph(rand_graph(N,8),
                ndata = rand(T, in_channel, N),
                graph_type = :sparse)

tg = TemporalSnapshotsGNNGraph([g1 for _ in 1:S])

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

@testset "GConvLSTMCell" begin
    gconvlstm = GraphNeuralNetworks.GConvLSTMCell(in_channel => out_channel, 2, g1.num_nodes)
    (h, c), h = gconvlstm(gconvlstm.state0, g1, g1.ndata.x)
    @test size(h) == (out_channel, N)
    @test size(c) == (out_channel, N)
end

@testset "GConvLSTM" begin
    gconvlstm = GConvLSTM(in_channel => out_channel, 2, g1.num_nodes)
    @test size(Flux.gradient(x -> sum(gconvlstm(g1, x)), g1.ndata.x)[1]) == (in_channel, N)
    model = GNNChain(GConvLSTM(in_channel => out_channel, 2, g1.num_nodes), Dense(out_channel, 1))
end

@testset "GConvGRUCell" begin
    gconvlstm = GraphNeuralNetworks.GConvGRUCell(in_channel => out_channel, 2, g1.num_nodes)
    h, h = gconvlstm(gconvlstm.state0, g1, g1.ndata.x)
    @test size(h) == (out_channel, N)
end

@testset "GConvGRU" begin
    gconvlstm = GConvGRU(in_channel => out_channel, 2, g1.num_nodes)
    @test size(Flux.gradient(x -> sum(gconvlstm(g1, x)), g1.ndata.x)[1]) == (in_channel, N)
    model = GNNChain(GConvGRU(in_channel => out_channel, 2, g1.num_nodes), Dense(out_channel, 1))
    @test size(model(g1, g1.ndata.x)) == (1, N)
    @test model(g1) isa GNNGraph            
end

@testset "DCGRU" begin
    dcgru = DCGRU(in_channel => out_channel, 2, g1.num_nodes)
    @test size(Flux.gradient(x -> sum(dcgru(g1, x)), g1.ndata.x)[1]) == (in_channel, N)
    model = GNNChain(DCGRU(in_channel => out_channel, 2, g1.num_nodes), Dense(out_channel, 1))
    @test size(model(g1, g1.ndata.x)) == (1, N)
    @test model(g1) isa GNNGraph            
end

@testset "EvolveGCNO" begin
    evolvegcno = EvolveGCNO(in_channel => out_channel)
    @test length(Flux.gradient(x -> sum(sum(evolvegcno(tg, x))), tg.ndata.x)[1]) == S
    @test size(evolvegcno(tg, tg.ndata.x)[1]) ==  (out_channel, N)
end

@testset "GINConv" begin
    ginconv = GINConv(Dense(in_channel => out_channel),0.3)
    @test length(ginconv(tg, tg.ndata.x)) == S
    @test size(ginconv(tg, tg.ndata.x)[1]) == (out_channel, N) 
    @test length(Flux.gradient(x ->sum(sum(ginconv(tg, x))), tg.ndata.x)[1]) == S    
end

@testset "ChebConv" begin
    chebconv = ChebConv(in_channel => out_channel, 5)
    @test length(chebconv(tg, tg.ndata.x)) == S
    @test size(chebconv(tg, tg.ndata.x)[1]) == (out_channel, N) 
    @test length(Flux.gradient(x ->sum(sum(chebconv(tg, x))), tg.ndata.x)[1]) == S
end

@testset "GATConv" begin
    gatconv = GATConv(in_channel => out_channel)
    @test length(gatconv(tg, tg.ndata.x)) == S
    @test size(gatconv(tg, tg.ndata.x)[1]) == (out_channel, N) 
    @test length(Flux.gradient(x ->sum(sum(gatconv(tg, x))), tg.ndata.x)[1]) == S
end

@testset "GATv2Conv" begin
    gatv2conv = GATv2Conv(in_channel => out_channel)
    @test length(gatv2conv(tg, tg.ndata.x)) == S
    @test size(gatv2conv(tg, tg.ndata.x)[1]) == (out_channel, N) 
    @test length(Flux.gradient(x ->sum(sum(gatv2conv(tg, x))), tg.ndata.x)[1]) == S
end

@testset "GatedGraphConv" begin
    gatedgraphconv = GatedGraphConv(5, 5)
    @test length(gatedgraphconv(tg, tg.ndata.x)) == S
    @test size(gatedgraphconv(tg, tg.ndata.x)[1]) == (out_channel, N) 
    @test length(Flux.gradient(x ->sum(sum(gatedgraphconv(tg, x))), tg.ndata.x)[1]) == S
end

@testset "CGConv" begin
    cgconv = CGConv(in_channel => out_channel)
    @test length(cgconv(tg, tg.ndata.x)) == S
    @test size(cgconv(tg, tg.ndata.x)[1]) == (out_channel, N) 
    @test length(Flux.gradient(x ->sum(sum(cgconv(tg, x))), tg.ndata.x)[1]) == S
end

@testset "SGConv" begin
    sgconv = SGConv(in_channel => out_channel)
    @test length(sgconv(tg, tg.ndata.x)) == S
    @test size(sgconv(tg, tg.ndata.x)[1]) == (out_channel, N) 
    @test length(Flux.gradient(x ->sum(sum(sgconv(tg, x))), tg.ndata.x)[1]) == S
end

@testset "TransformerConv" begin
    transformerconv = TransformerConv(in_channel => out_channel)
    @test length(transformerconv(tg, tg.ndata.x)) == S
    @test size(transformerconv(tg, tg.ndata.x)[1]) == (out_channel, N) 
    @test length(Flux.gradient(x ->sum(sum(transformerconv(tg, x))), tg.ndata.x)[1]) == S
end

@testset "GCNConv" begin
    gcnconv = GCNConv(in_channel => out_channel)
    @test length(gcnconv(tg, tg.ndata.x)) == S
    @test size(gcnconv(tg, tg.ndata.x)[1]) == (out_channel, N) 
    @test length(Flux.gradient(x ->sum(sum(gcnconv(tg, x))), tg.ndata.x)[1]) == S    
end

@testset "ResGatedGraphConv" begin
    resgatedconv = ResGatedGraphConv(in_channel => out_channel, tanh)
    @test length(resgatedconv(tg, tg.ndata.x)) == S
    @test size(resgatedconv(tg, tg.ndata.x)[1]) == (out_channel, N) 
    @test length(Flux.gradient(x ->sum(sum(resgatedconv(tg, x))), tg.ndata.x)[1]) == S    
end

@testset "SAGEConv" begin 
    sageconv = SAGEConv(in_channel => out_channel)
    @test length(sageconv(tg, tg.ndata.x)) == S
    @test size(sageconv(tg, tg.ndata.x)[1]) == (out_channel, N) 
    @test length(Flux.gradient(x ->sum(sum(sageconv(tg, x))), tg.ndata.x)[1]) == S    
end

@testset "GraphConv" begin
    graphconv = GraphConv(in_channel => out_channel, tanh)
    @test length(graphconv(tg, tg.ndata.x)) == S
    @test size(graphconv(tg, tg.ndata.x)[1]) == (out_channel, N) 
    @test length(Flux.gradient(x ->sum(sum(graphconv(tg, x))), tg.ndata.x)[1]) == S    
end

