@testmodule TemporalConvTestModule begin
    using GraphNeuralNetworks
    export in_channel, out_channel, N, S, T, g1, tg
    in_channel = 3
    out_channel = 5
    N = 4
    S = 5
    T = Float32

    g1 = GNNGraph(rand_graph(N,8),
                    ndata = rand(T, in_channel, N),
                    graph_type = :sparse)

    tg = TemporalSnapshotsGNNGraph([g1 for _ in 1:S])
end

@testitem "TGCNCell" setup=[TemporalConvTestModule, TestModule] begin
    using .TemporalConvTestModule, .TestModule
    tgcn = GraphNeuralNetworks.TGCNCell(in_channel => out_channel)
    h, x̃ = tgcn(tgcn.state0, g1, g1.ndata.x)
    @test size(h) == (out_channel, N)
    @test size(x̃) == (out_channel, N)
    @test h == x̃
end

@testitem "TGCN" setup=[TemporalConvTestModule, TestModule] begin
    using .TemporalConvTestModule, .TestModule
    tgcn = TGCN(in_channel => out_channel)
    @test size(Flux.gradient(x -> sum(tgcn(g1, x)), g1.ndata.x)[1]) == (in_channel, N)
    model = GNNChain(TGCN(in_channel => out_channel), Dense(out_channel, 1))
    @test size(model(g1, g1.ndata.x)) == (1, N)
    @test model(g1) isa GNNGraph            
end

@testitem "A3TGCN" setup=[TemporalConvTestModule, TestModule] begin
    using .TemporalConvTestModule, .TestModule
    a3tgcn = A3TGCN(in_channel => out_channel)
    @test size(Flux.gradient(x -> sum(a3tgcn(g1, x)), g1.ndata.x)[1]) == (in_channel, N)
    model = GNNChain(A3TGCN(in_channel => out_channel), Dense(out_channel, 1))
    @test size(model(g1, g1.ndata.x)) == (1, N)
    @test model(g1) isa GNNGraph            
end

@testitem "GConvLSTMCell" setup=[TemporalConvTestModule, TestModule] begin
    using .TemporalConvTestModule, .TestModule
    gconvlstm = GraphNeuralNetworks.GConvLSTMCell(in_channel => out_channel, 2, g1.num_nodes)
    (h, c), h = gconvlstm(gconvlstm.state0, g1, g1.ndata.x)
    @test size(h) == (out_channel, N)
    @test size(c) == (out_channel, N)
end

@testitem "GConvLSTM" setup=[TemporalConvTestModule, TestModule] begin
    using .TemporalConvTestModule, .TestModule
    gconvlstm = GConvLSTM(in_channel => out_channel, 2, g1.num_nodes)
    @test size(Flux.gradient(x -> sum(gconvlstm(g1, x)), g1.ndata.x)[1]) == (in_channel, N)
    model = GNNChain(GConvLSTM(in_channel => out_channel, 2, g1.num_nodes), Dense(out_channel, 1))
end

@testitem "GConvGRUCell" setup=[TemporalConvTestModule, TestModule] begin
    using .TemporalConvTestModule, .TestModule
    gconvlstm = GraphNeuralNetworks.GConvGRUCell(in_channel => out_channel, 2, g1.num_nodes)
    h, h = gconvlstm(gconvlstm.state0, g1, g1.ndata.x)
    @test size(h) == (out_channel, N)
end

@testitem "GConvGRU" setup=[TemporalConvTestModule, TestModule] begin
    using .TemporalConvTestModule, .TestModule
    gconvlstm = GConvGRU(in_channel => out_channel, 2, g1.num_nodes)
    @test size(Flux.gradient(x -> sum(gconvlstm(g1, x)), g1.ndata.x)[1]) == (in_channel, N)
    model = GNNChain(GConvGRU(in_channel => out_channel, 2, g1.num_nodes), Dense(out_channel, 1))
    @test size(model(g1, g1.ndata.x)) == (1, N)
    @test model(g1) isa GNNGraph            
end

@testitem "DCGRU" setup=[TemporalConvTestModule, TestModule] begin
    using .TemporalConvTestModule, .TestModule
    dcgru = DCGRU(in_channel => out_channel, 2, g1.num_nodes)
    @test size(Flux.gradient(x -> sum(dcgru(g1, x)), g1.ndata.x)[1]) == (in_channel, N)
    model = GNNChain(DCGRU(in_channel => out_channel, 2, g1.num_nodes), Dense(out_channel, 1))
    @test size(model(g1, g1.ndata.x)) == (1, N)
    @test model(g1) isa GNNGraph            
end

@testitem "EvolveGCNO" setup=[TemporalConvTestModule, TestModule] begin
    using .TemporalConvTestModule, .TestModule
    evolvegcno = EvolveGCNO(in_channel => out_channel)
    @test length(Flux.gradient(x -> sum(sum(evolvegcno(tg, x))), tg.ndata.x)[1]) == S
    @test size(evolvegcno(tg, tg.ndata.x)[1]) ==  (out_channel, N)
end

@testitem "GINConv" setup=[TemporalConvTestModule, TestModule] begin
    using .TemporalConvTestModule, .TestModule
    ginconv = GINConv(Dense(in_channel => out_channel),0.3)
    @test length(ginconv(tg, tg.ndata.x)) == S
    @test size(ginconv(tg, tg.ndata.x)[1]) == (out_channel, N) 
    @test length(Flux.gradient(x ->sum(sum(ginconv(tg, x))), tg.ndata.x)[1]) == S    
end

@testitem "GraphConv" setup=[TemporalConvTestModule, TestModule] begin
    using .TemporalConvTestModule, .TestModule
    graphconv = GraphConv(in_channel => out_channel, tanh)
    @test length(graphconv(tg, tg.ndata.x)) == S
    @test size(graphconv(tg, tg.ndata.x)[1]) == (out_channel, N) 
    @test length(Flux.gradient(x ->sum(sum(graphconv(tg, x))), tg.ndata.x)[1]) == S    
end

