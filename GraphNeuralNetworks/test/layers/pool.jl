@testset "GlobalPool" begin
    p = GlobalPool(+)
    n = 10
    chin = 6
    X = rand(Float32, 6, n)
    g = GNNGraph(random_regular_graph(n, 4), ndata = X, graph_type = GRAPH_T)
    u = p(g, X)
    @test u ≈ sum(X, dims = 2)

    ng = 3
    g = Flux.batch([GNNGraph(random_regular_graph(n, 4),
                                ndata = rand(Float32, chin, n),
                                graph_type = GRAPH_T)
                    for i in 1:ng])
    u = p(g, g.ndata.x)
    @test size(u) == (chin, ng)
    @test u[:, [1]] ≈ sum(g.ndata.x[:, 1:n], dims = 2)
    @test p(g).gdata.u == u

    test_layer(p, g, rtol = 1e-5, exclude_grad_fields = [:aggr], outtype = :graph)
end

@testset "GlobalAttentionPool" begin
    n = 10
    chin = 6
    chout = 5
    ng = 3

    fgate = Dense(chin, 1)
    ffeat = Dense(chin, chout)
    p = GlobalAttentionPool(fgate, ffeat)
    @test length(Flux.params(p)) == 4

    g = Flux.batch([GNNGraph(random_regular_graph(n, 4),
                                ndata = rand(Float32, chin, n),
                                graph_type = GRAPH_T)
                    for i in 1:ng])

    test_layer(p, g, rtol = 1e-5, outtype = :graph, outsize = (chout, ng))
end

@testset "TopKPool" begin
    N = 10
    k, in_channel = 4, 7
    X = rand(in_channel, N)
    for T in [Bool, Float64]
        adj = rand(T, N, N)
        p = TopKPool(adj, k, in_channel)
        @test eltype(p.p) === Float32
        @test size(p.p) == (in_channel,)
        @test eltype(p.Ã) === T
        @test size(p.Ã) == (k, k)
        y = p(X)
        @test size(y) == (in_channel, k)
    end
end

@testset "topk_index" begin
    X = [8, 7, 6, 5, 4, 3, 2, 1]
    @test topk_index(X, 4) == [1, 2, 3, 4]
    @test topk_index(X', 4) == [1, 2, 3, 4]
end

@testset "Set2Set" begin
    n_in = 3
    n_iters = 2
    n_layers = 1
    g = batch([rand_graph(10, 40, graph_type = GRAPH_T) for _ in 1:5])
    g = GNNGraph(g, ndata = rand(Float32, n_in, g.num_nodes))
    l = Set2Set(n_in, n_iters, n_layers)
    y = l(g, node_features(g))
    @test size(y) == (2 * n_in, g.num_graphs)
    
    ## TODO the numerical gradient seems to be 3 times smaller than zygote one
    # test_layer(l, g, rtol = 1e-4, atol=1e-4, outtype = :graph, outsize = (2 * n_in, g.num_graphs), 
    #         verbose=true, exclude_grad_fields = [:state0, :state])
end