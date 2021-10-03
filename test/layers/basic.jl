@testset "basic" begin
    @testset "GNNChain" begin
        n, din, d, dout = 10, 3, 4, 2
        
        g = GNNGraph(random_regular_graph(n, 4), 
                    graph_type=GRAPH_T,
                    ndata= randn(Float32, din, n))
        
        gnn = GNNChain(GCNConv(din => d),
                       BatchNorm(d),
                       x -> tanh.(x),
                       GraphConv(d => d, tanh),
                       Dropout(0.5),
                       Dense(d, dout))

        testmode!(gnn)
        
        test_layer(gnn, g, rtol=1e-5, exclude_grad_fields=[:μ, :σ²])


        @testset "Parallel" begin
            AddResidual(l) = Parallel(+, identity, l) 

            gnn = GNNChain(ResGatedGraphConv(din => d, tanh),
                           BatchNorm(d),
                           AddResidual(ResGatedGraphConv(d => d, tanh)),
                           BatchNorm(d),
                           Dense(d, dout))

            testmode!(gnn)
                           
            test_layer(gnn, g, rtol=1e-5, exclude_grad_fields=[:μ, :σ²])
        end

        @testset "Only graph input" begin
            nin, nout = 2, 4
            ndata = rand(nin, 3)
            edata = rand(nin, 3)
            g = GNNGraph([1,1,2], [2, 3, 3], ndata=ndata, edata=edata)
            m = NNConv(nin => nout, Dense(2, nin*nout, tanh))
            chain = GNNChain(m)
            y = m(g, g.ndata.x, g.edata.e)
            @test m(g).ndata.x == y
            @test chain(g).ndata.x == y
        end
    end
end

