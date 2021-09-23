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
        
        test_layer(gnn, g, rtol=1e-5)


        @testset "Parallel" begin
            AddResidual(l) = Parallel(+, identity, l) 

            gnn = GNNChain(ResGatedGraphConv(din => d, tanh),
                           BatchNorm(d),
                           AddResidual(ResGatedGraphConv(d => d, tanh)),
                           BatchNorm(d),
                           Dense(d, dout))

            testmode!(gnn)
                           
            test_layer(gnn, g, rtol=1e-5)
        end
    end
end

