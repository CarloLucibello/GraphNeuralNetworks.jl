@testset "basic" begin
    @testset "GNNChain" begin
        n, din, d, dout = 10, 3, 4, 2
        
        g = GNNGraph(random_regular_graph(n, 4), graph_type=GRAPH_T)
        
        gnn = GNNChain(GCNConv(din => d),
                       BatchNorm(d),
                       x -> relu.(x),
                       GraphConv(d => d, relu),
                       Dropout(0.5),
                       Dense(d, dout))
        
        X = randn(Float32, din, n)

        y = gnn(g, X)
  
        @test y isa Matrix{Float32}
        @test size(y) == (dout, n)

        @test length(params(gnn)) == 9
        
        gs = gradient(x -> sum(gnn(g, x)), X)[1]
        @test gs isa Matrix{Float32}
        @test size(gs) == size(X) 

        gs = gradient(() -> sum(gnn(g, X)), Flux.params(gnn))
        for p in Flux.params(gnn)
            @test eltype(gs[p]) == Float32
            @test size(gs[p]) == size(p)
        end
    end
end

