@testset "message passing" begin 
    in_channel = 10
    out_channel = 5
    num_V = 6
    num_E = 14
    T = Float32

    adj =  [0 1 0 0 0 0
            1 0 0 1 1 1
            0 0 0 0 0 1
            0 1 0 0 1 0
            0 1 0 1 0 1
            0 1 1 0 1 0]

    
    X = rand(T, in_channel, num_V)
    E = rand(T, in_channel, num_E)

    g = GNNGraph(adj, graph_type=GRAPH_T)

    @testset "propagate" begin
        function message(xi, xj, e) 
            @test xi === nothing
            @test e === nothing
            ones(T, out_channel, size(xj, 2))
        end
        
        m = propagate(message, g, +, xj=X)

        @test size(m) == (out_channel, num_V)
    end


    @testset "apply_edges" begin
        m = apply_edges(g, e=E) do xi, xj, e
                @test xi === nothing
                @test xj === nothing
                ones(out_channel, size(e, 2))
            end 

        @test m == ones(out_channel, num_E)

        # With NamedTuple input
        m = apply_edges(g, xj=(;a=X, b=2X), e=E) do xi, xj, e
                @test xi === nothing
                @test xj.b == 2*xj.a
                @test size(xj.a, 2) == size(xj.b, 2) == size(e, 2)
                ones(out_channel, size(e, 2))
            end 
    
        # NamedTuple output
        m = apply_edges(g, e=E) do xi, xj, e
            @test xi === nothing
            @test xj === nothing
            (; a=ones(out_channel, size(e, 2)))
        end 

        @test m.a == ones(out_channel, num_E)
    end
end
