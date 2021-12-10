@testset "Query" begin
    @testset "is_bidirected" begin
        g = rand_graph(10, 20, bidirected=true, graph_type=GRAPH_T)
        @test is_bidirected(g)
        
        g = rand_graph(10, 20, bidirected=false, graph_type=GRAPH_T)
        @test !is_bidirected(g)
    end

    @testset "has_multi_edges" begin
        if GRAPH_T == :coo
            s = [1, 1, 2, 3]
            t = [2, 2, 2, 4]
            g = GNNGraph(s, t, graph_type=GRAPH_T)
            @test has_multi_edges(g)

            s = [1, 2, 2, 3]
            t = [2, 1, 2, 4]
            g = GNNGraph(s, t, graph_type=GRAPH_T)
            @test !has_multi_edges(g)
        end
    end

    @testset "has_self_loops" begin
        s = [1, 1, 2, 3]
        t = [2, 2, 2, 4]
        g = GNNGraph(s, t, graph_type=GRAPH_T)
        @test has_self_loops(g)

        s = [1, 1, 2, 3]
        t = [2, 2, 3, 4]
        g = GNNGraph(s, t, graph_type=GRAPH_T)
        @test !has_self_loops(g)
    end

    @testset "degree" begin 
        @testset "unweighted" begin
            s = [1, 1, 2, 3]
            t = [2, 2, 2, 4]
            g = GNNGraph(s, t, graph_type=GRAPH_T)

            @test degree(g) == degree(g; dir=:out) == [2, 1, 1, 0] # default is outdegree
            @test degree(g; dir=:in) == [0, 3, 0, 1]
            @test degree(g; dir=:both) == [2, 4, 1, 1]
            @test eltype(degree(g, Float32)) == Float32

            if TEST_GPU
                g_gpu = g |> gpu
                d = degree(g)
                d_gpu = degree(g_gpu)
                @test d_gpu isa CuVector{Int}
                @test Array(d_gpu) == d
            end
        end
        
        @testset "weighted" begin
            # weighted degree
            s = [1, 1, 2, 3]
            t = [2, 2, 2, 4]    
            eweight = [0.1, 2.1, 1.2, 1]
            g = GNNGraph((s, t, eweight), graph_type=GRAPH_T)
            @test degree(g) ==  [2.2, 1.2, 1.0, 0.0]
            d = degree(g, edge_weight=false)
            if GRAPH_T == :coo
                @test d ==  [2, 1, 1, 0]
                @test degree(g, edge_weight=nothing) ==  [2, 1, 1, 0]
            else
                # Adjacency matrix representation cannot disambiguate multiple edges
                # and edge weights
                @test d ==  [1, 1, 1, 0]
                @test degree(g, edge_weight=nothing) ==  [1, 1, 1, 0]
            end
            @test eltype(d) <: Integer
            if GRAPH_T == :coo
                @test degree(g, edge_weight=2*eweight) == [4.4, 2.4, 2.0, 0.0]
            end

            if TEST_GPU
                g_gpu = g |> gpu
                d = degree(g)
                d_gpu = degree(g_gpu)
                @test d_gpu isa CuVector{Float32}
                @test Array(d_gpu) ≈ d
            end
        end
    end
end
