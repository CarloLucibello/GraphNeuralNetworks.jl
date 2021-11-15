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
end
