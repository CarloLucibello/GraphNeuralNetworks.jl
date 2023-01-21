@testset "generate" begin
    @testset "rand_graph" begin
        n, m = 10, 20
        m2 = m ÷ 2
        x = rand(3, n)
        e = rand(4, m2)

        g = rand_graph(n, m, ndata = x, edata = e, graph_type = GRAPH_T)
        @test g.num_nodes == n
        @test g.num_edges == m
        @test g.ndata.x === x
        if GRAPH_T == :coo
            s, t = edge_index(g)
            @test s[1:m2] == t[(m2 + 1):end]
            @test t[1:m2] == s[(m2 + 1):end]
            @test g.edata.e[:, 1:m2] == e
            @test g.edata.e[:, (m2 + 1):end] == e
        end

        g = rand_graph(n, m, bidirected = false, seed = 17, graph_type = GRAPH_T)
        @test g.num_nodes == n
        @test g.num_edges == m

        g2 = rand_graph(n, m, bidirected = false, seed = 17, graph_type = GRAPH_T)
        @test edge_index(g2) == edge_index(g)
    end

    @testset "knn_graph" begin
        n, k = 10, 3
        x = rand(3, n)
        g = knn_graph(x, k; graph_type = GRAPH_T)
        @test g.num_nodes == 10
        @test g.num_edges == n * k
        @test degree(g, dir = :in) == fill(k, n)
        @test has_self_loops(g) == false

        g = knn_graph(x, k; dir = :out, self_loops = true, graph_type = GRAPH_T)
        @test g.num_nodes == 10
        @test g.num_edges == n * k
        @test degree(g, dir = :out) == fill(k, n)
        @test has_self_loops(g) == true

        graph_indicator = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        g = knn_graph(x, k; graph_indicator, graph_type = GRAPH_T)
        @test g.num_graphs == 2
        s, t = edge_index(g)
        ne = n * k ÷ 2
        @test all(1 .<= s[1:ne] .<= 5)
        @test all(1 .<= t[1:ne] .<= 5)
        @test all(6 .<= s[(ne + 1):end] .<= 10)
        @test all(6 .<= t[(ne + 1):end] .<= 10)
    end

    @testset "radius_graph" begin
        n, r = 10, 0.5
        x = rand(3, n)
        g = radius_graph(x, r; graph_type = GRAPH_T)
        @test g.num_nodes == 10
        @test has_self_loops(g) == false

        g = radius_graph(x, r; dir = :out, self_loops = true, graph_type = GRAPH_T)
        @test g.num_nodes == 10
        @test has_self_loops(g) == true

        graph_indicator = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        g = radius_graph(x, r; graph_indicator, graph_type = GRAPH_T)
        @test g.num_graphs == 2
        s, t = edge_index(g)
        @test (s .> 5) == (t .> 5)
    end
end
