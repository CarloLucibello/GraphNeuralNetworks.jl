@testset "generate" begin 
    @testset "rand_graph" begin
        n, m = 10, 20
        m2 = m ÷ 2
        x = rand(3, n)
        e = rand(4, m2)
        
        g = rand_graph(n, m, ndata=x, edata=e, graph_type=GRAPH_T)
        @test g.num_nodes == n
        @test g.num_edges == m
        @test g.ndata.x === x 
        if GRAPH_T == :coo
            s, t = edge_index(g)
            @test s[1:m2] == t[m2+1:end]
            @test t[1:m2] == s[m2+1:end]
            @test g.edata.e[:,1:m2] == e
            @test g.edata.e[:,m2+1:end] == e
        end
        
        g = rand_graph(n, m, bidirected=false, seed=17, graph_type=GRAPH_T)
        @test g.num_nodes == n
        @test g.num_edges == m

        g2 = rand_graph(n, m, bidirected=false, seed=17, graph_type=GRAPH_T)
        @test edge_index(g2) == edge_index(g)
    end

    @testset "knn_graph" begin
        n = 10
        k = 3
        x = rand(3, n)
        g = knn_graph(x, k; graph_type=GRAPH_T)
        @test g.num_nodes == 10
        @test g.num_edges == n*k
        @test degree(g, dir=:in) == fill(k, n)        
        @test has_self_loops(g) == false 

        g = knn_graph(x, k; dir=:out, self_loops=true, graph_type=GRAPH_T)
        @test g.num_nodes == 10
        @test g.num_edges == n*k
        @test degree(g, dir=:out) == fill(k, n)        
        @test has_self_loops(g) == true 
    end
end
