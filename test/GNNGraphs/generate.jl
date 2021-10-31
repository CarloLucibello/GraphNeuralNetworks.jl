@testset "generate" begin 
    @testset "rand_graph" begin
        n, m = 10, 20
        x = rand(3, n)
        e = rand(4, m)
        g = rand_graph(n, m, ndata=x, edata=e, graph_type=GRAPH_T)
        @test g.num_nodes == n
        @test g.num_edges == 2m
        @test g.ndata.x === x 
        if GRAPH_T == :coo
            s, t = edge_index(g)
            @test s[1:m] == t[m+1:end]
            @test t[1:m] == s[m+1:end]
            @test g.edata.e[:,1:m] == e
            @test g.edata.e[:,m+1:end] == e
        end
        g = rand_graph(n, m, directed=true, graph_type=GRAPH_T)
        @test g.num_nodes == n
        @test g.num_edges == m
    end
end
