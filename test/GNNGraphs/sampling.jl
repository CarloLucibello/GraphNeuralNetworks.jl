@testset "sampling.jl" begin
    @testset "sample_neighbors" begin
        dir = :in
        nodes = 2:3
        g = rand_graph(10, 40, bidirected=false, graph_type=GRAPH_T)
        sg = sample_neighbors(g, nodes; dir)
        @test sg.num_nodes == 10
        @test sg.num_edges == sum(degree(g, i; dir) for i in nodes)
        adjlist = adjacency_list(g; dir)
        s, t = edge_index(sg)
        @test all(t .∈ Ref(nodes))
        for i in nodes
            @test sort(neighbors(sg, i; dir)) == sort(neighbors(g, i; dir))
        end

        dir = :out
        nodes = 2:3
        K = 2
        g = rand_graph(10, 40, bidirected=false, graph_type=GRAPH_T)
        sg = sample_neighbors(g, nodes, K; dir, replace=true)
        @test sg.num_nodes == 10
        @test sg.num_edges == sum(K for i in nodes)
        adjlist = adjacency_list(g; dir)
        s, t = edge_index(sg)
        @test all(s .∈ Ref(nodes))
        for i in nodes
            @test issubset(neighbors(sg, i; dir), adjlist[i])
        end
    end
end