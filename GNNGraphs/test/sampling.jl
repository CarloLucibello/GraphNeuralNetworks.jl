if GRAPH_T == :coo
    @testset "sample_neighbors" begin
        # replace = false
        dir = :in
        nodes = 2:3
        g = rand_graph(10, 40, bidirected = false, graph_type = GRAPH_T)
        sg = sample_neighbors(g, nodes; dir)
        @test sg.num_nodes == 10
        @test sg.num_edges == sum(degree(g, i; dir) for i in nodes)
        @test size(sg.edata.EID) == (sg.num_edges,)
        @test length(union(sg.edata.EID)) == length(sg.edata.EID)
        adjlist = adjacency_list(g; dir)
        s, t = edge_index(sg)
        @test all(t .∈ Ref(nodes))
        for i in nodes
            @test sort(neighbors(sg, i; dir)) == sort(neighbors(g, i; dir))
        end

        # replace = true
        dir = :out
        nodes = 2:3
        K = 2
        g = rand_graph(10, 40, bidirected = false, graph_type = GRAPH_T)
        sg = sample_neighbors(g, nodes, K; dir, replace = true)
        @test sg.num_nodes == 10
        @test sg.num_edges == sum(K for i in nodes)
        @test size(sg.edata.EID) == (sg.num_edges,)
        adjlist = adjacency_list(g; dir)
        s, t = edge_index(sg)
        @test all(s .∈ Ref(nodes))
        for i in nodes
            @test issubset(neighbors(sg, i; dir), adjlist[i])
        end

        # dropnodes = true
        dir = :in
        nodes = 2:3
        g = rand_graph(10, 40, bidirected = false, graph_type = GRAPH_T)
        g = GNNGraph(g, ndata = (x1 = rand(10),), edata = (e1 = rand(40),))
        sg = sample_neighbors(g, nodes; dir, dropnodes = true)
        @test sg.num_edges == sum(degree(g, i; dir) for i in nodes)
        @test size(sg.edata.EID) == (sg.num_edges,)
        @test size(sg.ndata.NID) == (sg.num_nodes,)
        @test sg.edata.e1 == g.edata.e1[sg.edata.EID]
        @test sg.ndata.x1 == g.ndata.x1[sg.ndata.NID]
        @test length(union(sg.ndata.NID)) == length(sg.ndata.NID)
    end

    @testset "induced_subgraph" begin
        s = [1, 2]
        t = [2, 3]
        
        graph = GNNGraph((s, t), ndata = (; x=rand(Float32, 32, 3), y=rand(Float32, 3)), edata = rand(Float32, 2))
        
        nodes = [1, 2, 3]
        subgraph = Graphs.induced_subgraph(graph, nodes)
        
        @test subgraph.num_nodes == 3  
        @test subgraph.num_edges == 2  
        @test subgraph.ndata.x == graph.ndata.x
        @test subgraph.ndata.y == graph.ndata.y
        @test subgraph.edata == graph.edata
        
        nodes = [1, 2]
        subgraph = Graphs.induced_subgraph(graph, nodes)

        @test subgraph.num_nodes == 2 
        @test subgraph.num_edges == 1 
        @test subgraph.ndata == getobs(graph.ndata, [1, 2])
        @test isapprox(getobs(subgraph.edata.e, 1), getobs(graph.edata.e, 1); atol=1e-6)

        graph = GNNGraph(2)
        graph = add_edges(graph, ([2], [1]))
        nodes = [1]
        subgraph = Graphs.induced_subgraph(graph, nodes)
        
        @test subgraph.num_nodes == 1 
        @test subgraph.num_edges == 0 
    end
end