@testset "transform" begin
    @testset "add self-loops" begin
        A = [1  1  0  0
             0  0  1  0
             0  0  0  1
             1  0  0  0]
        A2 =   [2  1  0  0
                0  1  1  0
                0  0  1  1
                1  0  0  1]

        g = GNNGraph(A; graph_type=GRAPH_T)
        fg2 = add_self_loops(g)
        @test adjacency_matrix(g) == A
        @test g.num_edges == sum(A)
        @test adjacency_matrix(fg2) == A2
        @test fg2.num_edges == sum(A2)
    end

    @testset "batch"  begin
        #TODO add graph_type=GRAPH_T
        g1 = GNNGraph(random_regular_graph(10,2), ndata=rand(16,10))
        g2 = GNNGraph(random_regular_graph(4,2), ndata=rand(16,4))
        g3 = GNNGraph(random_regular_graph(7,2), ndata=rand(16,7))
        
        g12 = Flux.batch([g1, g2])
        g12b = blockdiag(g1, g2)
        
        g123 = Flux.batch([g1, g2, g3])
        @test g123.graph_indicator == [fill(1, 10); fill(2, 4); fill(3, 7)]

        s, t = edge_index(g123)
        @test s == [edge_index(g1)[1]; 10 .+ edge_index(g2)[1]; 14 .+ edge_index(g3)[1]] 
        @test t == [edge_index(g1)[2]; 10 .+ edge_index(g2)[2]; 14 .+ edge_index(g3)[2]] 
        @test node_features(g123)[:,11:14] ≈ node_features(g2) 

        # scalar graph features
        g1 = GNNGraph(random_regular_graph(10,2), gdata=rand())
        g2 = GNNGraph(random_regular_graph(4,2), gdata=rand())
        g3 = GNNGraph(random_regular_graph(4,2), gdata=rand())
        g123 = Flux.batch([g1, g2, g3])
        @test g123.gdata.u == [g1.gdata.u, g2.gdata.u, g3.gdata.u]
    end

    @testset "unbatch" begin
        g1 = rand_graph(10, 20)
        g2 = rand_graph(5, 10)
        g12 = Flux.batch([g1, g2])
        gs = Flux.unbatch([g1,g2])
        @test length(gs) == 2
        @test gs[1].num_nodes == 10
        @test gs[1].num_edges == 20
        @test gs[1].num_graphs == 1
        @test gs[2].num_nodes == 5
        @test gs[2].num_edges == 10
        @test gs[2].num_graphs == 1
    end

    @testset "getgraph"  begin
        g1 = GNNGraph(random_regular_graph(10,2), ndata=rand(16,10), graph_type=GRAPH_T)
        g2 = GNNGraph(random_regular_graph(4,2), ndata=rand(16,4), graph_type=GRAPH_T)
        g3 = GNNGraph(random_regular_graph(7,2), ndata=rand(16,7), graph_type=GRAPH_T)
        g = Flux.batch([g1, g2, g3])
        
        g2b, nodemap = getgraph(g, 2, nmap=true)
        s, t = edge_index(g2b)
        @test s == edge_index(g2)[1]
        @test t == edge_index(g2)[2] 
        @test node_features(g2b) ≈ node_features(g2) 

        g2c = getgraph(g, 2)
        @test g2c isa GNNGraph{typeof(g.graph)}

        g1b, nodemap = getgraph(g1, 1, nmap=true)
        @test g1b === g1
        @test nodemap == 1:g1.num_nodes
    end

    @testset "add_edges" begin
        if GRAPH_T == :coo
            s = [1,1,2,3]
            t = [2,3,4,5]
            g = GNNGraph(s, t, graph_type=GRAPH_T)
            snew = [1]
            tnew = [4]
            gnew = add_edges(g, snew, tnew)
            @test gnew.num_edges == 5
            @test sort(inneighbors(gnew, 4)) == [1, 2]

            g = GNNGraph(s, t, edata=(e1=rand(2,4), e2=rand(3,4)), graph_type=GRAPH_T)
            # @test_throws ErrorException add_edges(g, snew, tnew)
            gnew = add_edges(g, snew, tnew, edata=(e1=ones(2,1), e2=zeros(3,1)))
            @test all(gnew.edata.e1[:,5] .== 1)
            @test all(gnew.edata.e2[:,5] .== 0)           
        end
    end

    @testset "add_nodes" begin
        if GRAPH_T == :coo
            g = rand_graph(6, 4, ndata=rand(2, 6), graph_type=GRAPH_T)
            gnew = add_nodes(g, 5, ndata=ones(2, 5))
            @test gnew.num_nodes == g.num_nodes + 5
            @test gnew.num_edges == g.num_edges
            @test gnew.num_graphs == g.num_graphs
            @test all(gnew.ndata.x[:,7:11] .== 1)         
        end
    end
end