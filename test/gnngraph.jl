@testset "GNNGraph" begin
    @testset "symmetric graph" begin
        s = [1, 1, 2, 2, 3, 3, 4, 4]
        t = [2, 4, 1, 3, 2, 4, 1, 3]
        adj_mat =  [0  1  0  1
                    1  0  1  0
                    0  1  0  1
                    1  0  1  0]
        adj_list_out =  [[2,4], [1,3], [2,4], [1,3]]
        adj_list_in =  [[2,4], [1,3], [2,4], [1,3]]

        # core functionality
        g = GNNGraph(s, t; graph_type=GRAPH_T)
        @test g.num_edges == 8
        @test g.num_nodes == 4
        @test collect(edges(g)) |> sort == collect(zip(s, t)) |> sort
        @test sort(outneighbors(g, 1)) == [2, 4] 
        @test sort(inneighbors(g, 1)) == [2, 4] 
        @test is_directed(g) == true
        s1, t1 = GraphNeuralNetworks.sort_edge_index(edge_index(g))
        @test s1 == s
        @test t1 == t
        
        # adjacency
        @test adjacency_matrix(g) == adj_mat
        @test adjacency_matrix(g; dir=:in) == adj_mat
        @test adjacency_matrix(g; dir=:out) == adj_mat
        @test sort.(adjacency_list(g; dir=:in)) == adj_list_in
        @test sort.(adjacency_list(g; dir=:out)) == adj_list_out

        @testset "constructors" begin
            g = GNNGraph(adj_mat; graph_type=GRAPH_T)
            adjacency_matrix(g; dir=:out) == adj_mat
            adjacency_matrix(g; dir=:in) == adj_mat
        end 

        @testset "degree" begin
            g = GNNGraph(adj_mat; graph_type=GRAPH_T)
            @test degree(g, dir=:out) == vec(sum(adj_mat, dims=2))
            @test degree(g, dir=:in) == vec(sum(adj_mat, dims=1))
        end
    end

    @testset "asymmetric graph" begin
        s = [1, 2, 3, 4]
        t = [2, 3, 4, 1]
        adj_mat_out =  [0  1  0  0
                        0  0  1  0
                        0  0  0  1
                        1  0  0  0]
        adj_list_out =  [[2], [3], [4], [1]]


        adj_mat_in =   [0  0  0  1
                        1  0  0  0
                        0  1  0  0
                        0  0  1  0]
        adj_list_in =  [[4], [1], [2], [3]]

        # core functionality
        g = GNNGraph(s, t; graph_type=GRAPH_T)
        @test g.num_edges == 4
        @test g.num_nodes == 4
        @test collect(edges(g)) |> sort == collect(zip(s, t)) |> sort
        @test sort(outneighbors(g, 1)) == [2] 
        @test sort(inneighbors(g, 1)) == [4] 
        @test is_directed(g) == true
        s1, t1 = GraphNeuralNetworks.sort_edge_index(edge_index(g))
        @test s1 == s
        @test t1 == t

        # adjacency
        @test adjacency_matrix(g) ==  adj_mat_out
        @test adjacency_list(g) ==  adj_list_out
        @test adjacency_matrix(g, dir=:out) ==  adj_mat_out
        @test adjacency_list(g, dir=:out) ==  adj_list_out
        @test adjacency_matrix(g, dir=:in) ==  adj_mat_in
        @test adjacency_list(g, dir=:in) ==  adj_list_in

        @testset "degree" begin
            g = GNNGraph(adj_mat_out; graph_type=GRAPH_T)
            @test degree(g, dir=:out) == vec(sum(adj_mat_out, dims=2))
            @test degree(g, dir=:in) == vec(sum(adj_mat_out, dims=1))
        end
    end

    @testset "LightGraphs constructor" begin
        lg = random_regular_graph(10, 4)
        @test !LightGraphs.is_directed(lg)
        g = GNNGraph(lg)
        @test g.num_edges == 2*ne(lg) # g in undirected
        @test LightGraphs.is_directed(g)
        for e in LightGraphs.edges(lg)
            i, j = src(e), dst(e)
            @test has_edge(g, i, j)
            @test has_edge(g, j, i)            
        end
    end

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

    @testset "getgraph"  begin
        #TODO add graph_type=GRAPH_T
        g1 = GNNGraph(random_regular_graph(10,2), ndata=rand(16,10))
        g2 = GNNGraph(random_regular_graph(4,2), ndata=rand(16,4))
        g3 = GNNGraph(random_regular_graph(7,2), ndata=rand(16,7))
        g = Flux.batch([g1, g2, g3])
        g2b, nodemap = getgraph(g, 2)
        
        s, t = edge_index(g2b)
        @test s == edge_index(g2)[1]
        @test t == edge_index(g2)[2] 
        @test node_features(g2b) ≈ node_features(g2) 
    end

    @testset "Features" begin
        g = GNNGraph(sprand(10, 10, 0.3), graph_type=GRAPH_T)
        
        # default names
        X = rand(10, g.num_nodes)
        E = rand(10, g.num_edges)
        U = rand(10, g.num_graphs)
        
        g = GNNGraph(g, ndata=X, edata=E, gdata=U)
        @test g.ndata.x === X
        @test g.edata.e === E
        @test g.gdata.u === U

        # Check no args
        g = GNNGraph(g)
        @test g.ndata.x === X
        @test g.edata.e === E
        @test g.gdata.u === U


        # multiple features names
        g = GNNGraph(g, ndata=(x2=2X, g.ndata...), edata=(e2=2E, g.edata...), gdata=(u2=2U, g.gdata...))
        @test g.ndata.x === X
        @test g.edata.e === E
        @test g.gdata.u === U
        @test g.ndata.x2 ≈ 2X
        @test g.edata.e2 ≈ 2E
        @test g.gdata.u2 ≈ 2U
    end 

    @testset "LearnBase and DataLoader compat" begin
        n, m, num_graphs = 10, 30, 50
        X = rand(10, n)
        E = rand(10, 2m)
        U = rand(10, 1)
        g = Flux.batch([GNNGraph(erdos_renyi(n, m), ndata=X, edata=E, gdata=U) 
                        for _ in 1:num_graphs])
        
        @test LearnBase.getobs(g, 3) == getgraph(g, 3)[1]
        @test LearnBase.getobs(g, 3:5) == getgraph(g, 3:5)[1]
        @test LearnBase.nobs(g) == g.num_graphs
        
        d = Flux.Data.DataLoader(g, batchsize = 2, shuffle=false)
        @test first(d) == getgraph(g, 1:2)[1]
    end
end
