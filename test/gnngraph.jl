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
        s1, t1 = sort_edge_index(edge_index(g))
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
        s1, t1 = sort_edge_index(edge_index(g))
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
        g1 = GNNGraph(random_regular_graph(10,2), nf=rand(16,10))
        g2 = GNNGraph(random_regular_graph(4,2), nf=rand(16,4))
        g3 = GNNGraph(random_regular_graph(7,2), nf=rand(16,7))
        
        g12 = Flux.batch([g1, g2])
        g12b = blockdiag(g1, g2)
        
        g123 = Flux.batch([g1, g2, g3])
        @test g123.graph_indicator == [fill(1, 10); fill(2, 4); fill(3, 7)]
    end
end
