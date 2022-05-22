@testset "GNNGraph" begin

    @testset "Constructor: adjacency matrix" begin
        A = sprand(10, 10, 0.5)
        sA, tA, vA = findnz(A)
        
        g = GNNGraph(A, graph_type=GRAPH_T)
        s, t = edge_index(g)
        v = get_edge_weight(g)
        @test s == sA
        @test t == tA
        @test v == vA

        g = GNNGraph(Matrix(A), graph_type=GRAPH_T)
        s, t = edge_index(g)
        v = get_edge_weight(g)
        @test s == sA
        @test t == tA
        @test v == vA

        g = GNNGraph([0 0 0
                      0 0 1
                      0 1 0], graph_type=GRAPH_T)
        @test g.num_nodes == 3
        @test g.num_edges == 2
        
        g = GNNGraph([0 1 0
                      1 0 0
                      0 0 0], graph_type=GRAPH_T)
        @test g.num_nodes == 3
        @test g.num_edges == 2
    end

    @testset "Constructor: integer" begin
        g = GNNGraph(10, graph_type=GRAPH_T)
        @test g.num_nodes == 10
        @test g.num_edges == 0

        g2 = rand_graph(10, 30, graph_type=GRAPH_T)
        G = typeof(g2)
        g = G(10)
        @test g.num_nodes == 10
        @test g.num_edges == 0
    end

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
        if TEST_GPU
            g_gpu = g |> gpu
        end

        @test g.num_edges == 8
        @test g.num_nodes == 4
        @test nv(g) == g.num_nodes
        @test ne(g) == g.num_edges
        @test Tuple.(collect(edges(g))) |> sort == collect(zip(s, t)) |> sort
        @test sort(outneighbors(g, 1)) == [2, 4] 
        @test sort(inneighbors(g, 1)) == [2, 4] 
        @test is_directed(g) == true
        s1, t1 = sort_edge_index(edge_index(g))
        @test s1 == s
        @test t1 == t
        @test vertices(g) == 1:g.num_nodes
        
        @test sort.(adjacency_list(g; dir=:in)) == adj_list_in
        @test sort.(adjacency_list(g; dir=:out)) == adj_list_out

        @testset "adjacency_matrix" begin
            @test adjacency_matrix(g) == adj_mat
            @test adjacency_matrix(g; dir=:in) == adj_mat
            @test adjacency_matrix(g; dir=:out) == adj_mat
            
            if TEST_GPU
                # See https://github.com/JuliaGPU/CUDA.jl/pull/1093
                mat_gpu = adjacency_matrix(g_gpu)
                @test mat_gpu isa ACUMatrix{Int}
                @test Array(mat_gpu) == adj_mat 
            end
        end
        
        @testset "normalized_laplacian" begin
            mat = normalized_laplacian(g)
            if TEST_GPU
                mat_gpu = normalized_laplacian(g_gpu)
                @test mat_gpu isa ACUMatrix{Float32}
                @test Array(mat_gpu) == mat 
            end
        end


        @testset "scaled_laplacian" begin
            if TEST_GPU
                @test_broken begin 
                    mat = scaled_laplacian(g)
                    mat_gpu = scaled_laplacian(g_gpu)
                    @test mat_gpu isa ACUMatrix{Float32}
                    @test Array(mat_gpu) == mat
                end
            end
        end

        @testset "constructors" begin
            adjacency_matrix(g; dir=:out) == adj_mat
            adjacency_matrix(g; dir=:in) == adj_mat
        end 

        if TEST_GPU
            @testset "functor" begin                
                s_cpu, t_cpu = edge_index(g)
                s_gpu, t_gpu = edge_index(g_gpu)
                @test s_gpu isa CuVector{Int}
                @test Array(s_gpu) == s_cpu
                @test t_gpu isa CuVector{Int}
                @test Array(t_gpu) == t_cpu
            end
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
        if TEST_GPU
            g_gpu = g |> gpu
        end

        @test g.num_edges == 4
        @test g.num_nodes == 4
        @test length(edges(g)) == 4
        @test sort(outneighbors(g, 1)) == [2] 
        @test sort(inneighbors(g, 1)) == [4] 
        @test is_directed(g) == true
        @test is_directed(typeof(g)) == true
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
    end

    @testset "zero" begin
        g = rand_graph(4, 6, graph_type=GRAPH_T)
        G = typeof(g)
        @test zero(G) == G(0)
    end

    @testset "Graphs.jl constructor" begin
        lg = random_regular_graph(10, 4)
        @test !Graphs.is_directed(lg)
        g = GNNGraph(lg)
        @test g.num_edges == 2*ne(lg) # g in undirected
        @test Graphs.is_directed(g)
        for e in Graphs.edges(lg)
            i, j = src(e), dst(e)
            @test has_edge(g, i, j)
            @test has_edge(g, j, i)
        end

        @testset "SimpleGraph{Int32}" begin
            g = GNNGraph(SimpleGraph{Int32}(6), graph_type=GRAPH_T)
            @test g.num_nodes == 6
        end
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

        # Dimension checks
        @test_throws AssertionError GNNGraph(erdos_renyi(10, 30), edata=rand(29), graph_type=GRAPH_T)
        @test_throws AssertionError GNNGraph(erdos_renyi(10, 30), edata=rand(2, 29), graph_type=GRAPH_T)
        @test_throws AssertionError GNNGraph(erdos_renyi(10, 30), edata=(; x=rand(30), y=rand(29)), graph_type=GRAPH_T)

        # Copy features on reverse edge
        e = rand(30)
        g = GNNGraph(erdos_renyi(10,  30), edata=e, graph_type=GRAPH_T)
        @test g.edata.e == [e; e]

        # non-array global 
        g = rand_graph(10,  30, gdata="ciao", graph_type=GRAPH_T)
        @test g.gdata.u == "ciao"
    
        # vectors stays vectors
        g = rand_graph(10,  30, ndata=rand(10),
                                edata=rand(30),
                                gdata=(u=rand(2), z=rand(1), q=1),
                                graph_type=GRAPH_T)
        @test size(g.ndata.x) == (10,)
        @test size(g.edata.e) == (30,)
        @test size(g.gdata.u) == (2, 1)
        @test size(g.gdata.z) == (1,)
        @test g.gdata.q === 1
        
        # Error for non-array ndata
        @test_throws AssertionError rand_graph(10,  30, ndata="ciao", graph_type=GRAPH_T)
        @test_throws AssertionError rand_graph(10,  30, ndata=1, graph_type=GRAPH_T)
    end

    @testset "MLUtils and DataLoader compat" begin
        n, m, num_graphs = 10, 30, 50
        X = rand(10, n)
        E = rand(10, m)
        U = rand(10, 1)
        data = [rand_graph(n, m, ndata=X, edata=E, gdata=U, graph_type=GRAPH_T) 
                for _ in 1:num_graphs]
        g = Flux.batch(data)

        @testset "batch then pass to dataloader" begin
            @test MLUtils.getobs(g, 3) == getgraph(g, 3)
            @test MLUtils.getobs(g, 3:5) == getgraph(g, 3:5)
            @test MLUtils.numobs(g) == g.num_graphs
            
            d = Flux.Data.DataLoader(g, batchsize=2, shuffle=false)
            @test first(d) == getgraph(g, 1:2)
        end

        @testset "pass to dataloader and collate" begin
            @test MLUtils.getobs(data, 3) == getgraph(g, 3)
            @test MLUtils.getobs(data, 3:5) == getgraph(g, 3:5)
            @test MLUtils.numobs(data) == g.num_graphs

            d = Flux.Data.DataLoader(data, batchsize=2, shuffle=false)
            @test first(d) == getgraph(g, 1:2)
        end
    end

    @testset "Graphs.jl integration" begin
        g = GNNGraph(erdos_renyi(10, 20), graph_type=GRAPH_T)
        @test g isa Graphs.AbstractGraph
    end

    @testset "==" begin
        g1 = rand_graph(5, 6, ndata=rand(5), edata=rand(6), graph_type=GRAPH_T)
        @test g1 == g1
        @test g1 == deepcopy(g1)
        @test g1 !== deepcopy(g1)
        
        g2 = GNNGraph(g1, graph_type=GRAPH_T)
        @test g1 == g2
        @test g1 === g2 # this is true since GNNGraph is immutable
        
        g2 = GNNGraph(g1, ndata=rand(5), graph_type=GRAPH_T)
        @test g1 != g2
        @test g1 !== g2
        
        g2 = GNNGraph(g1, edata=rand(6), graph_type=GRAPH_T)
        @test g1 != g2
        @test g1 !== g2
    end

    @testset "hash" begin
        g1 = rand_graph(5, 6, ndata=rand(5), edata=rand(6), graph_type=GRAPH_T)
        @test hash(g1) == hash(g1)
        @test hash(g1) == hash(deepcopy(g1))
        @test hash(g1) == hash(GNNGraph(g1, ndata=g1.ndata, graph_type=GRAPH_T))
        @test hash(g1) == hash(GNNGraph(g1, ndata=g1.ndata, graph_type=GRAPH_T))
        @test hash(g1) != hash(GNNGraph(g1, ndata=rand(5), graph_type=GRAPH_T))
        @test hash(g1) != hash(GNNGraph(g1, edata=rand(6), graph_type=GRAPH_T))
    end
end


