@testset "HeteroGraphConv" begin
    d, n = 3, 5
    g = rand_bipartite_heterograph(n, 2*n, 15)
    hg = rand_bipartite_heterograph((2,3), 6)

    model = HeteroGraphConv([(:A,:to,:B) => GraphConv(d => d), 
                            (:B,:to,:A) => GraphConv(d => d)])

    for x in [
                (A = rand(Float32, d, n), B = rand(Float32, d, 2n)),
                Dict(:A => rand(Float32, d, n), :B => rand(Float32, d, 2n)) 
             ]
        # x = (A = rand(Float32, d, n), B = rand(Float32, d, 2n))
        x = Dict(:A => rand(Float32, d, n), :B => rand(Float32, d, 2n)) 
      
        y = model(g, x)

        grad, dx = gradient((model, x) -> sum(model(g, x)[1]) + sum(model(g, x)[2].^2), model, x)
        ngrad, ndx = ngradient((model, x) -> sum(model(g, x)[1]) + sum(model(g, x)[2].^2), model, x)

        @test grad.layers[1].weight1 ≈ ngrad.layers[1].weight1  rtol=1e-4
        @test grad.layers[1].weight2 ≈ ngrad.layers[1].weight2  rtol=1e-4
        @test grad.layers[1].bias ≈ ngrad.layers[1].bias        rtol=1e-4
        @test grad.layers[2].weight1 ≈ ngrad.layers[2].weight1  rtol=1e-4
        @test grad.layers[2].weight2 ≈ ngrad.layers[2].weight2  rtol=1e-4
        @test grad.layers[2].bias ≈ ngrad.layers[2].bias        rtol=1e-4

        @test dx[:A] ≈ ndx[:A]        rtol=1e-4
        @test dx[:B] ≈ ndx[:B]        rtol=1e-4
    end

    @testset "Constructor from pairs" begin
        layer = HeteroGraphConv((:A, :to, :B) => GraphConv(64 => 32, relu),
                                (:B, :to, :A) => GraphConv(64 => 32, relu));
        @test length(layer.etypes) == 2
    end

    @testset "Destination node aggregation" begin
        # deterministic setup to validate the aggregation
        d, n = 3, 5
        g = GNNHeteroGraph(((:A, :to, :B) => ([1, 1, 2, 3], [1, 2, 2, 3]),
                (:B, :to, :A) => ([1, 1, 2, 3], [1, 2, 2, 3]),
                (:C, :to, :A) => ([1, 1, 2, 3], [1, 2, 2, 3])); num_nodes = Dict(:A => n, :B => n, :C => n))
        model = HeteroGraphConv([
                (:A, :to, :B) => GraphConv(d => d, init = ones, bias = false),
                (:B, :to, :A) => GraphConv(d => d, init = ones, bias = false),
                (:C, :to, :A) => GraphConv(d => d, init = ones, bias = false)]; aggr = +)
        x = (A = rand(Float32, d, n), B = rand(Float32, d, n), C = rand(Float32, d, n))
        y = model(g, x)
        weights = ones(Float32, d, d)

        ### Test default summation aggregation
        # B2 has 2 edges from A and itself (sense check)
        expected = sum(weights * x.A[:, [1, 2]]; dims = 2) .+ weights * x.B[:, [2]]
        output = y.B[:, [2]]
        @test expected ≈ output

        # B5 has only itself
        @test weights * x.B[:, [5]] ≈ y.B[:, [5]]

        # A1 has 1 edge from B, 1 from C and twice itself
        expected = sum(weights * x.B[:, [1]] + weights * x.C[:, [1]]; dims = 2) .+
                   2 * weights * x.A[:, [1]]
        output = y.A[:, [1]]
        @test expected ≈ output

        # A2 has 2 edges from B, 2 from C and twice itself
        expected = sum(weights * x.B[:, [1, 2]] + weights * x.C[:, [1, 2]]; dims = 2) .+
                   2 * weights * x.A[:, [2]]
        output = y.A[:, [2]]
        @test expected ≈ output

        # A5 has only itself but twice
        @test 2 * weights * x.A[:, [5]] ≈ y.A[:, [5]]

        #### Test different aggregation function
        model2 = HeteroGraphConv([
                (:A, :to, :B) => GraphConv(d => d, init = ones, bias = false),
                (:B, :to, :A) => GraphConv(d => d, init = ones, bias = false),
                (:C, :to, :A) => GraphConv(d => d, init = ones, bias = false)]; aggr = -)
        y2 = model2(g, x)
        # B no change
        @test y.B ≈ y2.B

        # A1 has 1 edge from B, 1 from C, itself cancels out
        expected = sum(weights * x.B[:, [1]] - weights * x.C[:, [1]]; dims = 2)
        output = y2.A[:, [1]]
        @test expected ≈ output

        # A2 has 2 edges from B, 2 from C, itself cancels out
        expected = sum(weights * x.B[:, [1, 2]] - weights * x.C[:, [1, 2]]; dims = 2)
        output = y2.A[:, [2]]
        @test expected ≈ output
    end

    @testset "CGConv" begin
        x = (A = rand(Float32, 4,2), B = rand(Float32, 4, 3))
        layers = HeteroGraphConv( (:A, :to, :B) => CGConv(4 => 2, relu),
                                    (:B, :to, :A) => CGConv(4 => 2, relu));
        y = layers(hg, x); 
        @test size(y.A) == (2,2) && size(y.B) == (2,3)
    end

    @testset "EdgeConv" begin
        x = (A = rand(Float32, 4,2), B = rand(Float32, 4, 3))
        layers = HeteroGraphConv( (:A, :to, :B) => EdgeConv(Dense(2 * 4, 2), aggr = +),
                                    (:B, :to, :A) => EdgeConv(Dense(2 * 4, 2), aggr = +));
        y = layers(hg, x); 
        @test size(y.A) == (2,2) && size(y.B) == (2,3)
    end
  
    @testset "SGConv" begin
        x = (A = rand(Float32, 4, 2), B = rand(Float32, 4, 3))
        layers = HeteroGraphConv((:A, :to, :B) => SGConv(4 => 2),
                                 (:B, :to, :A) => SGConv(4 => 2));
        y = layers(hg, x); 
        @test size(y.A) == (2, 2) && size(y.B) == (2, 3)
    end
  
    @testset "SAGEConv" begin
        x = (A = rand(Float32, 4, 2), B = rand(Float32, 4, 3))
        layers = HeteroGraphConv((:A, :to, :B) => SAGEConv(4 => 2, relu, bias = false, aggr = +),
                                 (:B, :to, :A) => SAGEConv(4 => 2, relu, bias = false, aggr = +));
        y = layers(hg, x); 
        @test size(y.A) == (2, 2) && size(y.B) == (2, 3)
    end

    @testset "EGNNConv with Heterogeneous Graphs" begin
        # Tests are work in progress
        hin_A, hout_A, hidden_A = 5, 5, 10
        hin_B, hout_B, hidden_B = 3, 3, 6
        num_nodes_A, num_nodes_B = 5, 3  

        hg = rand_bipartite_heterograph((num_nodes_A, num_nodes_B), 15) 

        layers = HeteroGraphConv([
            (:A, :to, :B) => EGNNConv((hin_A, 0) => hout_B; hidden_size = hidden_A, residual = false),
            (:B, :to, :A) => EGNNConv((hin_B, 0) => hout_A; hidden_size = hidden_B, residual = false)
        ])

        T = Float32  
        h = (A = randn(T, hin_A, num_nodes_A), B = randn(T, hin_B, num_nodes_B))
        x = (A = rand(T, 3, num_nodes_A), B = rand(T, 3, num_nodes_B)) 

        y = layers(hg, x, h)

        @test size(y[:A].h) == (hout_A, num_nodes_A)
        @test size(y[:B].h) == (hout_B, num_nodes_B)
        @test size(y[:A].x) == (3, num_nodes_A) 
        @test size(y[:B].x) == (3, num_nodes_B)
    end

    @testset "GINConv" begin
        x = (A = rand(4, 2), B = rand(4, 3))
        layers = HeteroGraphConv((:A, :to, :B) => GINConv(Dense(4, 2), 0.4),
                                    (:B, :to, :A) => GINConv(Dense(4, 2), 0.4));
        y = layers(hg, x); 
        @test size(y.A) == (2, 2) && size(y.B) == (2, 3)
    end

    @testset "ResGatedGraphConv" begin   
        x = (A = rand(Float32, 4, 2), B = rand(Float32, 4, 3))
        layers = HeteroGraphConv((:A, :to, :B) => ResGatedGraphConv(4 => 2),
                                 (:B, :to, :A) => ResGatedGraphConv(4 => 2));
        y = layers(hg, x); 
        @test size(y.A) == (2, 2) && size(y.B) == (2, 3)
    end
end
