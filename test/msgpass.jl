@testset "message passing" begin 
    in_channel = 10
    out_channel = 5
    num_V = 6
    num_E = 14
    T = Float32

    adj =  [0 1 0 0 0 0
            1 0 0 1 1 1
            0 0 0 0 0 1
            0 1 0 0 1 0
            0 1 0 1 0 1
            0 1 1 0 1 0]

    
    X = rand(T, in_channel, num_V)
    E = rand(T, in_channel, num_E)
    U = rand(T, in_channel)


    @testset "no aggregation" begin
        struct NewLayer1{G} end

        (l::NewLayer1{GRAPH_T})(g) = GraphNeuralNetworks.propagate(l, g, nothing)
        
        l = NewLayer1{GRAPH_T}()
        g = GNNGraph(adj, ndata=X, graph_type=GRAPH_T)
        g_ = l(g)

        @test adjacency_matrix(g_) == adj
        @test node_features(g_) === nothing
        @test edge_features(g_)  === nothing
        @test graph_features(g_) === nothing
    end

    @testset "neighbor aggregation (+)" begin
        struct NewLayer2{G} end

        (l::NewLayer2{GRAPH_T})(g) = GraphNeuralNetworks.propagate(l, g, +)
        
        l = NewLayer2{GRAPH_T}()
        g = GNNGraph(adj, ndata=X, edata=E, gdata=U, graph_type=GRAPH_T)
        g_ = l(g)

        @test adjacency_matrix(g_) == adj
        @test size(node_features(g_)) == (in_channel, num_V)
        @test edge_features(g_) ≈ E
        @test graph_features(g_) ≈ U
    end

    @testset "custom message and neighbor aggregation" begin
        struct NewLayer3{G} end
        
        GraphNeuralNetworks.message(l::NewLayer3{GRAPH_T}, xi, xj, e) = ones(T, out_channel, size(e,2))
        (l::NewLayer3{GRAPH_T})(g) = GraphNeuralNetworks.propagate(l, g, +)

        
        l = NewLayer3{GRAPH_T}()
        g = GNNGraph(adj, ndata=X, edata=E, gdata=U, graph_type=GRAPH_T)
        g_ = l(g)

        @test adjacency_matrix(g_) == adj
        @test size(node_features(g_)) == (out_channel, num_V)
        @test edge_features(g_) ≈ edge_features(g)
        @test graph_features(g_) ≈ graph_features(g)
    end


    @testset "update_edge" begin
        struct NewLayer4{G} end

        GraphNeuralNetworks.update_edge(l::NewLayer4{GRAPH_T}, m, e) = m
        GraphNeuralNetworks.message(l::NewLayer4{GRAPH_T}, xi, xj, e) = ones(T, out_channel, size(e,2))
        (l::NewLayer4{GRAPH_T})(g) = GraphNeuralNetworks.propagate(l, g, +)
        
        l = NewLayer4{GRAPH_T}()
        g = GNNGraph(adj, ndata=X, edata=E, gdata=U, graph_type=GRAPH_T)
        g_ = l(g)

        @test adjacency_matrix(g_) == adj
        @test size(node_features(g_)) == (out_channel, num_V)
        @test size(edge_features(g_)) == (out_channel, num_E)
        @test graph_features(g_) ≈ graph_features(g)
    end

    
    @testset "update edge/vertex" begin
        struct NewLayer5{G} end

        GraphNeuralNetworks.update_node(l::NewLayer5{GRAPH_T}, m̄, xi) = rand(T, 2*out_channel, size(xi, 2))
        GraphNeuralNetworks.update_edge(l::NewLayer5{GRAPH_T}, m, e) = m
        GraphNeuralNetworks.message(l::NewLayer5{GRAPH_T}, xi, xj, e) = ones(T, out_channel, size(e,2))
        (l::NewLayer5{GRAPH_T})(g) = GraphNeuralNetworks.propagate(l, g, +)

        l = NewLayer5{GRAPH_T}()
        g = GNNGraph(adj, ndata=X, edata=E, gdata=U, graph_type=GRAPH_T)
        g_ = l(g)

        @test all(adjacency_matrix(g_) .== adj)
        @test size(node_features(g_)) == (2*out_channel, num_V)
        @test size(edge_features(g_)) == (out_channel, num_E)
        @test size(graph_features(g_)) == (in_channel,)
    end

    @testset "message and update with weights" begin

        struct NewLayerW{G}
            weight
        end

        NewLayerW(in, out) = NewLayerW{GRAPH_T}(randn(T, out, in))

        GraphNeuralNetworks.message(l::NewLayerW{GRAPH_T}, x_i, x_j, e_ij) = l.weight * x_j
        GraphNeuralNetworks.update_node(l::NewLayerW{GRAPH_T}, m, x) = l.weight * x + m

        l = NewLayerW(in_channel, out_channel)
        (l::NewLayerW{GRAPH_T})(g) = GraphNeuralNetworks.propagate(l, g, +)

        g = GNNGraph(adj, ndata=X, edata=E, gdata=U, graph_type=GRAPH_T)
        g_ = l(g)

        @test adjacency_matrix(g_) == adj
        @test size(node_features(g_)) == (out_channel, num_V)
        @test edge_features(g_) === E
        @test graph_features(g_) === U
    end
end
