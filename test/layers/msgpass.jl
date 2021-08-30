@testset "MessagePassing" begin 
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

    struct NewLayer{G} <: MessagePassing end

    X = rand(T, in_channel, num_V)
    E = rand(T, in_channel, num_E)
    u = rand(T, in_channel)


    @testset "no aggregation" begin
        l = NewLayer{GRAPH_T}()
        (l::NewLayer{GRAPH_T})(fg) = GraphNeuralNetworks.propagate(l, fg, nothing)

        fg = FeaturedGraph(adj, nf=X, graph_type=GRAPH_T)
        fg_ = l(fg)

        @test adjacency_matrix(fg_) == adj
        @test node_feature(fg_) === nothing
        @test edge_feature(fg_)  === nothing
        @test global_feature(fg_) === nothing
    end

    @testset "neighbor aggregation (+)" begin
        l = NewLayer{GRAPH_T}()
        (l::NewLayer{GRAPH_T})(fg) = GraphNeuralNetworks.propagate(l, fg, +)

        fg = FeaturedGraph(adj, nf=X, ef=E, gf=u, graph_type=GRAPH_T)
        fg_ = l(fg)

        @test adjacency_matrix(fg_) == adj
        @test size(node_feature(fg_)) == (in_channel, num_V)
        @test edge_feature(fg_) ≈ E
        @test global_feature(fg_) ≈ u
    end

    GraphNeuralNetworks.message(l::NewLayer{GRAPH_T}, xi, xj, e, u) = ones(T, out_channel, size(e,2))

    @testset "custom message and neighbor aggregation" begin
        l = NewLayer{GRAPH_T}()
        (l::NewLayer{GRAPH_T})(fg) = GraphNeuralNetworks.propagate(l, fg, +)

        fg = FeaturedGraph(adj, nf=X, ef=E, gf=u, graph_type=GRAPH_T)
        fg_ = l(fg)

        @test adjacency_matrix(fg_) == adj
        @test size(node_feature(fg_)) == (out_channel, num_V)
        @test edge_feature(fg_) ≈ edge_feature(fg)
        @test global_feature(fg_) ≈ global_feature(fg)
    end

    GraphNeuralNetworks.update_edge(l::NewLayer{GRAPH_T}, m, e) = m

    @testset "update_edge" begin
        l = NewLayer{GRAPH_T}()
        (l::NewLayer{GRAPH_T})(fg) = GraphNeuralNetworks.propagate(l, fg, +)

        fg = FeaturedGraph(adj, nf=X, ef=E, gf=u, graph_type=GRAPH_T)
        fg_ = l(fg)

        @test adjacency_matrix(fg_) == adj
        @test size(node_feature(fg_)) == (out_channel, num_V)
        @test size(edge_feature(fg_)) == (out_channel, num_E)
        @test global_feature(fg_) ≈ global_feature(fg)
    end

    GraphNeuralNetworks.update(l::NewLayer{GRAPH_T}, m̄, xi, u) = rand(T, 2*out_channel, size(xi, 2))

    @testset "update edge/vertex" begin
        l = NewLayer{GRAPH_T}()
        (l::NewLayer{GRAPH_T})(fg) = GraphNeuralNetworks.propagate(l, fg, +)

        fg = FeaturedGraph(adj, nf=X, ef=E, gf=u, graph_type=GRAPH_T)
        fg_ = l(fg)

        @test all(adjacency_matrix(fg_) .== adj)
        @test size(node_feature(fg_)) == (2*out_channel, num_V)
        @test size(edge_feature(fg_)) == (out_channel, num_E)
        @test size(global_feature(fg_)) == (in_channel,)
    end

    struct NewLayerW{G} <: MessagePassing
        weight
    end

    NewLayerW(in, out) = NewLayerW{GRAPH_T}(randn(T, out, in))

    GraphNeuralNetworks.message(l::NewLayerW{GRAPH_T}, x_i, x_j, e_ij) = l.weight * x_j
    GraphNeuralNetworks.update(l::NewLayerW{GRAPH_T}, m, x) = l.weight * x + m

    @testset "message and update with weights" begin
        l = NewLayerW(in_channel, out_channel)
        (l::NewLayerW{GRAPH_T})(fg) = GraphNeuralNetworks.propagate(l, fg, +)

        fg = FeaturedGraph(adj, nf=X, ef=E, gf=u, graph_type=GRAPH_T)
        fg_ = l(fg)

        @test adjacency_matrix(fg_) == adj
        @test size(node_feature(fg_)) == (out_channel, num_V)
        @test edge_feature(fg_) === E
        @test global_feature(fg_) === u
    end
end
