@testitem "msgpass" setup=[SharedTestSetup] begin
    #TODO test all graph types
    GRAPH_T = :coo
    in_channel = 10
    out_channel = 5
    num_V = 6
    num_E = 14
    T = Float32

    adj = [0 1 0 0 0 0
            1 0 0 1 1 1
            0 0 0 0 0 1
            0 1 0 0 1 0
            0 1 0 1 0 1
            0 1 1 0 1 0]

    X = rand(T, in_channel, num_V)
    E = rand(T, in_channel, num_E)

    g = GNNGraph(adj, graph_type = GRAPH_T)

    @testset "propagate" begin
        function message(xi, xj, e)
            @test xi === nothing
            @test e === nothing
            ones(T, out_channel, size(xj, 2))
        end

        m = propagate(message, g, +, xj = X)

        @test size(m) == (out_channel, num_V)

        @testset "isolated nodes" begin
            x1 = rand(1, 6)
            g1 = GNNGraph(collect(1:5), collect(1:5), num_nodes = 6)
            y1 = propagate((xi, xj, e) -> xj, g, +, xj = x1)
            @test size(y1) == (1, 6)
        end
    end

    @testset "apply_edges" begin
        m = apply_edges(g, e = E) do xi, xj, e
            @test xi === nothing
            @test xj === nothing
            ones(out_channel, size(e, 2))
        end

        @test m == ones(out_channel, num_E)

        # With NamedTuple input
        m = apply_edges(g, xj = (; a = X, b = 2X), e = E) do xi, xj, e
            @test xi === nothing
            @test xj.b == 2 * xj.a
            @test size(xj.a, 2) == size(xj.b, 2) == size(e, 2)
            ones(out_channel, size(e, 2))
        end

        # NamedTuple output
        m = apply_edges(g, e = E) do xi, xj, e
            @test xi === nothing
            @test xj === nothing
            (; a = ones(out_channel, size(e, 2)))
        end

        @test m.a == ones(out_channel, num_E)

        @testset "sizecheck" begin
            x = rand(3, g.num_nodes - 1)
            @test_throws AssertionError apply_edges(copy_xj, g, xj = x)
            @test_throws AssertionError apply_edges(copy_xj, g, xi = x)

            x = (a = rand(3, g.num_nodes), b = rand(3, g.num_nodes + 1))
            @test_throws AssertionError apply_edges(copy_xj, g, xj = x)
            @test_throws AssertionError apply_edges(copy_xj, g, xi = x)

            e = rand(3, g.num_edges - 1)
            @test_throws AssertionError apply_edges(copy_xj, g, e = e)
        end
    end

    @testset "copy_xj" begin
        n = 128
        A = sprand(n, n, 0.1)
        Adj = map(x -> x > 0 ? 1 : 0, A)
        X = rand(10, n)

        g = GNNGraph(A, ndata = X, graph_type = GRAPH_T)

        function spmm_copyxj_fused(g)
            propagate(copy_xj,
                        g, +; xj = g.ndata.x)
        end

        function spmm_copyxj_unfused(g)
            propagate((xi, xj, e) -> xj,
                        g, +; xj = g.ndata.x)
        end

        @test spmm_copyxj_unfused(g) ≈ X * Adj
        @test spmm_copyxj_fused(g) ≈ X * Adj
    end

    @testset "e_mul_xj and w_mul_xj for weighted conv" begin
        n = 128
        A = sprand(n, n, 0.1)
        Adj = map(x -> x > 0 ? 1 : 0, A)
        X = rand(10, n)

        g = GNNGraph(A, ndata = X, edata = A.nzval, graph_type = GRAPH_T)

        function spmm_unfused(g)
            propagate((xi, xj, e) -> reshape(e, 1, :) .* xj,
                        g, +; xj = g.ndata.x, e = g.edata.e)
        end
        function spmm_fused(g)
            propagate(e_mul_xj,
                        g, +; xj = g.ndata.x, e = g.edata.e)
        end

        function spmm_fused2(g)
            propagate(w_mul_xj,
                        g, +; xj = g.ndata.x)
        end

        @test spmm_unfused(g) ≈ X * A
        @test spmm_fused(g) ≈ X * A
        @test spmm_fused2(g) ≈ X * A
    end

    @testset "aggregate_neighbors" begin
        @testset "sizecheck" begin
            m = rand(2, g.num_edges - 1)
            @test_throws AssertionError aggregate_neighbors(g, +, m)

            m = (a = rand(2, g.num_edges + 1), b = nothing)
            @test_throws AssertionError aggregate_neighbors(g, +, m)
        end
    end

end