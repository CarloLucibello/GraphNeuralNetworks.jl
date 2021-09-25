const ACUMatrix{T} = Union{CuMatrix{T}, CUDA.CUSPARSE.CuSparseMatrix{T}}

@testset "cuda/gnngraph" begin
    s = [1,1,2,3,4]
    t = [2,5,3,5,5]
    s, t = [s; t], [t; s]  #symmetrize
    g = GNNGraph(s, t, graph_type=GRAPH_T) 
    g_gpu = g |> gpu

    @testset "functor" begin
        s_cpu, t_cpu = edge_index(g)
        s_gpu, t_gpu = edge_index(g_gpu)
        @test s_gpu isa CuVector{Int}
        @test Array(s_gpu) == s_cpu
        @test t_gpu isa CuVector{Int}
        @test Array(t_gpu) == t_cpu
    end

    @testset "adjacency_matrix" begin
        # See https://github.com/JuliaGPU/CUDA.jl/pull/1093

        mat = adjacency_matrix(g)
        mat_gpu = adjacency_matrix(g_gpu)
        @test mat_gpu isa ACUMatrix{Int}
        @test Array(mat_gpu) == mat 
    end

    @testset "normalized_laplacian" begin
        mat = normalized_laplacian(g)
        mat_gpu = normalized_laplacian(g_gpu)
        @test mat_gpu isa ACUMatrix{Float32}
        @test Array(mat_gpu) == mat 
    end

    @teset "degree" begin
        d = degree(g)
        d_gpu = degree(g_gpu)
        @test d_gpu isa CuVector
        @test Array(d_gpu) == d
    end

    @testset "scaled_laplacian" begin
        @test_broken begin 
            mat = scaled_laplacian(g)
            mat_gpu = scaled_laplacian(g_gpu)
            @test mat_gpu isa ACUMatrix{Float32}
            @test Array(mat_gpu) == mat
        end
    end
end
