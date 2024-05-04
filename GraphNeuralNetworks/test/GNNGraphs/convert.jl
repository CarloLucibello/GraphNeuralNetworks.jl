if TEST_GPU
    @testset "to_coo(dense) on gpu" begin
        get_st(A) = GNNGraphs.to_coo(A)[1][1:2]
        get_val(A) = GNNGraphs.to_coo(A)[1][3]

        A = cu([0 2 2; 2.0 0 2; 2 2 0])

        y = get_val(A)
        @test y isa CuVector{Float32}
        @test Array(y) ≈ [2, 2, 2, 2, 2, 2]

        s, t = get_st(A)
        @test s isa CuVector{<:Integer}
        @test t isa CuVector{<:Integer}
        @test Array(s) == [2, 3, 1, 3, 1, 2]
        @test Array(t) == [1, 1, 2, 2, 3, 3]

        @test gradient(A -> sum(get_val(A)), A)[1] isa CuMatrix{Float32}
    end
end
