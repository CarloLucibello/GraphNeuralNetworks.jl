@testitem "layers/conv" setup=[SharedTestSetup] begin
    rng = StableRNG(1234)
    g = rand_graph(10, 40, seed=1234)
    in_dims = 3
    out_dims = 5
    x = randn(rng, Float32, in_dims, 10)

    @testset "MEGNetConv" begin
        in_dims = 6
        out_dims = 8
        
        l = MEGNetConv(in_dims => out_dims)
        
        ps = LuxCore.initialparameters(rng, l)
        st = LuxCore.initialstates(rng, l)
        
        (x_new, e_new), st_new = l(g, x, ps, st)
        
        @test size(x_new) == (out_dims, g.num_nodes)
        @test size(e_new) == (out_dims, g.num_edges)
    end
end
