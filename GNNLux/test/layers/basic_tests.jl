@testitem "layers/basic" setup=[SharedTestSetup] begin
    rng = StableRNG(17)
    g = rand_graph(10, 40, seed=17)
    x = randn(rng, Float32, 3, 10)        

    @testset "GNNLayer" begin
        @test GNNLayer <: LuxCore.AbstractExplicitLayer
    end

    @testset "GNNChain" begin
        @test GNNChain <: LuxCore.AbstractExplicitContainerLayer{(:layers,)}
        @test GNNChain <: GNNContainerLayer
        c = GNNChain(GraphConv(3 => 5, relu), GCNConv(5 => 3))
        ps = LuxCore.initialparameters(rng, c)
        st = LuxCore.initialstates(rng, c)
        @test LuxCore.parameterlength(c) == LuxCore.parameterlength(ps)
        @test LuxCore.statelength(c) == LuxCore.statelength(st)
        y, st′ = c(g, x, ps, st)
        @test LuxCore.outputsize(c) == (3,)
        @test size(y) == (3, 10)
        loss = (x, ps) -> sum(first(c(g, x, ps, st)))
        @eval @test_gradients $loss $x $ps atol=1.0f-3 rtol=1.0f-3 skip_tracker=true skip_reverse_diff=true
    end
end
