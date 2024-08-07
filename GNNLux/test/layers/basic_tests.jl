@testitem "layers/basic" setup=[SharedTestSetup] begin
    rng = StableRNG(17)
    g = rand_graph(rng, 10, 40)
    x = randn(rng, Float32, 3, 10)        

    @testset "GNNLayer" begin
        @test GNNLayer <: LuxCore.AbstractExplicitLayer
    end

    @testset "GNNContainerLayer" begin
        @test GNNContainerLayer <: LuxCore.AbstractExplicitContainerLayer
    end

    @testset "GNNChain" begin
        @test GNNChain <: LuxCore.AbstractExplicitContainerLayer{(:layers,)}
        c = GNNChain(GraphConv(3 => 5, relu), GCNConv(5 => 3))
        test_lux_layer(rng, c, g, x, outputsize=(3,), container=true)
    end
end
