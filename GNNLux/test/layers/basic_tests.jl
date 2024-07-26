@testitem "layers/basic" setup=[SharedTestSetup] begin
    @testset "GNNLayer" begin
        @test GNNLayer <: LuxCore.AbstractExplicitLayer
    end
end
