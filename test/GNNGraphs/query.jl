@testset "Query" begin
    @testset "is_bidirected" begin
        g = rand_graph(10, 20, bidirected=true)
        @test is_bidirected(g)
        
        g = rand_graph(10, 20, bidirected=false)
        @test !is_bidirected(g)
    end
end
