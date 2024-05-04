@testset "intersect" begin
    g = rand_graph(10, 20, graph_type = GRAPH_T)
    @test intersect(g, g).num_edges == 20
end
