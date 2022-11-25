using DataFrames, MLDatasets
using GraphNeuralNetworks
using GraphNeuralNetworks.GNNGraphs: HeteroGNNGraph
using Test

@testset "HeteroGNNGraph" begin
    d = MovieLens("latest-small")[1]
    hg = HeteroGNNGraph(d.edge_indices)
    @test hg.num_nodes == Dict("movie" => 193609, "user" => 193609)
    @test hg.num_edges == Dict(("user", "tag", "movie")  => 7366, ("user", "rating", "movie") => 201672)    
    @test hg.graph_indicator === nothing
end
