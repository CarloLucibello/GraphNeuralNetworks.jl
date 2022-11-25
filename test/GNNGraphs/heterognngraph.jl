using DataFrames, MLDatasets
using GraphNeuralNetworks
using GraphNeuralNetworks.GNNGraphs: HeteroGNNGraph
using Test

@testset "HeteroGNNGraph" begin
    # d = MovieLens("100k")[1]
    # hg = HeteroGNNGraph(d.edge_indices)
    # @test hg.num_nodes == Dict("movie" => 193609, "user" => 193609)
    # @test hg.num_edges == Dict(("user", "tag", "movie")  => 7366, ("user", "rating", "movie") => 201672)    
    # @test hg.graph_indicator === nothing

    # @test hg["user", "tag", "movie"] == (graph = hg.graph[("user", "tag", "movie")], edata = hg.edata[("user", "tag", "movie")])


    hg = rand_heterograph(Dict("A" => 10, "B" => 20), 
                          Dict(("A", "rel1", "B") => 30, ("B", "rel2", "A") => 10))

    @test hg.num_nodes == Dict("A" => 10, "B" => 20)
    @test hg.num_edges == Dict(("A", "rel1", "B") => 30, ("B", "rel2", "A") => 10)
end
