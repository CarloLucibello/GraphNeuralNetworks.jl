# d = MovieLens("100k")[1]
# hg = GNNHeteroGraph(d.edge_indices)
# @test hg.num_nodes == Dict("movie" => 193609, "user" => 193609)
# @test hg.num_edges == Dict(("user", "tag", "movie")  => 7366, ("user", "rating", "movie") => 201672)    
# @test hg.graph_indicator === nothing

# @test hg["user", "tag", "movie"] == (graph = hg.graph[("user", "tag", "movie")], edata = hg.edata[("user", "tag", "movie")])

@testset "Generation" begin
    hg = rand_heterograph(Dict(:A => 10, :B => 20),
                            Dict((:A, :rel1, :B) => 30, (:B, :rel2, :A) => 10))

    @test hg.num_nodes == Dict(:A => 10, :B => 20)
    @test hg.num_edges == Dict((:A, :rel1, :B) => 30, (:B, :rel2, :A) => 10)
    @test hg.graph_indicator === nothing
    @test hg.num_graphs == 1
    @test hg.ndata == Dict()
    @test hg.edata == Dict()
    @test isempty(hg.gdata)
    @test sort(hg.ntypes) == [:A, :B]
    @test sort(hg.etypes) == [:rel1, :rel2]
end

@testset "features" begin
    hg = rand_heterograph(Dict(:A => 10, :B => 20),
                            Dict((:A, :rel1, :B) => 30, (:B, :rel2, :A) => 10),
                            ndata = Dict(:A => rand(2, 10),
                                        :B => (x = rand(3, 20), y = rand(4, 20))),
                            edata = Dict((:A, :rel1, :B) => rand(5, 30)),
                            gdata = 1)

    @test size(hg.ndata[:A].x) == (2, 10)
    @test size(hg.ndata[:B].x) == (3, 20)
    @test size(hg.ndata[:B].y) == (4, 20)
    @test size(hg.edata[(:A, :rel1, :B)].e) == (5, 30)
    @test hg.gdata == DataStore(u = 1)
end

@testset "simplified constructor" begin
    hg = rand_heterograph((:A => 10, :B => 20),
                            ((:A, :rel1, :B) => 30, (:B, :rel2, :A) => 10),
                            ndata = (:A => rand(2, 10),
                                    :B => (x = rand(3, 20), y = rand(4, 20))),
                            edata = (:A, :rel1, :B) => rand(5, 30),
                            gdata = 1)

    @test hg.num_nodes == Dict(:A => 10, :B => 20)
    @test hg.num_edges == Dict((:A, :rel1, :B) => 30, (:B, :rel2, :A) => 10)
    @test hg.graph_indicator === nothing
    @test hg.num_graphs == 1
    @test size(hg.ndata[:A].x) == (2, 10)
    @test size(hg.ndata[:B].x) == (3, 20)
    @test size(hg.ndata[:B].y) == (4, 20)
    @test size(hg.edata[(:A, :rel1, :B)].e) == (5, 30)
    @test hg.gdata == DataStore(u = 1)

    nA, nB = 10, 20
    edges1 = rand(1:nA, 20), rand(1:nB, 20)
    edges2 = rand(1:nB, 30), rand(1:nA, 30)
    hg = GNNHeteroGraph(((:A, :rel1, :B) => edges1, (:B, :rel2, :A) => edges2))
    @test hg.num_edges == Dict((:A, :rel1, :B) => 20, (:B, :rel2, :A) => 30)

    nA, nB = 10, 20
    edges1 = rand(1:nA, 20), rand(1:nB, 20)
    edges2 = rand(1:nB, 30), rand(1:nA, 30)
    hg = GNNHeteroGraph(((:A, :rel1, :B) => edges1, (:B, :rel2, :A) => edges2);
                        num_nodes = (:A => nA, :B => nB))
    @test hg.num_nodes == Dict(:A => 10, :B => 20)
    @test hg.num_edges == Dict((:A, :rel1, :B) => 20, (:B, :rel2, :A) => 30)
end

## Cannot test this because DataStore is not an oredered collection
## Uncomment when/if it will be based on OrderedDict
# @testset "show" begin
#     num_nodes = Dict(:A => 10, :B => 20);
#     edges1 = rand(1:num_nodes[:A], 20), rand(1:num_nodes[:B], 20)
#     edges2 = rand(1:num_nodes[:B], 30), rand(1:num_nodes[:A], 30)
#     eindex = ((:A, :rel1, :B) => edges1, (:B, :rel2, :A) => edges2)
#     ndata = Dict(:A => (x = rand(2, num_nodes[:A]), y = rand(3, num_nodes[:A])),:B => rand(10, num_nodes[:B]))
#     edata= Dict((:A, :rel1, :B) => (x = rand(2, 20), y = rand(3, 20)),(:B, :rel2, :A) => rand(10, 30))
#     hg1 = GraphNeuralNetworks.GNNHeteroGraph(eindex; num_nodes)
#     hg2 = GraphNeuralNetworks.GNNHeteroGraph(eindex; num_nodes, ndata,edata)
#     hg3 = GraphNeuralNetworks.GNNHeteroGraph(eindex; num_nodes, ndata)
#     @test sprint(show, hg1) == "GNNHeteroGraph(Dict(:A => 10, :B => 20), Dict((:A, :rel1, :B) => 20, (:B, :rel2, :A) => 30))"
#     @test sprint(show, hg2) == sprint(show, hg1)
#     @test sprint(show, MIME("text/plain"), hg1; context=:compact => true) == "GNNHeteroGraph(Dict(:A => 10, :B => 20), Dict((:A, :rel1, :B) => 20, (:B, :rel2, :A) => 30))"
#     @test sprint(show, MIME("text/plain"), hg2; context=:compact => true) == sprint(show, MIME("text/plain"), hg1;context=:compact => true)
#     @test sprint(show, MIME("text/plain"), hg1; context=:compact => false) == "GNNHeteroGraph:\n num_nodes: (:A => 10, :B => 20)\n num_edges: ((:A, :rel1, :B) => 20, (:B, :rel2, :A) => 30)"
#     @test sprint(show, MIME("text/plain"), hg2; context=:compact => false) == "GNNHeteroGraph:\n num_nodes: (:A => 10, :B => 20)\n num_edges: ((:A, :rel1, :B) => 20, (:B, :rel2, :A) => 30)\n ndata:\n\t:A  =>  (x = 2×10 Matrix{Float64}, y = 3×10 Matrix{Float64})\n\t:B  =>  x = 10×20 Matrix{Float64}\n edata:\n\t(:A, :rel1, :B)  =>  (x = 2×20 Matrix{Float64}, y = 3×20 Matrix{Float64})\n\t(:B, :rel2, :A)  =>  e = 10×30 Matrix{Float64}"
#     @test sprint(show, MIME("text/plain"), hg3; context=:compact => false) =="GNNHeteroGraph:\n num_nodes: (:A => 10, :B => 20)\n num_edges: ((:A, :rel1, :B) => 20, (:B, :rel2, :A) => 30)\n ndata:\n\t:A  =>  (x = 2×10 Matrix{Float64}, y = 3×10 Matrix{Float64})\n\t:B  =>  x = 10×20 Matrix{Float64}"
#     @test sprint(show, MIME("text/plain"), hg2; context=:compact => false) != sprint(show, MIME("text/plain"), hg3; context=:compact => false)
# end

