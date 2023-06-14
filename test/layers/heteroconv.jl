@testset "HeteroGraphConv" begin
    d, n = 3, 5
    g = rand_bipartite_heterograph(n, 2*n, 15)

    model = HeteroGraphConv([(:A,:to,:B) => GraphConv(d => d), 
                            (:B,:to,:A) => GraphConv(d => d)])

    x = (A = rand(Float32, d, n), B = rand(Float32, d, 2n))

    y = model(g, x)

    grad = gradient(model -> sum(model(g, x)[1]) + sum(model(g, x)[2].^2), model)[1]
    ngrad = ngradient(model -> sum(model(g, x)[1]) + sum(model(g, x)[2].^2), model)[1]

    @test grad.layers[1].weight1 ≈ ngrad.layers[1].weight1  rtol=1e-4
    @test grad.layers[1].weight2 ≈ ngrad.layers[1].weight2  rtol=1e-4
    @test grad.layers[1].bias ≈ ngrad.layers[1].bias        rtol=1e-4
    @test grad.layers[2].weight1 ≈ ngrad.layers[2].weight1  rtol=1e-4
    @test grad.layers[2].weight2 ≈ ngrad.layers[2].weight2  rtol=1e-4
    @test grad.layers[2].bias ≈ ngrad.layers[2].bias        rtol=1e-4
end


GNNHeteroGraph(::Dict{Tuple{Symbol, Symbol, Symbol}, Tuple{Vector{Int64}, Vector{Int64}, Nothing}}, 
               ::Dict{Symbol, Int64}, 
               ::Dict{Tuple{Symbol, Symbol, Symbol}, Int64}, 
               ::Int64, 
               ::Nothing, 
               ::Dict{Any, Any}, 
               ::Dict{Any, Any}, 
               ::DataStore, 
               ::Vector{Symbol}, 
               ::Vector{Tuple{Symbol, Symbol, Symbol}})