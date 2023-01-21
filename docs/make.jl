using Flux, NNlib, GraphNeuralNetworks, Graphs, SparseArrays
using Documenter, DemoCards

tutorials, tutorials_cb, tutorial_assets = makedemos("tutorials")

assets = []
isnothing(tutorial_assets) || push!(assets, tutorial_assets)

DocMeta.setdocmeta!(GraphNeuralNetworks, :DocTestSetup,
                    :(using GraphNeuralNetworks, Graphs, SparseArrays, NNlib, Flux);
                    recursive = true)

prettyurls = get(ENV, "CI", nothing) == "true"
mathengine = MathJax3()

makedocs(;
         modules = [GraphNeuralNetworks, NNlib, Flux, Graphs, SparseArrays],
         doctest = false,
         clean = true,
         format = Documenter.HTML(; mathengine, prettyurls, assets = assets),
         sitename = "GraphNeuralNetworks.jl",
         pages = ["Home" => "index.md",
             "Graphs" => "gnngraph.md",
             "Message Passing" => "messagepassing.md",
             "Model Building" => "models.md",
             "Datasets" => "datasets.md",
             "HeteroGraphs" => "gnnheterograph.md",
             "Tutorials" => tutorials,
             "API Reference" => [
                 "GNNGraph" => "api/gnngraph.md",
                 "Basic Layers" => "api/basic.md",
                 "Convolutional Layers" => "api/conv.md",
                 "Pooling Layers" => "api/pool.md",
                 "Message Passing" => "api/messagepassing.md",
                 "Utils" => "api/utils.md",
             ],
             "Developer Notes" => "dev.md",
             "Summer Of Code" => "gsoc.md",
         ])

tutorials_cb()

deploydocs(repo = "github.com/CarloLucibello/GraphNeuralNetworks.jl.git")
