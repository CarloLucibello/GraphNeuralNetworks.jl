using Flux, NNlib, GraphNeuralNetworks, LightGraphs, SparseArrays
using Documenter

DocMeta.setdocmeta!(GraphNeuralNetworks, :DocTestSetup, :(using GraphNeuralNetworks); recursive=true)

makedocs(;
    modules=[GraphNeuralNetworks, NNlib, Flux, LightGraphs, SparseArrays],
    doctest=false, clean=true,     
    sitename = "GraphNeuralNetworks.jl",
    pages = ["Home" => "index.md",
             "GNNGraph" => "gnngraph.md",
             "Message Passing" => "messagepassing.md",
             "Model Building" => "models.md",
             "API Reference" =>
               [
                "GNNGraph" => "api/gnngraph.md",
                "Basic Layers" => "api/basic.md",
                "Convolutional Layers" => "api/conv.md",
                "Pooling Layers" => "api/pool.md",
                "Message Passing" => "api/messagepassing.md",
                "NNlib" => "api/nnlib.md",
               ],
              "Developer Notes" => "dev.md",
            ],
)

deploydocs(repo="github.com/CarloLucibello/GraphNeuralNetworks.jl.git")
