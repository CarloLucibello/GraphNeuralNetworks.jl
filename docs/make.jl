using Flux, NNlib, GraphNeuralNetworks, Graphs, SparseArrays
using Documenter

DocMeta.setdocmeta!(GraphNeuralNetworks, :DocTestSetup, 
    :(using GraphNeuralNetworks, Graphs, SparseArrays, NNlib, Flux); 
    recursive=true)

makedocs(;
    modules=[GraphNeuralNetworks, NNlib, Flux, Graphs, SparseArrays],
    doctest=false, clean=true,     
    sitename = "GraphNeuralNetworks.jl",
    pages = ["Home" => "index.md",
             "Graphs" => "gnngraph.md",
             "Message Passing" => "messagepassing.md",
             "Model Building" => "models.md",
             "Datasets" => "datasets.md",
             "API Reference" =>
               [
                "GNNGraph" => "api/gnngraph.md",
                "Basic Layers" => "api/basic.md",
                "Convolutional Layers" => "api/conv.md",
                "Pooling Layers" => "api/pool.md",
                "Message Passing" => "api/messagepassing.md",
                "Utils" => "api/utils.md",
               ],
              "Developer Notes" => "dev.md",
            ],
)

deploydocs(repo="github.com/CarloLucibello/GraphNeuralNetworks.jl.git")
