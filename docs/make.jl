using GraphNeuralNetworks
using Documenter

DocMeta.setdocmeta!(GraphNeuralNetworks, :DocTestSetup, :(using GraphNeuralNetworks); recursive=true)

makedocs(;
    modules=[GraphNeuralNetworks],
    sitename = "GraphNeuralNetworks.jl",
    pages = ["Home" => "index.md",
             "Graphs" => "graphs.md",
             "Message passing" => "messagepassing.md",
             "Building models" => "models.md",
             "API Reference" =>
               [
                "Graphs" => "api/graphs.md",
                "Convolutional Layers" => "api/conv.md",
                "Pooling Layers" => "api/pool.md",
               ],
              "Developer Notes" => "dev.md",
            ],
)

deploydocs(repo="github.com/CarloLucibello/GraphNeuralNetworks.jl.git")
