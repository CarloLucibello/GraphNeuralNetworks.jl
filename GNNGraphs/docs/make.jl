using Documenter
using GNNGraphs
using Graphs

assets=[]
prettyurls = get(ENV, "CI", nothing) == "true"
mathengine = MathJax3()

makedocs(;
         modules = [GNNGraphs],
         doctest = false,
         clean = true,
         format = Documenter.HTML(; mathengine, prettyurls, assets = assets, size_threshold=nothing),
         sitename = "GNNGraphs.jl",
         pages = ["Home" => "index.md",
            "Graphs" => ["gnngraph.md", "heterograph.md", "temporalgraph.md"],
            #  "Message Passing" => "messagepassing.md",
            #  "Model Building" => "models.md",
            "Datasets" => "datasets.md",
            #  #"Tutorials" => tutorials,
            "API Reference" => [
                 "GNNGraph" => "api/gnngraph.md",
            #      "Basic Layers" => "api/basic.md",
            #      "Convolutional Layers" => "api/conv.md",
            #      "Pooling Layers" => "api/pool.md",
            #      "Message Passing" => "api/messagepassing.md",
                 "Heterogeneous Graphs" => "api/heterograph.md",
                 "Temporal Graphs" => "api/temporalgraph.md",
            #      "Utils" => "api/utils.md",
              ],
            #  "Developer Notes" => "dev.md",
            
         ]
         )
         
         


deploydocs(;repo = "https://github.com/CarloLucibello/GraphNeuralNetworks.jl.git",
dirname = "GNNGraphs")