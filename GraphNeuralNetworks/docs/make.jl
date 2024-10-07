using Documenter
using GraphNeuralNetworks
using DocumenterInterLinks


assets=[]
prettyurls = get(ENV, "CI", nothing) == "true"
mathengine = MathJax3()

interlinks = InterLinks(
    "NNlib" => "https://fluxml.ai/NNlib.jl/stable/",
    "GNNGraphs" => ("https://carlolucibello.github.io/GraphNeuralNetworks.jl/GNNGraphs/",  joinpath(dirname(dirname(@__DIR__)), "GNNGraphs", "docs", "build", "objects.inv")),
    "GNNlib" => ("https://carlolucibello.github.io/GraphNeuralNetworks.jl/GNNlib/",  joinpath(dirname(dirname(@__DIR__)), "GNNlib", "docs", "build", "objects.inv"))
   
   )

makedocs(;
         modules = [GraphNeuralNetworks],
         doctest = false,
         clean = true,
         plugins = [interlinks],
         format = Documenter.HTML(; mathengine, prettyurls, assets = assets, size_threshold=nothing),
         sitename = "GraphNeuralNetworks.jl",
         pages = ["Monorepo" => [ 
               "Home" => "index.md",
               "Developer guide" => "dev.md",
               "Google Summer of Code" => "gsoc.md",


            ],
            "GraphNeuralNetworks.jl" =>[
            "Home" => "home.md",
            "Models" => "models.md",],

            "API Reference" => [
     
                  "Basic" => "api/basic.md",
                  "Convolutional layers" => "api/conv.md",
                  "Pooling layers" => "api/pool.md",
                  "Temporal Convolutional layers" => "api/temporalconv.md",
                  "Hetero Convolutional layers" => "api/heteroconv.md"
          
                
              ],
            
         ],
         )
         
         


deploydocs(;repo = "https://github.com/CarloLucibello/GraphNeuralNetworks.jl.git", dirname= "GraphNeuralNetworks")