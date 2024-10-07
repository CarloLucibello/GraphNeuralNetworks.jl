using Documenter
using GNNlib
using GNNGraphs
using DocumenterInterLinks


assets=[]
prettyurls = get(ENV, "CI", nothing) == "true"
mathengine = MathJax3()

interlinks = InterLinks(
    "NNlib" => "https://fluxml.ai/NNlib.jl/stable/",
    "GNNGraphs" => ("https://carlolucibello.github.io/GraphNeuralNetworks.jl/GNNGraphs/",  joinpath(dirname(dirname(@__DIR__)), "GNNGraphs", "docs", "build", "objects.inv")))


makedocs(;
         modules = [GNNlib],
         doctest = false,
         clean = true,
         plugins = [interlinks],
         format = Documenter.HTML(; mathengine, prettyurls, assets = assets, size_threshold=nothing),
         sitename = "GNNlib.jl",
         pages = ["Home" => "index.md",
            "Message Passing" => "messagepassing.md",

            "API Reference" => [
     
                  "Message Passing" => "api/messagepassing.md",
          
                "Utils" => "api/utils.md",
              ]
            
         ]
         )
         
         


deploydocs(;repo = "https://github.com/CarloLucibello/GraphNeuralNetworks.jl.git",
dirname = "GNNlib")