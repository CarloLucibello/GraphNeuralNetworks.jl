using Documenter
using GraphNeuralNetworks
using DocumenterInterLinks


assets=[]
prettyurls = get(ENV, "CI", nothing) == "true"
mathengine = MathJax3()

interlinks = InterLinks(
    "NNlib" => "https://fluxml.ai/NNlib.jl/stable/",
    "GNNGraphs" => ("https://carlolucibello.github.io/GraphNeuralNetworks.jl/gnngraphs/",  joinpath(dirname(dirname(@__DIR__)),"GraphNeuralNetworks.jl", "GNNGraphs", "docs", "build", "objects.inv")),
    "GNNlib" => ("https://carlolucibello.github.io/GraphNeuralNetworks.jl/gnnlib/",  joinpath(dirname(dirname(@__DIR__)),"GraphNeuralNetworks.jl", "GNNlib", "docs", "build", "objects.inv"))
   
   )

makedocs(;
         modules = [GraphNeuralNetworks],
         doctest = false,
         clean = true,
         plugins = [interlinks],
         format = Documenter.HTML(; mathengine, prettyurls, assets = assets, size_threshold=nothing),
         sitename = "GraphNeuralNetworks.jl",
         pages = ["Home" => "index.md",
            "Models" => "models.md",
            "Dev" => "dev.md",
            "Gsoc" => "gsoc.md",

            "API Reference" => [
     
                  "Basic" => "api/basic.md",
                  "Conv" => "api/conv.md",
                  "Pool" => "api/pool.md",
                  "TempConv" => "api/temporalconv.md",
                  "HeteroConv" => "api/heteroconv.md"
          
                
              ],
            #  "Developer Notes" => "dev.md",
            
         ],
         )
         
         


deploydocs(;repo = "https://github.com/CarloLucibello/GraphNeuralNetworks.jl.git")