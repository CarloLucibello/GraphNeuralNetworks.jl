using Documenter
using DocumenterInterLinks
using GNNlib
using GNNLux



assets=[]
prettyurls = get(ENV, "CI", nothing) == "true"
mathengine = MathJax3()

interlinks = InterLinks(
    "GNNGraphs" => ("https://carlolucibello.github.io/GraphNeuralNetworks.jl/GNNGraphs/",  joinpath(dirname(dirname(@__DIR__)), "GNNGraphs", "docs", "build", "objects.inv")),
    "GNNlib" => ("https://carlolucibello.github.io/GraphNeuralNetworks.jl/GNNlib/",  joinpath(dirname(dirname(@__DIR__)), "GNNlib", "docs", "build", "objects.inv")))
        
makedocs(;
         modules = [GNNLux],
         doctest = false,
         clean = true,
         plugins = [interlinks],
         format = Documenter.HTML(; mathengine, prettyurls, assets = assets, size_threshold=nothing),
         sitename = "GNNLux.jl",
         pages = ["Home" => "index.md",
                   "API Reference" => [
                                    "Basic" => "api/basic.md",
                                    "Convolutional layers" => "api/conv.md"]]
         )
         
         
deploydocs(;repo = "github.com/JuliaGraphs/GraphNeuralNetworks.jl.git",
devbranch = "master",
push_preview = true,
dirname = "GNNLux")