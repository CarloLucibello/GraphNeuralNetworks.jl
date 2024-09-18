using Documenter
using GNNlib
using GNNLux



assets=[]
prettyurls = get(ENV, "CI", nothing) == "true"
mathengine = MathJax3()


makedocs(;
         modules = [GNNLux],
         doctest = false,
         clean = true,
         format = Documenter.HTML(; mathengine, prettyurls, assets = assets, size_threshold=nothing),
         sitename = "GNNLux.jl",
         pages = ["Home" => "index.md",
         "Basic" => "api/basic.md"],
         )
         
         


deploydocs(;repo = "https://github.com/CarloLucibello/GraphNeuralNetworks.jl.git",
dirname = "GNNLux")