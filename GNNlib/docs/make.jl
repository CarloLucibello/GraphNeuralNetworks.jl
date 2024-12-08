using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, "..", "..", "GNNGraphs"))
Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Documenter
using GNNlib
using GNNGraphs
using DocumenterInterLinks

assets=[]
prettyurls = get(ENV, "CI", nothing) == "true"
mathengine = MathJax3()

interlinks = InterLinks(
    "NNlib" => "https://fluxml.ai/NNlib.jl/stable/",
)


makedocs(;
    modules = [GNNlib],
    doctest = false,
    clean = true,
    plugins = [interlinks],
    format = Documenter.HTML(; mathengine, prettyurls, assets = assets, size_threshold=nothing),
    sitename = "GNNlib.jl",
    pages = [
        "Home" => "index.md",
        "Message Passing" => "guides/messagepassing.md",
        "API Reference" => [
            "Message Passing" => "api/messagepassing.md",
            "Utils" => "api/utils.md",
        ]
    ]
)

deploydocs(
    repo = "github.com/JuliaGraphs/GraphNeuralNetworks.jl", 
    target = "build",
    branch = "docs-gnnlib",
    devbranch = "master", 
    tag_prefix="GNNlib-",
)
