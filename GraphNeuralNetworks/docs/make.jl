using Documenter
using GraphNeuralNetworks
using Flux, GNNGraphs, GNNlib, Graphs, NNlib
# using GNNGraphs, GNNlib # no need to load since they are re-exported by GraphNeuralNetworks
using DocumenterInterLinks

DocMeta.setdocmeta!(GraphNeuralNetworks, :DocTestSetup, :(using GraphNeuralNetworks); recursive = true)

mathengine = MathJax3(Dict(:loader => Dict("load" => ["[tex]/require", "[tex]/mathtools"]),
    :tex => Dict("inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
        "packages" => [
            "base",
            "ams",
            "autoload",
            "mathtools",
            "require"
        ])))


interlinks = InterLinks(
    "NNlib" => "https://fluxml.ai/NNlib.jl/stable/",
    # "GNNGraphs" => ("https://carlolucibello.github.io/GraphNeuralNetworks.jl/GNNGraphs/",  joinpath(dirname(dirname(@__DIR__)), "GNNGraphs", "docs", "build", "objects.inv")),
    # "GNNlib" => ("https://carlolucibello.github.io/GraphNeuralNetworks.jl/GNNlib/",  joinpath(dirname(dirname(@__DIR__)), "GNNlib", "docs", "build", "objects.inv"))
   )

# Copy the docs from GNNGraphs and GNNlib. Will be removed at the end of the script
cp(joinpath(@__DIR__, "../../GNNGraphs/docs/src"),
   joinpath(@__DIR__, "src/GNNGraphs"), force=true)
cp(joinpath(@__DIR__, "../../GNNlib/docs/src"),
   joinpath(@__DIR__, "src/GNNlib"), force=true)

makedocs(;
    modules = [GraphNeuralNetworks, GNNGraphs, GNNlib],
    doctest = false, # TODO: enable doctest
    plugins = [interlinks],
    format = Documenter.HTML(; mathengine, 
                            prettyurls = get(ENV, "CI", nothing) == "true", 
                            assets = [],
                            size_threshold=nothing, 
                            size_threshold_warn=200000),
    sitename = "GraphNeuralNetworks.jl",
    pages = [
    
    "Home" => "index.md",
    
    "Guides" => [
        "Graphs" => ["GNNGraphs/guides/gnngraph.md", 
                    "GNNGraphs/guides/heterograph.md", 
                    "GNNGraphs/guides/temporalgraph.md"],
        "Message Passing" => "GNNlib/guides/messagepassing.md",
        "Models" => "guides/models.md",
        "Datasets" => "GNNGraphs/guides/datasets.md",
    ],

    "API Reference" => [
            "Message Passing" => "GNNlib/api/messagepassing.md",
            "Utils" => "GNNlib/api/utils.md",
            "Basic" => "api/basic.md",
            "Convolutional layers" => "api/conv.md",
            "Pooling layers" => "api/pool.md",
            "Temporal Convolutional layers" => "api/temporalconv.md",
            "Hetero Convolutional layers" => "api/heteroconv.md",

        ],
    
    "Developer guide" => "dev.md",
    ],
)

rm(joinpath(@__DIR__, "src/GNNGraphs"), force=true, recursive=true)
rm(joinpath(@__DIR__, "src/GNNlib"), force=true, recursive=true)

deploydocs(repo = "github.com/JuliaGraphs/GraphNeuralNetworks.jl.git", 
          devbranch = "master", 
          dirname= "GraphNeuralNetworks")
