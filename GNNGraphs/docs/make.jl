using Documenter
using DocumenterInterLinks
using GNNGraphs
using MLUtils # this is needed by setdocmeta!
import Graphs
using Graphs: induced_subgraph, has_edge


DocMeta.setdocmeta!(GNNGraphs, :DocTestSetup, :(using GNNGraphs, MLUtils); recursive = true)

mathengine = MathJax3(Dict(:loader => Dict("load" => ["[tex]/require", "[tex]/mathtools"]),
    :tex => Dict("inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
        "packages" => [
            "base",
            "ams",
            "autoload",
            "mathtools",
            "require"
        ])))

makedocs(;
    modules = [GNNGraphs],
    doctest = false, # TODO enable doctest
    format = Documenter.HTML(; mathengine, 
                    prettyurls = get(ENV, "CI", nothing) == "true", 
                    assets = [],
                    size_threshold=nothing, 
                    size_threshold_warn=200000),sitename = "GNNGraphs.jl",
    pages = [
    "Home" => "index.md",
    
    "Guides" => [
        "Graphs" => "guides/gnngraph.md", 
        "Heterogeneous Graphs" => "guides/heterograph.md",
        "Temporal Graphs" => "guides/temporalgraph.md",
        "Datasets" => "guides/datasets.md",
    ],

    "API Reference" => [
        "GNNGraph" => "api/gnngraph.md",
        "GNNHeteroGraph" => "api/heterograph.md",
        "TemporalSnapshotsGNNGraph" => "api/temporalgraph.md",
        "Samplers" => "api/samplers.md",
        ],      
    ]
)
         
deploydocs(repo = "github.com/JuliaGraphs/GraphNeuralNetworks.jl.git", 
         devbranch = "master", 
         dirname = "GNNGraphs",
         tag_prefix="GNNGraphs-")