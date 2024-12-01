using Documenter
using DocumenterInterLinks
using GNNGraphs
import Graphs
using Graphs: induced_subgraph

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
            "Guides" => [
               "Graphs" => ["guides/gnngraph.md", "guides/heterograph.md", "guides/temporalgraph.md"],
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
         
deploydocs(;repo = "github.com/JuliaGraphs/GraphNeuralNetworks.jl.git", devbranch = "master", dirname = "GNNGraphs")