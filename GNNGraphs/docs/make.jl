using Documenter
using DocumenterInterLinks
using GNNGraphs
using Graphs

assets=[]
prettyurls = get(ENV, "CI", nothing) == "true"
mathengine = MathJax3()

interlinks = InterLinks(
    "Graphs" => "http://juliagraphs.org/Graphs.jl/stable/")

makedocs(;
         modules = [GNNGraphs],
         doctest = false,
         clean = true,
         plugins = [interlinks],
         format = Documenter.HTML(; mathengine, prettyurls, assets = assets, size_threshold=nothing),
         sitename = "GNNGraphs.jl",
         pages = ["Home" => "index.md",
            "Graphs" => ["gnngraph.md", "heterograph.md", "temporalgraph.md"],
            "Datasets" => "datasets.md",
            "API Reference" => [
                 "GNNGraph" => "api/gnngraph.md",
                 "GNNHeteroGraph" => "api/heterograph.md",
                 "TemporalSnapshotsGNNGraph" => "api/temporalgraph.md",
              ],      
         ]
         )
         
         


deploydocs(;repo = "https://github.com/CarloLucibello/GraphNeuralNetworks.jl.git",
dirname = "GNNGraphs")