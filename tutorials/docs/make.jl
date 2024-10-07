using Documenter


assets = []
prettyurls = get(ENV, "CI", nothing) == "true"
mathengine = MathJax3()

# interlinks = InterLinks(
#     "NNlib" => "https://fluxml.ai/NNlib.jl/stable/",
#     "GNNGraphs" => ("https://carlolucibello.github.io/GraphNeuralNetworks.jl/GNNGraphs/",  joinpath(dirname(dirname(@__DIR__)), "GNNGraphs", "docs", "build", "objects.inv")),
#     "GraphNeuralNetworks" => ("https://carlolucibello.github.io/GraphNeuralNetworks.jl/GraphNeuralNetworks/",  joinpath(dirname(dirname(@__DIR__)), "docs", "build", "objects.inv")),)

makedocs(;
    doctest = false,
    clean = true,
    format = Documenter.HTML(;
        mathengine, prettyurls, assets = assets, size_threshold = nothing),
    sitename = "Tutorials",
    pages = ["Home" => "index.md",
        "Introductory tutorials" => [
            "Hands on" => "pluto_output/gnn_intro_pluto.md",
            "Node classification" => "pluto_output/node_classification_pluto.md", 
            "Graph classification" => "pluto_output/graph_classification_pluto.md"
            ],
        "Temporal graph neural networks" =>[
            "Node autoregression" => "pluto_output/traffic_prediction.md",
            "Temporal graph classification" => "pluto_output/temporal_graph_classification_pluto.md"

        ]])



deploydocs(; repo = "https://github.com/CarloLucibello/GraphNeuralNetworks.jl.git",
    dirname = "tutorials")