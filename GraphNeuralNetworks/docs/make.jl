using Documenter
using GraphNeuralNetworks
using DocumenterInterLinks


assets=[]
prettyurls = get(ENV, "CI", nothing) == "true"
mathengine = MathJax3()

interlinks = InterLinks(
    "NNlib" => "https://fluxml.ai/NNlib.jl/stable/",
    "GNNGraphs" => ("https://carlolucibello.github.io/GraphNeuralNetworks.jl/GNNGraphs/",  joinpath(dirname(dirname(@__DIR__)), "GNNGraphs", "docs", "build", "objects.inv")),
    "GNNlib" => ("https://carlolucibello.github.io/GraphNeuralNetworks.jl/GNNlib/",  joinpath(dirname(dirname(@__DIR__)), "GNNlib", "docs", "build", "objects.inv"))
   
   )

# Copy the guides from GNNGraphs and GNNlib
dest_guides_dir = joinpath(@__DIR__, "src/other")
gnngraphs_guides_dir = joinpath(@__DIR__, "../../GNNGraphs/docs/src/guides")
gnnlib_guides_dir = joinpath(@__DIR__, "../../GNNlib/docs/src/guides") 
for file in readdir(gnngraphs_guides_dir)
    cp(joinpath(gnngraphs_guides_dir, file), joinpath(dest_guides_dir, file), force=true)
end
for file in readdir(gnnlib_guides_dir)
    cp(joinpath(gnnlib_guides_dir, file), joinpath(dest_guides_dir, file), force=true)
end

makedocs(;
    modules = [GraphNeuralNetworks],
    doctest = false, # TODO: enable doctest
    clean = true,
    plugins = [interlinks],
    format = Documenter.HTML(; mathengine, prettyurls, assets = assets, size_threshold=nothing),
    sitename = "GraphNeuralNetworks.jl",
    pages = [
    
    "Home" => "index.md",
    
    "Guides" => [
        "Graphs" => ["other/gnngraph.md", "other/heterograph.md", "other/temporalgraph.md"],
        "Message Passing" => "other/messagepassing.md",
        "Models" => "guides/models.md",
        "Datasets" => "other/datasets.md",
    ],

    "API Reference" => [
            "Basic" => "api/basic.md",
            "Convolutional layers" => "api/conv.md",
            "Pooling layers" => "api/pool.md",
            "Temporal Convolutional layers" => "api/temporalconv.md",
            "Hetero Convolutional layers" => "api/heteroconv.md",
        ],
        "Developer guide" => "dev.md",
    ],
)
         
deploydocs(;repo = "github.com/JuliaGraphs/GraphNeuralNetworks.jl.git", devbranch = "master", dirname= "GraphNeuralNetworks")
