using Flux, NNlib, GraphNeuralNetworks, Graphs, SparseArrays
using Documenter

# See this for a template of Pluto+Documenter 
# https://github.com/rikhuijzer/PlutoStaticHTML.jl/blob/main/docs/make.jl
using Pluto, PlutoStaticHTML
using Documenter: MathJax3

pluto_src_folder = joinpath(@__DIR__, "src", "tutorials")

"""
    build()

Run all Pluto notebooks (".jl" files) in `NOTEBOOK_DIR`.
"""
function build()
    println("Building notebooks")
    hopts = HTMLOptions(; append_build_context=false)
    output_format = documenter_output
    bopts = BuildOptions(pluto_src_folder; output_format)
    build_notebooks(bopts, hopts)
    return nothing
end

# Build the notebooks; defaults to true.
if get(ENV, "BUILD_DOCS_NOTEBOOKS", "true") == "true"
    build()
end

DocMeta.setdocmeta!(GraphNeuralNetworks, :DocTestSetup, 
    :(using GraphNeuralNetworks, Graphs, SparseArrays, NNlib, Flux); 
    recursive=true)

prettyurls = get(ENV, "CI", nothing) == "true"
mathengine = MathJax3()
    
makedocs(;
    modules = [GraphNeuralNetworks, NNlib, Flux, Graphs, SparseArrays],
    doctest = false, 
    clean = true,    
    format= Documenter.HTML(; mathengine, prettyurls),
    sitename = "GraphNeuralNetworks.jl",
    pages = ["Home" => "index.md",
             "Graphs" => "gnngraph.md",
             "Message Passing" => "messagepassing.md",
             "Model Building" => "models.md",
             "Datasets" => "datasets.md",
             "Tutorials" => 
                [
                    "Intro to Graph Neural Networks" => "tutorials/gnn_intro_pluto.md",
                    "Graph Classification" => "tutorials/graph_classification_pluto.md",
                    "Node Classification" => "tutorials/node_classification_pluto.md",
                ],
             "API Reference" =>
               [
                "GNNGraph" => "api/gnngraph.md",
                "Basic Layers" => "api/basic.md",
                "Convolutional Layers" => "api/conv.md",
                "Pooling Layers" => "api/pool.md",
                "Message Passing" => "api/messagepassing.md",
                "Utils" => "api/utils.md",
               ],
              "Developer Notes" => "dev.md",
              "Summer Of Code" => "gsoc.md",
            ],
)

deploydocs(repo="github.com/CarloLucibello/GraphNeuralNetworks.jl.git")
