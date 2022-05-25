using Flux, NNlib, GraphNeuralNetworks, Graphs, SparseArrays
using Documenter

# See this for a template of Pluto+Documenter 
# https://github.com/rikhuijzer/PlutoStaticHTML.jl/blob/main/docs/make.jl
using Pluto, PlutoStaticHTML
using Documenter: MathJax3

# tutorial_menu = Array{Pair{String,String},1}()

#
# Generate Pluto Tutorial HTMLs

pluto_src_folder = joinpath(@__DIR__, "src", "tutorials")
# pluto_output_folder = joinpath(@__DIR__, "src", "tutorials")
# pluto_relative_path = "tutorials/"
# mkpath(pluto_output_folder)

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

# # Please do not use the same name as for a(n old) literate Tutorial
# pluto_files = [
#     "gnn_intro_pluto",
#     "graph_classification_pluto",
# ]
# pluto_titles = [
#     "Intro to Graph Neural Networks ",
#     "Graph Classification",
# ]

# # build menu and write files myself - tp set edit url correctly.
# for (title, file) in zip(pluto_titles, pluto_files)
#     global tutorial_menu
#     rendered = build_notebooks( #though not really parallel here
#         BuildOptions(
#             pluto_src_folder;
#             output_format=documenter_output,
#             write_files=false,
#             use_distributed=false,
#         ),
#         ["$(file).jl"],
#     )
#     write(
#         joinpath(pluto_output_folder, file * ".md"),
#         """
#         ```@meta
#         EditURL = "$(joinpath(pluto_src_folder, file * ".jl"))"
#         ```
#         $(rendered[1])
#         """,
#     )
#     push!(tutorial_menu, title => joinpath(pluto_relative_path, file * ".md"))
# end

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
            #  "Tutorials" => tutorial_menu,
             "Tutorials" => 
                [
                    "Intro to Graph Neural Networks" => "gnn_intro_pluto.md",
                    "Graph Classification" => "graph_classification_pluto.md",
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
