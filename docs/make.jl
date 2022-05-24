using Flux, NNlib, GraphNeuralNetworks, Graphs, SparseArrays
using Documenter
using Pluto, PlutoStaticHTML

TutorialMenu = Array{Pair{String,String},1}()

#
# Generate Pluto Tutorial HTMLs

# First tutorial with AD
pluto_src_folder = joinpath(@__DIR__, "tutorials/")
pluto_output_folder = joinpath(@__DIR__, "tutorials/")
pluto_relative_path = "tutorials/"
mkpath(pluto_output_folder)
#
#
# Please do not use the same name as for a(n old) literate Tutorial
pluto_files = [
    "gnn_intro.pluto",
    "graph_classification.pluto",
]
pluto_titles = [
    "Intro to Graph Neural Networks ",
    "Graph Classification",
]

# build menu and write files myself - tp set edit url correctly.
for (title, file) in zip(pluto_titles, pluto_files)
    global TutorialMenu
    rendered = build_notebooks( #though not really parallel here
        BuildOptions(
            pluto_src_folder;
            output_format=documenter_output,
            write_files=false,
            use_distributed=false,
        ),
        ["$(file).jl"],
    )
    write(
        pluto_output_folder * file * ".md",
        """
        ```@meta
        EditURL = "$(pluto_src_folder)$(file).jl"
        ```
        $(rendered[1])
        """,
    )
    push!(TutorialMenu, title => joinpath(pluto_relative_path, file * ".md"))
end

DocMeta.setdocmeta!(GraphNeuralNetworks, :DocTestSetup, 
    :(using GraphNeuralNetworks, Graphs, SparseArrays, NNlib, Flux); 
    recursive=true)

makedocs(;
    modules=[GraphNeuralNetworks, NNlib, Flux, Graphs, SparseArrays],
    doctest=false, clean=true,     
    sitename = "GraphNeuralNetworks.jl",
    pages = ["Home" => "index.md",
             "Graphs" => "gnngraph.md",
             "Message Passing" => "messagepassing.md",
             "Model Building" => "models.md",
             "Datasets" => "datasets.md",
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
