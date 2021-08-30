using GraphNeuralNetworks
using Documenter

DocMeta.setdocmeta!(GraphNeuralNetworks, :DocTestSetup, :(using GraphNeuralNetworks); recursive=true)

makedocs(;
    modules=[GraphNeuralNetworks],
    authors="Carlo Lucibello <carlo.lucibello@gmail.com> and contributors",
    repo="https://github.com/CarloLucibello/GraphNeuralNetworks.jl/blob/{commit}{path}#{line}",
    sitename="GraphNeuralNetworks.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://CarloLucibello.github.io/GraphNeuralNetworks.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/CarloLucibello/GraphNeuralNetworks.jl",
)
