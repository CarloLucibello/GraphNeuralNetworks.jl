# We use the MultiDocumenter package, followoing the build and deployment approach
# used in https://github.com/JuliaAstro/EphemerisSources.jl
# This script is executed after building the docs for each package
# See the pipeling in .github/workflows/multidocs.yml

using MultiDocumenter

clonedir = mktempdir()

function package(name; path = joinpath("docs", name), branch)
    MultiDocumenter.MultiDocRef(
        upstream = joinpath(clonedir, name),
        path = path,
        name = name,
        branch = branch,
        giturl = "https://github.com/JuliaGraphs/GraphNeuralNetworks.jl.git",
        fix_canonical_url = false,
    )
end

docs = [
    package("GraphNeuralNetworks.jl", branch = "docs-graphneuralnetworks"),
    package("GNNLux", branch = "docs-gnnlux"),
    package("GNNGraphs", branch = "docs-gnngraphs"),
    package("GNNlib", branch = "docs-gnnlib"),
]

outpath = joinpath(@__DIR__, "build")

MultiDocumenter.make(
    outpath,
    docs;
    search_engine = MultiDocumenter.SearchConfig(
        index_versions = ["stable"],
        engine = MultiDocumenter.FlexSearch
    ),
    brand_image = MultiDocumenter.BrandImage("", "logo.svg"),
    rootpath = "/GraphNeuralNetworks.jl/"
)

Documenter.deploydocs(
    target = outpath,
    versions = nothing,
    repo = "github.com/JuliaGraphs/GraphNeuralNetworks.jl",
)
