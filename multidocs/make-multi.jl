using MultiDocumenter

docs = [
    MultiDocumenter.MultiDocRef(
        upstream = joinpath(dirname(@__DIR__), "GNNGraphs", "docs", "build"),
        path = "gnngraphs",
        name = "GNNGraphs",
        fix_canonical_url = false),
    MultiDocumenter.MultiDocRef(
        upstream = joinpath(dirname(@__DIR__), "GNNlib", "docs", "build"),
        path = "gnnlib",
        name = "GNNlib",
        fix_canonical_url = false)
]

outpath = joinpath(@__DIR__, "build")

MultiDocumenter.make(
    outpath,
    docs;
    search_engine = MultiDocumenter.SearchConfig(
        index_versions = ["stable"],
        engine = MultiDocumenter.FlexSearch
    ),
    brand_image = MultiDocumenter.BrandImage("", "logo.svg")
)

cp(joinpath(@__DIR__, "logo.svg"),
    joinpath(outpath, "logo.svg"))
