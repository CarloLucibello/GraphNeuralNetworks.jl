using MultiDocumenter


docs = [ MultiDocumenter.MultiDocRef(

        upstream = joinpath(dirname(@__DIR__), "GNNGraphs", "docs", "build"),
                path = "gnngraphs",
                name = "GNNGraphs",
                fix_canonical_url = false),
]

outpath = joinpath(@__DIR__, "build")

MultiDocumenter.make(
    outpath,
    docs;
    search_engine = MultiDocumenter.SearchConfig(
        index_versions = ["stable"],
        engine = MultiDocumenter.FlexSearch
    )
)