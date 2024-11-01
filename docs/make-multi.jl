using MultiDocumenter

for (root, dirs, files) in walkdir(".")
    for file in files
        filepath = joinpath(root, file)
        if islink(filepath)
            linktarget = abspath(dirname(filepath), readlink(filepath))
            rm(filepath)
            cp(linktarget, filepath; force=true)
        end
    end
end

docs = [
    MultiDocumenter.MultiDocRef(
        upstream = joinpath(dirname(@__DIR__),"GraphNeuralNetworks", "docs", "build"),
        path = "graphneuralnetworks",
        name = "GraphNeuralNetworks",
        fix_canonical_url = false),
    MultiDocumenter.MultiDocRef(
        upstream = joinpath(dirname(@__DIR__), "GNNGraphs", "docs", "build"),
        path = "gnngraphs",
        name = "GNNGraphs",
        fix_canonical_url = false),
    MultiDocumenter.MultiDocRef(
        upstream = joinpath(dirname(@__DIR__), "GNNlib", "docs", "build"),
        path = "gnnlib",
        name = "GNNlib",
        fix_canonical_url = false),
    MultiDocumenter.MultiDocRef(
        upstream = joinpath(dirname(@__DIR__), "GNNLux", "docs", "build"),
        path = "gnnlux",
        name = "GNNLux",
        fix_canonical_url = false), 
    MultiDocumenter.MultiDocRef(
        upstream = joinpath(dirname(@__DIR__), "tutorials", "docs", "build"),
        path = "tutorials",
        name = "tutorials",
        fix_canonical_url = false),    
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

cp(joinpath(@__DIR__, "logo.svg"),
    joinpath(outpath, "logo.svg"))

@warn "Deploying to GitHub as in DataToolkit" 
outbranch = "branch-multidoc"
has_outbranch = true

if !success(`git checkout --orphan $outbranch`)
    has_outbranch = false
    @info "Creating orphaned branch $outbranch"
    if !success(`git switch --orphan $outbranch`)
        @error "Cannot create new orphaned branch $outbranch."
        exit(1)
    end
else 
    @info "Switched to orphaned branch $outbranch"
end

run(`git add --all`)

if success(`git commit -m 'Aggregate documentation'`)
    @info "Pushing updated documentation."
    run(`git push origin --force $outbranch`)
else
    @info "No changes to aggregated documentation."
end