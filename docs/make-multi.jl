using MultiDocumenter

clonedir = ("--temp" in ARGS) ? mktempdir() : joinpath(@__DIR__, "clones")
outpath = mktempdir()
@info """
Cloning packages into: $(clonedir)
Building aggregate site into: $(outpath)
"""


docs = [
    MultiDocumenter.MultiDocRef(
        upstream = joinpath(clonedir,"GraphNeuralNetworks", "docs", "build"),
        path = "GraphNeuralNetworks",
        name = "GraphNeuralNetworks",
        fix_canonical_url = false),
    MultiDocumenter.MultiDocRef(
        upstream = joinpath(clonedir, "GNNGraphs", "docs", "build"),
        path = "GNNGraphs",
        name = "GNNGraphs",
        fix_canonical_url = false),
    MultiDocumenter.MultiDocRef(
        upstream = joinpath(clonedir, "GNNlib", "docs", "build"),
        path = "GNNlib",
        name = "GNNlib",
        fix_canonical_url = false),
    MultiDocumenter.MultiDocRef(
        upstream = joinpath(clonedir, "GNNLux", "docs", "build"),
        path = "GNNLux",
        name = "GNNLux",
        fix_canonical_url = false), 
    MultiDocumenter.MultiDocRef(
        upstream = joinpath(clonedir, "tutorials", "docs", "build"),
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


@warn "Deploying to GitHub" 
gitroot = normpath(joinpath(@__DIR__, ".."))

run(`git pull`)
outbranch = "test-branch"
has_outbranch = true
if !success(`git checkout $outbranch`)
    has_outbranch = false
    if !success(`git switch --orphan $outbranch`)
        @error "Cannot create new orphaned branch $outbranch."
        exit(1)
    end
end
for file in readdir(gitroot; join = true)
    endswith(file, ".git") && continue
    rm(file; force = true, recursive = true)
end
for file in readdir(outpath)
    cp(joinpath(outpath, file), joinpath(gitroot, file))
end
run(`git add .`)
if success(`git commit -m 'Aggregate documentation'`)
    @info "Pushing updated documentation."
    if has_outbranch
        run(`git push`)
    else
        run(`git push -u origin $outbranch`)
    end
    run(`git checkout main`)
else
    @info "No changes to aggregated documentation."
end
