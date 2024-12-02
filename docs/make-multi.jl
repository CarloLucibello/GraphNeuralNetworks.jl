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
        name = "GraphNeuralNetworks.jl",
        fix_canonical_url = false),
    MultiDocumenter.MultiDocRef(
        upstream = joinpath(dirname(@__DIR__), "GNNLux", "docs", "build"),
        path = "gnnlux",
        name = "GNNLux.jl",
        fix_canonical_url = false), 
    MultiDocumenter.MultiDocRef(
        upstream = joinpath(dirname(@__DIR__), "GNNGraphs", "docs", "build"),
        path = "gnngraphs",
        name = "GNNGraphs.jl",
        fix_canonical_url = false),
    MultiDocumenter.MultiDocRef(
        upstream = joinpath(dirname(@__DIR__), "GNNlib", "docs", "build"),
        path = "gnnlib",
        name = "GNNlib.jl",
        fix_canonical_url = false),  
]

outpath = joinpath(@__DIR__, "build")

MultiDocumenter.make(
    outpath,
    docs;
    search_engine = MultiDocumenter.SearchConfig(),
    # search_engine = MultiDocumenter.SearchConfig(
    #     index_versions = ["stable"],
    #     engine = MultiDocumenter.FlexSearch
    # ),
    brand_image = MultiDocumenter.BrandImage("", "logo.svg"),
    rootpath = "/GraphNeuralNetworks.jl/"
)

cp(joinpath(@__DIR__, "logo.svg"),
    joinpath(outpath, "logo.svg"))

if !("PR" in ARGS)
    @warn "Deploying to GitHub as MultiDocumenter" 
    gitroot = normpath(joinpath(@__DIR__, ".."))
    run(`git pull`)

    outbranch = "dep-multidocs"
    has_outbranch = true

    status_output = read(`git status --porcelain docs/Project.toml`, String)
    if !isempty(status_output)
        @info "Restoring docs/Project.toml due to changes."
        run(`git restore docs/Project.toml`)
    else
        @info "No changes detected in docs/Project.toml."
    end

    if !success(`git checkout -f $outbranch`)
        has_outbranch = false
        if !success(`git switch --orphan $outbranch`)
            @error "Cannot create new orphaned branch $outbranch."
            exit(1)
        end
    end

    @info "Cleaning up $gitroot."
    for file in readdir(gitroot; join = true)
        file == "/home/runner/work/GraphNeuralNetworks.jl/GraphNeuralNetworks.jl/docs" && continue
        endswith(file, ".git") && continue
        rm(file; force = true, recursive = true)
    end

    @info "Copying aggregated documentation to $gitroot."
    for file in readdir(outpath)
        cp(joinpath(outpath, file), joinpath(gitroot, file))
    end

    rm("/home/runner/work/GraphNeuralNetworks.jl/GraphNeuralNetworks.jl/docs"; force = true, recursive = true)

    run(`git add .`)
    if success(`git commit -m 'Aggregate documentation'`)
        @info "Pushing updated documentation."
        if has_outbranch
            run(`git push`)
        else
            run(`git push -u origin $outbranch`)
        end
        run(`git checkout master`)
    else
        @info "No changes to aggregated documentation."
    end
end