# Developer Notes

## Develop Monorepo

GraphNeuralNetworks.jl is package hosted in a monorepo that contains multiple packages. 
The GraphNeuralNetworks.jl package depends on GNNGraphs.jl, also hosted in the same monorepo.

```julia
pkg> activate .

pkg> dev ./GNNGraphs
```

Each PR should update the version number in the Porject.toml file of each involved package if needed by semnatic versioning. For instance, when adding new features GNNGraphs could move from "1.17.5" to "1.18.0-DEV". The "DEV" will be removed when the package is tagged and released.


For generating the documentation locally instead
```
cd docs
julia
```
```julia
(@v1.10) pkg> activate .
  Activating project at `~/.julia/dev/GraphNeuralNetworks/docs`

(docs) pkg> dev ../ ../GNNGraphs/
   Resolving package versions...
  No Changes to `~/.julia/dev/GraphNeuralNetworks/docs/Project.toml`
  No Changes to `~/.julia/dev/GraphNeuralNetworks/docs/Manifest.toml`

julia> include("make.jl")
```
## Benchmarking

You can benchmark the effect on performance of your commits using the script `perf/perf.jl`.

First, checkout and benchmark the master branch:

```julia
julia> include("perf.jl")

julia> df = run_benchmarks()

# observe results
julia> for g in groupby(df, :layer); println(g, "\n"); end

julia> @save "perf_master_20210803_mymachine.jld2" dfmaster=df
```

Now checkout your branch and do the same:

```julia
julia> df = run_benchmarks()

julia> @save "perf_pr_20210803_mymachine.jld2" dfpr=df
```

Finally, compare the results:

```julia
julia> @load "perf_master_20210803_mymachine.jld2"

julia> @load "perf_pr_20210803_mymachine.jld2"

julia> compare(dfpr, dfmaster)
```

## Caching tutorials

Tutorials in GraphNeuralNetworks.jl are written in Pluto and rendered using [DemoCards.jl](https://github.com/JuliaDocs/DemoCards.jl) and [PlutoStaticHTML.jl](https://github.com/rikhuijzer/PlutoStaticHTML.jl). Rendering a Pluto notebook is time and resource-consuming, especially in a CI environment. So we use the [caching functionality](https://huijzer.xyz/PlutoStaticHTML.jl/dev/#Caching) provided by PlutoStaticHTML.jl to reduce CI time.

If you are contributing a new tutorial or making changes to the existing notebook, generate the docs locally before committing/pushing. For caching to work, the cache environment(your local) and the documenter CI should have the same Julia version (e.g. "v1.9.1", also the patch number must match). So use the [documenter CI Julia version](https://github.com/CarloLucibello/GraphNeuralNetworks.jl/blob/master/.github/workflows/docs.yml#L17) for generating docs locally.

```console
julia --version # check julia version before generating docs
julia --project=docs docs/make.jl
```

Note: Use [juliaup](https://github.com/JuliaLang/juliaup) for easy switching of Julia versions.

During the doc generation process, DemoCards.jl stores the cache notebooks in docs/pluto_output. So add any changes made in this folder in your git commit. Remember that every file in this folder is machine-generated and should not be edited manually.

```
git add docs/pluto_output # add generated cache
```

Check the [documenter CI logs](https://github.com/CarloLucibello/GraphNeuralNetworks.jl/actions/workflows/docs.yml) to ensure that it used the local cache:

![](https://user-images.githubusercontent.com/55111154/210061301-c84b7274-9e66-46fd-b272-d45b1c681d00.png)