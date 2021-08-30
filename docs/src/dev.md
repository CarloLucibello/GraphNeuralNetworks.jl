# Developer Notes

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
