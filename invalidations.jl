using SnoopCompileCore

invalidations = @snoopr begin 
    using GraphNeuralNetworks
    using Flux
    # using CUDA
    # using Graphs
    # using Random, Statistics, LinearAlgebra
end

function workload()
    num_graphs = 3
    gs = [rand_graph(5, 10) for _ in 1:num_graphs]
    g = Flux.batch(gs)
    x = rand(Float32, 4, g.num_nodes)
    model = GNNChain(GCNConv(4 => 4, relu), 
                     GCNConv(4 => 4), 
                     GlobalPool(max), 
                     Dense(4, 1))
    y = model(g, x)
    # @assert size(y) == (1, num_graphs)
end

tinf = @snoopi_deep begin
    workload()
end

using SnoopCompile
trees = invalidation_trees(invalidations)
staletrees = precompile_blockers(trees, tinf)

@show length(uinvalidated(invalidations))  # show total invalidations

show(trees[end])  # show the most invalidating method

# Count number of children (number of invalidations per invalidated method)
n_invalidations = map(SnoopCompile.countchildren, trees)

# (optional) plot the number of children per method invalidations
import Plots
Plots.plot(
    1:length(trees),
    n_invalidations;
    markershape=:circle,
    xlabel="i-th method invalidation",
    label="Number of children per method invalidations"
)

# (optional) report invalidations summary
using PrettyTables  # needed for `report_invalidations` to be defined
SnoopCompile.report_invalidations(;
     invalidations,
     process_filename = x -> last(split(x, ".julia/packages/")),
     n_rows = 0,  # no-limit (show all invalidations)
  )