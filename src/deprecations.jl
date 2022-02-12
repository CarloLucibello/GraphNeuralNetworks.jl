## Deprecated in v0.2

function compute_message end
function update_node end
function update_edge end

compute_message(l, xi, xj, e) = compute_message(l, xi, xj)
update_node(l, x, m̄) = m̄
update_edge(l, e, m) = e

function propagate(l::GNNLayer, g::GNNGraph, aggr, x, e=nothing)
    @warn """
          Passing a GNNLayer to propagate is deprecated, 
          you should pass the message function directly.
          The new signature is `propagate(f, g, aggr; [xi, xj, e])`.

          The functions `compute_message`, `update_node`,
          and `update_edge` have been deprecated as well. Please
          refer to the documentation.
          """
    m = apply_edges((a...) -> compute_message(l, a...), g, x, x, e)
    m̄ = aggregate_neighbors(g, aggr, m)
    x = update_node(l, x, m̄)
    e = update_edge(l, e, m)
    return x, e
end

## Deprecated in v0.3 

@deprecate copyxj(xi, xj, e)  copy_xj(xi, xj, e) 

@deprecate CGConv(nin::Int, ein::Int, out::Int, args...; kws...) CGConv((nin, ein) => out, args...; kws...)

