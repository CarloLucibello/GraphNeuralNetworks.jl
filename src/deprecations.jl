# Deprecated in v0.1 

@deprecate GINConv(nn; eps=0, aggr=+)  GINConv(nn, eps; aggr)


# Deprecated in v0.2
# TODO check if argument order is exact
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
