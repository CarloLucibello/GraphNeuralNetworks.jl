struct HeteroGraphConv
    etypes::Vector{EType}
    layers::Vector{<:GNNLayer}
    aggr::Function
end

Flux.@functor HeteroGraphConv

function HeteroGraphConv(itr; aggr = +)
    etypes = [k[1] for k in itr]
    layers = [k[2] for k in itr]
    return HeteroGraphConv(etypes, layers, aggr)
end

function (hgc::HeteroGraphConv)(g::GNNHeteroGraph, x::NamedTuple)
    function forw(l, et)
        sg = edge_type_subgraph(g, et)
        node1_t, _, node2_t = et
        return l(sg, (x[node1_t], x[node2_t]))
    end
    outs = [forw(l, et) for (l, et) in zip(hgc.layers, hgc.etypes)]
    dst_ntypes = [et[3] for et in hgc.etypes]
    return _reduceby_node_t(hgc.aggr, outs, dst_ntypes)
end

function _reduceby_node_t(aggr, outs, ntypes)
    function _reduce(node_t)
        idxs = findall(x -> x == node_t, ntypes)
        if length(idxs) == 0
            return nothing
        elseif length(idxs) == 1
            return outs[idxs[1]]
        else
            return foldl(aggr, outs[i] for i in idxs)
        end
    end
    vals = [_reduce(node_t) for node_t in ntypes]
    return NamedTuple{tuple(ntypes...)}(vals)
end
