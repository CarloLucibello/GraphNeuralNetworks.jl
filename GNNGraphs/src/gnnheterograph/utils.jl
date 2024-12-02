function check_num_nodes(g::GNNHeteroGraph, x::Tuple)
    @assert length(x) == 2
    @assert length(g.etypes) == 1
    nt1, _, nt2 = only(g.etypes)
    if x[1] isa AbstractArray
        @assert size(x[1], ndims(x[1])) == g.num_nodes[nt1]
    end
    if x[2] isa AbstractArray
        @assert size(x[2], ndims(x[2])) == g.num_nodes[nt2] 
    end
    return true
end

function check_num_edges(g::GNNHeteroGraph, e::AbstractArray)
    num_edgs = only(g.num_edges)[2]
    @assert only(num_edgs)==size(e, ndims(e)) "Got $(size(e, ndims(e))) as last dimension size instead of num_edges=$(num_edgs)"
    return true
end
