@concrete struct StatefulRecurrentCell <: AbstractExplicitContainerLayer{(:cell,)}     
    cell <: Union{<:Lux.AbstractRecurrentCell, <:GNNContainerLayer}
end

function initialstates(rng::AbstractRNG, r::StatefulRecurrentCell)
    return (cell=initialstates(rng, r.cell), carry=nothing)
end

function (r::StatefulRecurrentCell)(g, x, ps, st::NamedTuple)
    (out, carry), st_ = applyrecurrentcell(r.cell, g, x, ps, st.cell, st.carry)
    return out, (; cell=st_, carry)
end

function applyrecurrentcell(l, g, x, ps, st, carry)
         return Lux.apply(l, g, (x, carry), ps, st)
end

function applyrecurrentcell(l, g, x, ps, st, ::Nothing)
    return Lux.apply(l, g, x, ps, st)
end

LuxCore.apply(m::GNNContainerLayer, g, x, ps, st) = m(g, x, ps, st)

@concrete struct TGCNCell <: GNNContainerLayer{(:conv, :gru)}
    in_dims::Int
    out_dims::Int
    conv
    gru
end

function TGCNCell(ch::Pair{Int, Int}; use_bias = true, init_weight = glorot_uniform, init_state = zeros32, init_bias = zeros32, add_self_loops = false, use_edge_weight = true) 
    in_dims, out_dims = ch
    conv = GCNConv(ch, sigmoid; init_weight, init_bias, use_bias, add_self_loops, use_edge_weight, allow_fast_activation= true)
    gru = Lux.GRUCell(out_dims => out_dims; use_bias, init_weight = (init_weight, init_weight, init_weight), init_bias = (init_bias, init_bias, init_bias), init_state = init_state)
    return TGCNCell(in_dims, out_dims, conv, gru)
end

LuxCore.outputsize(l::TGCNCell) = (l.out_dims,)

function (l::TGCNCell)(g, x, h, ps, st)
    conv = StatefulLuxLayer{true}(l.conv, ps.conv, _getstate(st, :conv))
    gru = StatefulLuxLayer{true}(l.gru, ps.gru, _getstate(st, :gru))
    #m = (; conv, gru)
    
    x̃, stconv = l.conv(g, x, ps.conv, st.conv)
    (h, (h,)), st = l.gru((x̃,(h,)), ps.gru,st.gru)
    return  (h, (h,)), st
end

function Base.show(io::IO, tgcn::TGCNCell)
    print(io, "TGCNCell($(tgcn.in_dims) => $(tgcn.out_dims))")
end

tgcn = StatefulRecurrentCell(TGCNCell(1 =>3))