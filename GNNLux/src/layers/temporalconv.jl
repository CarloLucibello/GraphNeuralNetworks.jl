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

function (l::TGCNCell)(h, g, x)
    conv = StatefulLuxLayer{true}(l.conv, ps.conv, _getstate(st, :conv))
    gru = StatefulLuxLayer{true}(l.gru, ps.gru, _getstate(st, :gru))
    m = (; conv, gru)
    return GNNlib.tgcn_conv(m, h, g, x)
end

function Base.show(io::IO, tgcn::TGCNCell)
    print(io, "TGCNCell($(tgcn.in_dims) => $(tgcn.out_dims))")
end