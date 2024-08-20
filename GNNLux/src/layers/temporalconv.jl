@concrete struct StatefulRecurrentCell <: AbstractExplicitContainerLayer{(:cell,)}     
    cell <: Union{<:Lux.AbstractRecurrentCell, <:GNNContainerLayer}
end

function LuxCore.initialstates(rng::AbstractRNG, r::GNNLux.StatefulRecurrentCell)
    return (cell=LuxCore.initialstates(rng, r.cell), carry=nothing)
end

function (r::StatefulRecurrentCell)(g, x::AbstractMatrix, ps, st::NamedTuple)
    (out, carry), st = applyrecurrentcell(r.cell, g, x, ps, st.cell, st.carry)
    return out, (; cell=st, carry)
end

function (r::StatefulRecurrentCell)(g, x::AbstractVector, ps, st::NamedTuple)
    st, carry = st.cell, st.carry
    for xᵢ in x
        (out, carry), st = applyrecurrentcell(r.cell, g, xᵢ, ps, st, carry)
    end
    return out, (; cell=st, carry)
end

function applyrecurrentcell(l, g, x, ps, st, carry)
    return Lux.apply(l, g, (x, carry), ps, st)
end

LuxCore.apply(m::GNNContainerLayer, g, x, ps, st) = m(g, x, ps, st)

@concrete struct TGCNCell <: GNNContainerLayer{(:conv, :gru)}
    in_dims::Int
    out_dims::Int
    conv
    gru
    init_state::Function
end

function TGCNCell(ch::Pair{Int, Int}; use_bias = true, init_weight = glorot_uniform, init_state = zeros32, init_bias = zeros32, add_self_loops = false, use_edge_weight = true) 
    in_dims, out_dims = ch
    conv = GCNConv(ch, sigmoid; init_weight, init_bias, use_bias, add_self_loops, use_edge_weight, allow_fast_activation= true)
    gru = Lux.GRUCell(out_dims => out_dims; use_bias, init_weight = (init_weight, init_weight, init_weight), init_bias = (init_bias, init_bias, init_bias), init_state = init_state)
    return TGCNCell(in_dims, out_dims, conv, gru, init_state)
end

function (l::TGCNCell)(g, (x, h), ps, st)
    if h === nothing
        h = l.init_state(l.out_dims, 1)
    end
    x̃, stconv = l.conv(g, x, ps.conv, st.conv)
    (h, (h,)), stgru = l.gru((x̃,(h,)), ps.gru,st.gru)
    return  (h, h), (conv=stconv, gru=stgru)
end

LuxCore.outputsize(l::TGCNCell) = (l.out_dims,)
LuxCore.outputsize(l::GNNLux.StatefulRecurrentCell) = (l.cell.out_dims,)

function Base.show(io::IO, tgcn::TGCNCell)
    print(io, "TGCNCell($(tgcn.in_dims) => $(tgcn.out_dims))")
end

TGCN(ch::Pair{Int, Int}; kwargs...) = GNNLux.StatefulRecurrentCell(TGCNCell(ch; kwargs...))

@concrete struct A3TGCN <: GNNContainerLayer{(:tgcn, :dense1, :dense2)}
    in_dims::Int
    out_dims::Int
    tgcn
    dense1
    dense2
end

function A3TGCN(ch::Pair{Int, Int}; use_bias = true, init_weight = glorot_uniform, init_state = zeros32, init_bias = zeros32, add_self_loops = false, use_edge_weight = true) 
    in_dims, out_dims = ch
    tgcn = TGCN(ch; use_bias, init_weight, init_state, init_bias, add_self_loops, use_edge_weight)
    dense1 = Dense(out_dims, out_dims)
    dense2 = Dense(out_dims, out_dims)
    return A3TGCN(in_dims, out_dims, tgcn, dense1, dense2)
end

function (l::A3TGCN)(g, x, ps, st)
    dense1 = StatefulLuxLayer{true}(l.dense1, ps.dense1, _getstate(st, :dense1))
    dense2 = StatefulLuxLayer{true}(l.dense2, ps.dense2, _getstate(st, :dense2))
    h, st = l.tgcn(g, x, ps.tgcn, st.tgcn)
    x = dense1(h)
    x = dense2(x)
    a = NNlib.softmax(x, dims = 3)
    c = sum(a .* h , dims = 3)
    if length(size(c)) == 3
        c = dropdims(c, dims = 3)
    end
    return c, st
end

LuxCore.outputsize(l::A3TGCN) = (l.out_dims,)

function Base.show(io::IO, l::A3TGCN)
    print(io, "A3TGCN($(l.in_dims) => $(l.out_dims))")
end
