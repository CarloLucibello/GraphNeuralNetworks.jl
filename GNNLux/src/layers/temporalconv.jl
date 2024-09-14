@concrete struct StatefulRecurrentCell <: AbstractLuxContainerLayer{(:cell,)}     
    cell <: Union{<:Lux.AbstractRecurrentCell, <:GNNContainerLayer}
end

function LuxCore.initialstates(rng::AbstractRNG, r::GNNLux.StatefulRecurrentCell)
    return (cell=LuxCore.initialstates(rng, r.cell), carry=nothing)
end

function (r::StatefulRecurrentCell)(g, x::AbstractMatrix, ps, st::NamedTuple)
    (out, carry), st = applyrecurrentcell(r.cell, g, x, ps.cell, st.cell, st.carry)
    return out, (; cell=st, carry)
end

function (r::StatefulRecurrentCell)(g, x::AbstractVector, ps, st::NamedTuple)
    st, carry = st.cell, st.carry
    for xᵢ in x
        (out, carry), st = applyrecurrentcell(r.cell, g, xᵢ, ps.cell, st, carry)
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
    conv = GCNConv(ch, sigmoid; init_weight, init_bias, use_bias, add_self_loops, use_edge_weight)
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

@concrete struct GConvGRUCell <: GNNContainerLayer{(:conv_x_r, :conv_h_r, :conv_x_z, :conv_h_z, :conv_x_h, :conv_h_h)}
    in_dims::Int
    out_dims::Int
    k::Int
    conv_x_r
    conv_h_r
    conv_x_z
    conv_h_z
    conv_x_h
    conv_h_h
    init_state::Function
end

function GConvGRUCell(ch::Pair{Int, Int}, k::Int; use_bias = true, init_weight = glorot_uniform, init_state = zeros32, init_bias = zeros32) 
    in_dims, out_dims = ch
    #reset gate
    conv_x_r = ChebConv(in_dims => out_dims, k; use_bias, init_weight, init_bias)
    conv_h_r = ChebConv(out_dims => out_dims, k; use_bias, init_weight, init_bias)
    #update gate
    conv_x_z = ChebConv(in_dims => out_dims, k; use_bias, init_weight, init_bias)
    conv_h_z = ChebConv(out_dims => out_dims, k; use_bias, init_weight, init_bias)
    #hidden state
    conv_x_h = ChebConv(in_dims => out_dims, k; use_bias, init_weight, init_bias)
    conv_h_h = ChebConv(out_dims => out_dims, k; use_bias, init_weight, init_bias)
    return GConvGRUCell(in_dims, out_dims, k, conv_x_r, conv_h_r, conv_x_z, conv_h_z, conv_x_h, conv_h_h, init_state)
end

function (l::GConvGRUCell)(g, (x, h), ps, st)
    if h === nothing
        h = l.init_state(l.out_dims, g.num_nodes)
    end
    xr, st_conv_xr =  l.conv_x_r(g, x, ps.conv_x_r, st.conv_x_r)
    hr, st_conv_hr =  l.conv_h_r(g, h, ps.conv_h_r, st.conv_h_r)
    r = xr .+ hr
    r = NNlib.sigmoid_fast(r)
    xz, st_conv_x_z = l.conv_x_z(g, x, ps.conv_x_z, st.conv_x_z)
    hz, st_conv_h_z =  l.conv_h_z(g, h, ps.conv_h_z, st.conv_h_z)
    z = xz .+ hz
    z = NNlib.sigmoid_fast(z)
    xh, st_conv_x_h =  l.conv_x_h(g, x, ps.conv_x_h, st.conv_x_h)
    hh, st_conv_h_h =  l.conv_h_h(g, r .* h, ps.conv_h_h, st.conv_h_h)
    h̃ = xh .+ hh
    h̃ = NNlib.tanh_fast(h)
    h = (1 .- z).* h̃ + z.* h
    return (h, h), (conv_x_r = st_conv_xr, conv_h_r = st_conv_hr, conv_x_z = st_conv_x_z, conv_h_z = st_conv_h_z, conv_x_h = st_conv_x_h, conv_h_h = st_conv_h_h)
end

function Base.show(io::IO, l::GConvGRUCell)
    print(io, "GConvGRUCell($(l.in_dims) => $(l.out_dims))")
end

LuxCore.outputsize(l::GConvGRUCell) = (l.out_dims,)

GConvGRU(ch::Pair{Int, Int}, k::Int; kwargs...) = GNNLux.StatefulRecurrentCell(GConvGRUCell(ch, k; kwargs...))

@concrete struct GConvLSTMCell <: GNNContainerLayer{(:conv_x_i, :conv_h_i, :dense_i, :conv_x_f, :conv_h_f, :dense_f, :conv_x_c, :conv_h_c, :dense_c, :conv_x_o, :conv_h_o, :dense_o)}
    in_dims::Int
    out_dims::Int
    k::Int
    conv_x_i
    conv_h_i
    dense_i
    conv_x_f
    conv_h_f
    dense_f
    conv_x_c
    conv_h_c
    dense_c
    conv_x_o
    conv_h_o
    dense_o
    init_state::Function
end

function GConvLSTMCell(ch::Pair{Int, Int}, k::Int; use_bias = true, init_weight = glorot_uniform, init_state = zeros32, init_bias = zeros32) 
    in_dims, out_dims = ch
    #input gate
    conv_x_i = ChebConv(in_dims => out_dims, k; use_bias, init_weight, init_bias)
    conv_h_i = ChebConv(out_dims => out_dims, k; use_bias, init_weight, init_bias)
    dense_i = Dense(out_dims, 1; use_bias, init_weight, init_bias)
    #forget gate
    conv_x_f = ChebConv(in_dims => out_dims, k; use_bias, init_weight, init_bias)
    conv_h_f = ChebConv(out_dims => out_dims, k; use_bias, init_weight, init_bias)
    dense_f = Dense(out_dims, 1; use_bias, init_weight, init_bias)
    #cell gate
    conv_x_c = ChebConv(in_dims => out_dims, k; use_bias, init_weight, init_bias)
    conv_h_c = ChebConv(out_dims => out_dims, k; use_bias, init_weight, init_bias)
    dense_c = Dense(out_dims, 1; use_bias, init_weight, init_bias)
    #output gate
    conv_x_o = ChebConv(in_dims => out_dims, k; use_bias, init_weight, init_bias)
    conv_h_o = ChebConv(out_dims => out_dims, k; use_bias, init_weight, init_bias)
    dense_o = Dense(out_dims, 1; use_bias, init_weight, init_bias)
    return GConvLSTMCell(in_dims, out_dims, k, conv_x_i, conv_h_i, dense_i, conv_x_f, conv_h_f, dense_f, conv_x_c, conv_h_c, dense_c, conv_x_o, conv_h_o, dense_o, init_state)
end

function (l::GConvLSTMCell)(g, (x, m), ps, st)
    if m === nothing
        h = l.init_state(l.out_dims, g.num_nodes)
        c = l.init_state(l.out_dims, g.num_nodes)
    else 
        h, c = m
    end

    dense_i = StatefulLuxLayer{true}(l.dense_i, ps.dense_i, _getstate(st, :dense_i))
    dense_f = StatefulLuxLayer{true}(l.dense_f, ps.dense_f, _getstate(st, :dense_f))
    dense_c = StatefulLuxLayer{true}(l.dense_c, ps.dense_c, _getstate(st, :dense_c))
    dense_o = StatefulLuxLayer{true}(l.dense_o, ps.dense_o, _getstate(st, :dense_o))
    
    xi, st_conv_x_i = l.conv_x_i(g, x, ps.conv_x_i, st.conv_x_i)
    hi, st_conv_h_i = l.conv_h_i(g, h, ps.conv_h_i, st.conv_h_i)
    i = xi .+ hi .+ dense_i(c)
    i = NNlib.sigmoid_fast(i)
   
    xf, st_conv_x_f = l.conv_x_f(g, x, ps.conv_x_f, st.conv_x_f)
    hf, st_conv_h_f = l.conv_h_f(g, h, ps.conv_h_f, st.conv_h_f)
    f = xf .+ hf .+ dense_f(c)
    f =  NNlib.sigmoid_fast(f)
    
    xc, st_conv_x_c = l.conv_x_c(g, x, ps.conv_x_c, st.conv_x_c)
    hc, st_conv_h_c = l.conv_h_c(g, h, ps.conv_h_c, st.conv_h_c)
    c = f .* c + i.* NNlib.tanh_fast(xc .+ hc .+ dense_c(c))
    
    xo, st_conv_x_o = l.conv_x_o(g, x, ps.conv_x_o, st.conv_x_o)
    ho, st_conv_h_o = l.conv_h_o(g, h, ps.conv_h_o, st.conv_h_o)
    o = xo .+ ho .+ dense_o(c)
    o = NNlib.sigmoid_fast(o)
    h = o.* NNlib.tanh_fast(c)
    return (h, (h, c)), (conv_x_i = st_conv_x_i, conv_h_i = st_conv_h_i, conv_x_f = st_conv_x_f, conv_h_f = st_conv_h_f, conv_x_c = st_conv_x_c, conv_h_c = st_conv_h_c, conv_x_o = st_conv_x_o, conv_h_o = st_conv_h_o)
end

function Base.show(io::IO, l::GConvLSTMCell)
    print(io, "GConvLSTMCell($(l.in_dims) => $(l.out_dims))")
end

LuxCore.outputsize(l::GConvLSTMCell) = (l.out_dims,)

GConvLSTM(ch::Pair{Int, Int}, k::Int; kwargs...) = GNNLux.StatefulRecurrentCell(GConvLSTMCell(ch, k; kwargs...))

@concrete struct DCGRUCell <: GNNContainerLayer{(:dconv_u, :dconv_r, :dconv_c)}
    in_dims::Int
    out_dims::Int
    k::Int
    dconv_u
    dconv_r
    dconv_c
    init_state::Function
end

function DCGRUCell(ch::Pair{Int, Int}, k::Int; use_bias = true, init_weight = glorot_uniform, init_state = zeros32, init_bias = zeros32) 
    in_dims, out_dims = ch
    dconv_u = DConv((in_dims + out_dims) => out_dims, k; use_bias = use_bias, init_weight = init_weight, init_bias = init_bias)
    dconv_r = DConv((in_dims + out_dims) => out_dims, k; use_bias = use_bias, init_weight = init_weight, init_bias = init_bias)
    dconv_c = DConv((in_dims + out_dims) => out_dims, k; use_bias = use_bias, init_weight = init_weight, init_bias = init_bias)
    return DCGRUCell(in_dims, out_dims, k, dconv_u, dconv_r, dconv_c, init_state)
end

function (l::DCGRUCell)(g, (x, h), ps, st)
    if h === nothing
        h = l.init_state(l.out_dims, g.num_nodes)
    end
    h̃ = vcat(x, h)
    z, st_dconv_u = l.dconv_u(g, h̃, ps.dconv_u, st.dconv_u)
    z = NNlib.sigmoid_fast.(z)
    r, st_dconv_r = l.dconv_r(g, h̃, ps.dconv_r, st.dconv_r)
    r = NNlib.sigmoid_fast.(r)
    ĥ = vcat(x, h .* r)
    c, st_dconv_c = l.dconv_c(g, ĥ, ps.dconv_c, st.dconv_c)
    c = NNlib.tanh_fast.(c)
    h = z.* h + (1 .- z).* c
    return (h, h), (dconv_u = st_dconv_u, dconv_r = st_dconv_r, dconv_c = st_dconv_c)
end

function Base.show(io::IO, l::DCGRUCell)
    print(io, "DCGRUCell($(l.in_dims) => $(l.out_dims))")
end

LuxCore.outputsize(l::DCGRUCell) = (l.out_dims,)

DCGRU(ch::Pair{Int, Int}, k::Int; kwargs...) = GNNLux.StatefulRecurrentCell(DCGRUCell(ch, k; kwargs...))

