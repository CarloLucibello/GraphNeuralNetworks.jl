struct TGCNCell{C,G} <: GNNLayer
    conv::C
    gru::G
    din::Int
    dout::Int
end

Flux.@layer TGCNCell

function TGCNCell(ch::Pair{Int, Int};
                  bias::Bool = true,
                  init = Flux.glorot_uniform,
                  add_self_loops = false)
    din, dout = ch
    conv = GCNConv(din => dout, sigmoid; init, bias, add_self_loops)
    gru = GRUCell(dout => dout)
    return TGCNCell(conv, gru, din, dout)
end

initialstates(cell::GRUCell) = zeros_like(cell.Wh, size(cell.Wh, 2))
initialstates(cell::TGCNCell) = initialstates(cell.gru)
(cell::TGCNCell)(g::GNNGraph, x::AbstractVecOrMat) = cell(g, x, initialstates(cell))

function (cell::TGCNCell)(g::GNNGraph, x::AbstractVecOrMat, h::AbstractVecOrMat)
    x = cell.conv(g, x)
    h = cell.gru(x, h)
    return h
end

function Base.show(io::IO, cell::TGCNCell)
    print(io, "TGCNCell($(cell.din) => $(cell.dout))")
end

"""
    TGCN(din => dout; [bias, init, add_self_loops])

Temporal Graph Convolutional Network (T-GCN) recurrent layer from the paper [T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction](https://arxiv.org/pdf/1811.05320.pdf).

Performs a layer of GCNConv to model spatial dependencies, followed by a Gated Recurrent Unit (GRU) cell to model temporal dependencies.

# Arguments

- `din`: Number of input features.
- `dout`: Number of output features.
- `bias`: Add learnable bias. Default `true`.
- `init`: Convolution's weights initializer. Default `glorot_uniform`.
- `add_self_loops`: Add self loops to the graph before performing the convolution. Default `false`.

# Forward 

    tgcn(g::GNNGraph, x, [h])

- `g`: The input graph.
- `x`: The input to the TGCN. It should be a matrix size `din x timesteps` or an array of size `din x timesteps x num_nodes`.
- `h`: The initial hidden state of the GRU cell. If given, it is a vector of size `out` or a matrix of size `dout x num_nodes`.
       If not provided, it is assumed to be a vector of zeros.

# Examples

```jldoctest
julia> din, dout = 2, 3;

julia> tgcn = TGCN(din => dout)
TGCN(
  TGCNCell(
    GCNConv(2 => 3, σ),                 # 9 parameters
    GRUCell(3 => 3),                    # 63 parameters
  ),
)                   # Total: 5 arrays, 72 parameters, 560 bytes.

julia> num_nodes = 5; num_edges = 10; timesteps = 4;

julia> g = rand_graph(num_nodes, num_edges);

julia> x = rand(Float32, din, timesteps, num_nodes);

julia> tgcn(g, x) |> size
(3, 4, 5)
```
"""
struct TGCN{C<:TGCNCell} <: GNNLayer
    cell::C
end

Flux.@layer TGCN

TGCN(ch::Pair{Int, Int}; kws...) = TGCN(TGCNCell(ch; kws...))

initialstates(tgcn::TGCN) = initialstates(tgcn.cell)

(tgcn::TGCN)(g::GNNGraph, x) = tgcn(g, x, initialstates(tgcn))

function (tgcn::TGCN)(g::GNNGraph, x::AbstractArray, h)
    @assert ndims(x) == 2 || ndims(x) == 3
    # [x] = [din, timesteps] or [din, timesteps, num_nodes]
    # y = AbstractArray[] # issue https://github.com/JuliaLang/julia/issues/56771
    y = []
    for xt in eachslice(x, dims = 2)
        h = tgcn.cell(g, xt, h)
        y = vcat(y, [h])
    end
    return stack(y, dims = 2) # [dout, timesteps, num_nodes]
end

Base.show(io::IO, tgcn::TGCN) = print(io, "TGCN($(tgcn.cell.din) => $(tgcn.cell.dout))")

######## TO BE PORTED TO FLUX v0.15 from here ############################

# """
#     A3TGCN(din => dout; [bias, init, add_self_loops])

# Attention Temporal Graph Convolutional Network (A3T-GCN) model from the paper [A3T-GCN: Attention Temporal Graph
# Convolutional Network for Traffic Forecasting](https://arxiv.org/pdf/2006.11583.pdf).

# Performs a TGCN layer, followed by a soft attention layer.

# # Arguments

# - `din`: Number of input features.
# - `dout`: Number of output features.
# - `bias`: Add learnable bias. Default `true`.
# - `init`: Convolution's weights initializer. Default `glorot_uniform`.
# - `add_self_loops`: Add self loops to the graph before performing the convolution. Default `false`.

# # Examples

# ```jldoctest
# julia> din, dout = 2, 3;

# julia> model = A3TGCN(din => dout)
# TGCN(
#   TGCNCell(
#     GCNConv(2 => 3, σ),                 # 9 parameters
#     GRUCell(3 => 3),                    # 63 parameters
#   ),
# )                   # Total: 5 arrays, 72 parameters, 560 bytes.

# julia> num_nodes = 5; num_edges = 10; timesteps = 4;

# julia> g = rand_graph(num_nodes, num_edges);

# julia> x = rand(Float32, din, timesteps, num_nodes);

# julia> model(g, x) |> size
# (3, 4, 5)
# ```

# !!! warning "Batch size changes"
#     Failing to call `reset!` when the input batch size changes can lead to unexpected behavior.
# """
# struct A3TGCN <: GNNLayer
#     tgcn::TGCN
#     dense1::Dense
#     dense2::Dense
#     din::Int
#     dout::Int
# end

# Flux.@layer A3TGCN

# function A3TGCN(ch::Pair{Int, Int},
#                   bias::Bool = true,
#                   init = Flux.glorot_uniform,
#                   add_self_loops = false)
#     din, dout = ch
#     tgcn = TGCN(din => dout; bias, init, init_state, add_self_loops)
#     dense1 = Dense(dout => dout)
#     dense2 = Dense(dout => dout)
#     return A3TGCN(tgcn, dense1, dense2, din, dout)
# end

# function (a3tgcn::A3TGCN)(g::GNNGraph, x::AbstractArray, h)
#     h = a3tgcn.tgcn(g, x, h)
#     e = a3tgcn.dense1(h) # WHY NOT RELU?
#     e = a3tgcn.dense2(e)
#     a = softmax(e, dims = 2)
#     c = sum(a .* h , dims = 2)
#     if length(size(c)) == 3
#         c = dropdims(c, dims = 2)
#     end
#     return c
# end

# function Base.show(io::IO, a3tgcn::A3TGCN)
#     print(io, "A3TGCN($(a3tgcn.din) => $(a3tgcn.dout))")
# end

# struct GConvGRUCell <: GNNLayer
#     conv_x_r::ChebConv
#     conv_h_r::ChebConv
#     conv_x_z::ChebConv
#     conv_h_z::ChebConv
#     conv_x_h::ChebConv
#     conv_h_h::ChebConv
#     k::Int
#     state0
#     in::Int
#     out::Int
# end

# Flux.@layer GConvGRUCell

# function GConvGRUCell(ch::Pair{Int, Int}, k::Int, n::Int;
#                    bias::Bool = true,
#                    init = Flux.glorot_uniform,
#                    init_state = Flux.zeros32)
#     in, out = ch
#     # reset gate
#     conv_x_r = ChebConv(in => out, k; bias, init)
#     conv_h_r = ChebConv(out => out, k; bias, init)
#     # update gate
#     conv_x_z = ChebConv(in => out, k; bias, init)
#     conv_h_z = ChebConv(out => out, k; bias, init)
#     # new gate
#     conv_x_h = ChebConv(in => out, k; bias, init)
#     conv_h_h = ChebConv(out => out, k; bias, init)
#     state0 = init_state(out, n)
#     return GConvGRUCell(conv_x_r, conv_h_r, conv_x_z, conv_h_z, conv_x_h, conv_h_h, k, state0, in, out)
# end

# function (ggru::GConvGRUCell)(h, g::GNNGraph, x)
#     r = ggru.conv_x_r(g, x) .+ ggru.conv_h_r(g, h)
#     r = Flux.sigmoid_fast(r)
#     z = ggru.conv_x_z(g, x) .+ ggru.conv_h_z(g, h)
#     z = Flux.sigmoid_fast(z)
#     h̃ = ggru.conv_x_h(g, x) .+ ggru.conv_h_h(g, r .* h)
#     h̃ = Flux.tanh_fast(h̃)
#     h = (1 .- z) .* h̃ .+ z .* h 
#     return h, h
# end

# function Base.show(io::IO, ggru::GConvGRUCell)
#     print(io, "GConvGRUCell($(ggru.in) => $(ggru.out))")
# end

# """
#     GConvGRU(in => out, k, n; [bias, init, init_state])

# Graph Convolutional Gated Recurrent Unit (GConvGRU) recurrent layer from the paper [Structured Sequence Modeling with Graph Convolutional Recurrent Networks](https://arxiv.org/pdf/1612.07659).

# Performs a layer of ChebConv to model spatial dependencies, followed by a Gated Recurrent Unit (GRU) cell to model temporal dependencies.

# # Arguments

# - `in`: Number of input features.
# - `out`: Number of output features.
# - `k`: Chebyshev polynomial order.
# - `n`: Number of nodes in the graph.
# - `bias`: Add learnable bias. Default `true`.
# - `init`: Weights' initializer. Default `glorot_uniform`.
# - `init_state`: Initial state of the hidden stat of the GRU layer. Default `zeros32`.

# # Examples

# ```jldoctest
# julia> g1, x1 = rand_graph(5, 10), rand(Float32, 2, 5);

# julia> ggru = GConvGRU(2 => 5, 2, g1.num_nodes);

# julia> y = ggru(g1, x1);

# julia> size(y)
# (5, 5)

# julia> g2, x2 = rand_graph(5, 10), rand(Float32, 2, 5, 30);

# julia> z = ggru(g2, x2);

# julia> size(z)
# (5, 5, 30)
# ```
# """ 
# # GConvGRU(ch, k, n; kwargs...) = Flux.Recur(GConvGRUCell(ch, k, n; kwargs...))
# # Flux.Recur(ggru::GConvGRUCell) = Flux.Recur(ggru, ggru.state0)

# # (l::Flux.Recur{GConvGRUCell})(g::GNNGraph) = GNNGraph(g, ndata = l(g, node_features(g)))
# # _applylayer(l::Flux.Recur{GConvGRUCell}, g::GNNGraph, x) = l(g, x)
# # _applylayer(l::Flux.Recur{GConvGRUCell}, g::GNNGraph) = l(g)

# struct GConvLSTMCell <: GNNLayer
#     conv_x_i::ChebConv
#     conv_h_i::ChebConv
#     w_i
#     b_i
#     conv_x_f::ChebConv
#     conv_h_f::ChebConv
#     w_f
#     b_f
#     conv_x_c::ChebConv
#     conv_h_c::ChebConv
#     w_c
#     b_c
#     conv_x_o::ChebConv
#     conv_h_o::ChebConv
#     w_o
#     b_o
#     k::Int
#     state0
#     in::Int
#     out::Int
# end

# Flux.@layer GConvLSTMCell

# function GConvLSTMCell(ch::Pair{Int, Int}, k::Int, n::Int;
#                         bias::Bool = true,
#                         init = Flux.glorot_uniform,
#                         init_state = Flux.zeros32)
#     in, out = ch
#     # input gate
#     conv_x_i = ChebConv(in => out, k; bias, init)
#     conv_h_i = ChebConv(out => out, k; bias, init)
#     w_i = init(out, 1)
#     b_i = bias ? Flux.create_bias(w_i, true, out) : false
#     # forget gate
#     conv_x_f = ChebConv(in => out, k; bias, init)
#     conv_h_f = ChebConv(out => out, k; bias, init)
#     w_f = init(out, 1)
#     b_f = bias ? Flux.create_bias(w_f, true, out) : false
#     # cell state
#     conv_x_c = ChebConv(in => out, k; bias, init)
#     conv_h_c = ChebConv(out => out, k; bias, init)
#     w_c = init(out, 1)
#     b_c = bias ? Flux.create_bias(w_c, true, out) : false
#     # output gate
#     conv_x_o = ChebConv(in => out, k; bias, init)
#     conv_h_o = ChebConv(out => out, k; bias, init)
#     w_o = init(out, 1)
#     b_o = bias ? Flux.create_bias(w_o, true, out) : false
#     state0 = (init_state(out, n), init_state(out, n))
#     return GConvLSTMCell(conv_x_i, conv_h_i, w_i, b_i,
#                          conv_x_f, conv_h_f, w_f, b_f,
#                          conv_x_c, conv_h_c, w_c, b_c,
#                          conv_x_o, conv_h_o, w_o, b_o,
#                          k, state0, in, out)
# end

# function (gclstm::GConvLSTMCell)((h, c), g::GNNGraph, x)
#     # input gate
#     i = gclstm.conv_x_i(g, x) .+ gclstm.conv_h_i(g, h) .+ gclstm.w_i .* c .+ gclstm.b_i 
#     i = Flux.sigmoid_fast(i)
#     # forget gate
#     f = gclstm.conv_x_f(g, x) .+ gclstm.conv_h_f(g, h) .+ gclstm.w_f .* c .+ gclstm.b_f
#     f = Flux.sigmoid_fast(f)
#     # cell state
#     c = f .* c .+ i .* Flux.tanh_fast(gclstm.conv_x_c(g, x) .+ gclstm.conv_h_c(g, h) .+ gclstm.w_c .* c .+ gclstm.b_c)
#     # output gate
#     o = gclstm.conv_x_o(g, x) .+ gclstm.conv_h_o(g, h) .+ gclstm.w_o .* c .+ gclstm.b_o
#     o = Flux.sigmoid_fast(o)
#     h =  o .* Flux.tanh_fast(c)
#     return (h,c), h
# end

# function Base.show(io::IO, gclstm::GConvLSTMCell)
#     print(io, "GConvLSTMCell($(gclstm.in) => $(gclstm.out))")
# end

# """
#     GConvLSTM(in => out, k, n; [bias, init, init_state])

# Graph Convolutional Long Short-Term Memory (GConvLSTM) recurrent layer from the paper [Structured Sequence Modeling with Graph Convolutional Recurrent Networks](https://arxiv.org/pdf/1612.07659). 

# Performs a layer of ChebConv to model spatial dependencies, followed by a Long Short-Term Memory (LSTM) cell to model temporal dependencies.

# # Arguments

# - `in`: Number of input features.
# - `out`: Number of output features.
# - `k`: Chebyshev polynomial order.
# - `n`: Number of nodes in the graph.
# - `bias`: Add learnable bias. Default `true`.
# - `init`: Weights' initializer. Default `glorot_uniform`.
# - `init_state`: Initial state of the hidden stat of the LSTM layer. Default `zeros32`.

# # Examples

# ```jldoctest
# julia> g1, x1 = rand_graph(5, 10), rand(Float32, 2, 5);

# julia> gclstm = GConvLSTM(2 => 5, 2, g1.num_nodes);

# julia> y = gclstm(g1, x1);

# julia> size(y)
# (5, 5)

# julia> g2, x2 = rand_graph(5, 10), rand(Float32, 2, 5, 30);

# julia> z = gclstm(g2, x2);

# julia> size(z)
# (5, 5, 30)
# ```
# """
# # GConvLSTM(ch, k, n; kwargs...) = Flux.Recur(GConvLSTMCell(ch, k, n; kwargs...))
# # Flux.Recur(tgcn::GConvLSTMCell) = Flux.Recur(tgcn, tgcn.state0)

# # (l::Flux.Recur{GConvLSTMCell})(g::GNNGraph) = GNNGraph(g, ndata = l(g, node_features(g)))
# # _applylayer(l::Flux.Recur{GConvLSTMCell}, g::GNNGraph, x) = l(g, x)
# # _applylayer(l::Flux.Recur{GConvLSTMCell}, g::GNNGraph) = l(g)

# struct DCGRUCell
#     in::Int
#     out::Int
#     state0
#     k::Int
#     dconv_u::DConv
#     dconv_r::DConv
#     dconv_c::DConv
# end

# Flux.@layer DCGRUCell

# function DCGRUCell(ch::Pair{Int,Int}, k::Int, n::Int; bias = true, init = glorot_uniform, init_state = Flux.zeros32)
#     in, out = ch
#     dconv_u = DConv((in + out) => out, k; bias=bias, init=init)
#     dconv_r = DConv((in + out) => out, k; bias=bias, init=init)
#     dconv_c = DConv((in + out) => out, k; bias=bias, init=init)
#     state0 = init_state(out, n)
#     return DCGRUCell(in, out, state0, k, dconv_u, dconv_r, dconv_c)
# end

# function (dcgru::DCGRUCell)(h, g::GNNGraph, x)
#     h̃ = vcat(x, h)
#     z = dcgru.dconv_u(g, h̃)
#     z = NNlib.sigmoid_fast.(z)
#     r = dcgru.dconv_r(g, h̃)
#     r = NNlib.sigmoid_fast.(r)
#     ĥ = vcat(x, h .* r)
#     c = dcgru.dconv_c(g, ĥ)
#     c = tanh.(c)
#     h = z.* h + (1 .- z) .* c
#     return h, h
# end

# function Base.show(io::IO, dcgru::DCGRUCell)
#     print(io, "DCGRUCell($(dcgru.in) => $(dcgru.out), $(dcgru.k))")
# end

# """
#     DCGRU(in => out, k, n; [bias, init, init_state])

# Diffusion Convolutional Recurrent Neural Network (DCGRU) layer from the paper [Diffusion Convolutional Recurrent Neural
# Network: Data-driven Traffic Forecasting](https://arxiv.org/pdf/1707.01926).

# Performs a Diffusion Convolutional layer to model spatial dependencies, followed by a Gated Recurrent Unit (GRU) cell to model temporal dependencies.

# # Arguments

# - `in`: Number of input features.
# - `out`: Number of output features.
# - `k`: Diffusion step.
# - `n`: Number of nodes in the graph.
# - `bias`: Add learnable bias. Default `true`.
# - `init`: Weights' initializer. Default `glorot_uniform`.
# - `init_state`: Initial state of the hidden stat of the LSTM layer. Default `zeros32`.

# # Examples

# ```jldoctest
# julia> g1, x1 = rand_graph(5, 10), rand(Float32, 2, 5);

# julia> dcgru = DCGRU(2 => 5, 2, g1.num_nodes);

# julia> y = dcgru(g1, x1);

# julia> size(y)
# (5, 5)

# julia> g2, x2 = rand_graph(5, 10), rand(Float32, 2, 5, 30);

# julia> z = dcgru(g2, x2);

# julia> size(z)
# (5, 5, 30)
# ```
# """
# # DCGRU(ch, k, n; kwargs...) = Flux.Recur(DCGRUCell(ch, k, n; kwargs...))
# # Flux.Recur(dcgru::DCGRUCell) = Flux.Recur(dcgru, dcgru.state0)

# # (l::Flux.Recur{DCGRUCell})(g::GNNGraph) = GNNGraph(g, ndata = l(g, node_features(g)))
# # _applylayer(l::Flux.Recur{DCGRUCell}, g::GNNGraph, x) = l(g, x)
# # _applylayer(l::Flux.Recur{DCGRUCell}, g::GNNGraph) = l(g)

# """
#     EvolveGCNO(ch; bias = true, init = glorot_uniform, init_state = Flux.zeros32)

# Evolving Graph Convolutional Network (EvolveGCNO) layer from the paper [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](https://arxiv.org/pdf/1902.10191).

# Perfoms a Graph Convolutional layer with parameters derived from a Long Short-Term Memory (LSTM) layer across the snapshots of the temporal graph.


# # Arguments

# - `in`: Number of input features.
# - `out`: Number of output features.
# - `bias`: Add learnable bias. Default `true`.
# - `init`: Weights' initializer. Default `glorot_uniform`.
# - `init_state`: Initial state of the hidden stat of the LSTM layer. Default `zeros32`.

# # Examples

# ```jldoctest
# julia> tg = TemporalSnapshotsGNNGraph([rand_graph(10,20; ndata = rand(4,10)), rand_graph(10,14; ndata = rand(4,10)), rand_graph(10,22; ndata = rand(4,10))])
# TemporalSnapshotsGNNGraph:
#   num_nodes: [10, 10, 10]
#   num_edges: [20, 14, 22]
#   num_snapshots: 3

# julia> ev = EvolveGCNO(4 => 5)
# EvolveGCNO(4 => 5)

# julia> size(ev(tg, tg.ndata.x))
# (3,)

# julia> size(ev(tg, tg.ndata.x)[1])
# (5, 10)
# ```
# """
# struct EvolveGCNO
#     conv
#     W_init
#     init_state
#     in::Int
#     out::Int
#     Wf
#     Uf
#     Bf
#     Wi
#     Ui
#     Bi
#     Wo
#     Uo
#     Bo
#     Wc
#     Uc
#     Bc
# end
 
# function EvolveGCNO(ch; bias = true, init = glorot_uniform, init_state = Flux.zeros32)
#     in, out = ch
#     W = init(out, in)
#     conv = GCNConv(ch; bias = bias, init = init)
#     Wf = init(out, in)
#     Uf = init(out, in)
#     Bf = bias ? init(out, in) : nothing
#     Wi = init(out, in)
#     Ui = init(out, in)
#     Bi = bias ? init(out, in) : nothing
#     Wo = init(out, in)
#     Uo = init(out, in)
#     Bo = bias ? init(out, in) : nothing
#     Wc = init(out, in)
#     Uc = init(out, in)
#     Bc = bias ? init(out, in) : nothing
#     return EvolveGCNO(conv, W, init_state, in, out, Wf, Uf, Bf, Wi, Ui, Bi, Wo, Uo, Bo, Wc, Uc, Bc)
# end

# function (egcno::EvolveGCNO)(tg::TemporalSnapshotsGNNGraph, x)
#     H = egcno.init_state(egcno.out, egcno.in)
#     C = egcno.init_state(egcno.out, egcno.in)
#     W = egcno.W_init
#     X = map(1:tg.num_snapshots) do i
#         F = Flux.sigmoid_fast.(egcno.Wf .* W + egcno.Uf .* H + egcno.Bf)
#         I = Flux.sigmoid_fast.(egcno.Wi .* W + egcno.Ui .* H + egcno.Bi)
#         O = Flux.sigmoid_fast.(egcno.Wo .* W + egcno.Uo .* H + egcno.Bo)
#         C̃ = Flux.tanh_fast.(egcno.Wc .* W + egcno.Uc .* H + egcno.Bc)
#         C = F .* C + I .* C̃
#         H = O .* tanh_fast.(C)
#         W = H
#         egcno.conv(tg.snapshots[i], x[i]; conv_weight = H)
#     end
#     return X
# end
 
# function Base.show(io::IO, egcno::EvolveGCNO)
#     print(io, "EvolveGCNO($(egcno.in) => $(egcno.out))")
# end
