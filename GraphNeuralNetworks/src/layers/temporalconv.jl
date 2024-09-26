# Adapting Flux.Recur to work with GNNGraphs
function (m::Flux.Recur)(g::GNNGraph, x)
    m.state, y = m.cell(m.state, g, x)
    return y
end
    
function (m::Flux.Recur)(g::GNNGraph, x::AbstractArray{T, 3}) where T
    h = [m(g, x_t) for x_t in Flux.eachlastdim(x)]
    sze = size(h[1])
    reshape(reduce(hcat, h), sze[1], sze[2], length(h))
end

struct TGCNCell <: GNNLayer
    conv::GCNConv
    gru::Flux.GRUv3Cell
    state0
    in::Int
    out::Int
end

Flux.@layer TGCNCell

function TGCNCell(ch::Pair{Int, Int};
                  bias::Bool = true,
                  init = Flux.glorot_uniform,
                  init_state = Flux.zeros32,
                  add_self_loops = false,
                  use_edge_weight = true)
    in, out = ch
    conv = GCNConv(in => out, sigmoid; init, bias, add_self_loops,
                   use_edge_weight)
    gru = Flux.GRUv3Cell(out, out)
    state0 = init_state(out,1)
    return TGCNCell(conv, gru, state0, in,out)
end

function (tgcn::TGCNCell)(h, g::GNNGraph, x::AbstractArray)
    x̃ = tgcn.conv(g, x)
    h, x̃ = tgcn.gru(h, x̃)
    return h, x̃
end

function Base.show(io::IO, tgcn::TGCNCell)
    print(io, "TGCNCell($(tgcn.in) => $(tgcn.out))")
end

"""
    TGCN(in => out; [bias, init, init_state, add_self_loops, use_edge_weight])

Temporal Graph Convolutional Network (T-GCN) recurrent layer from the paper [T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction](https://arxiv.org/pdf/1811.05320.pdf).

Performs a layer of GCNConv to model spatial dependencies, followed by a Gated Recurrent Unit (GRU) cell to model temporal dependencies.

# Arguments

- `in`: Number of input features.
- `out`: Number of output features.
- `bias`: Add learnable bias. Default `true`.
- `init`: Weights' initializer. Default `glorot_uniform`.
- `init_state`: Initial state of the hidden stat of the GRU layer. Default `zeros32`.
- `add_self_loops`: Add self loops to the graph before performing the convolution. Default `false`.
- `use_edge_weight`: If `true`, consider the edge weights in the input graph (if available).
                     If `add_self_loops=true` the new weights will be set to 1. 
                     This option is ignored if the `edge_weight` is explicitly provided in the forward pass.
                     Default `false`.
# Examples

```jldoctest
julia> tgcn = TGCN(2 => 6)
Recur(
  TGCNCell(
    GCNConv(2 => 6, σ),                 # 18 parameters
    GRUv3Cell(6 => 6),                  # 240 parameters
    Float32[0.0; 0.0; … ; 0.0; 0.0;;],  # 6 parameters  (all zero)
    2,
    6,
  ),
)         # Total: 8 trainable arrays, 264 parameters,
          # plus 1 non-trainable, 6 parameters, summarysize 1.492 KiB.

julia> g, x = rand_graph(5, 10), rand(Float32, 2, 5);

julia> y = tgcn(g, x);

julia> size(y)
(6, 5)

julia> Flux.reset!(tgcn);

julia> tgcn(rand_graph(5, 10), rand(Float32, 2, 5, 20)) |> size # batch size of 20
(6, 5, 20)
```

!!! warning "Batch size changes"
    Failing to call `reset!` when the input batch size changes can lead to unexpected behavior.
"""
TGCN(ch; kwargs...) = Flux.Recur(TGCNCell(ch; kwargs...))

Flux.Recur(tgcn::TGCNCell) = Flux.Recur(tgcn, tgcn.state0)

# make TGCN compatible with GNNChain
(l::Flux.Recur{TGCNCell})(g::GNNGraph) = GNNGraph(g, ndata = l(g, node_features(g)))
_applylayer(l::Flux.Recur{TGCNCell}, g::GNNGraph, x) = l(g, x)
_applylayer(l::Flux.Recur{TGCNCell}, g::GNNGraph) = l(g)


"""
    A3TGCN(in => out; [bias, init, init_state, add_self_loops, use_edge_weight])

Attention Temporal Graph Convolutional Network (A3T-GCN) model from the paper [A3T-GCN: Attention Temporal Graph
Convolutional Network for Traffic Forecasting](https://arxiv.org/pdf/2006.11583.pdf).

Performs a TGCN layer, followed by a soft attention layer.

# Arguments

- `in`: Number of input features.
- `out`: Number of output features.
- `bias`: Add learnable bias. Default `true`.
- `init`: Weights' initializer. Default `glorot_uniform`.
- `init_state`: Initial state of the hidden stat of the GRU layer. Default `zeros32`.
- `add_self_loops`: Add self loops to the graph before performing the convolution. Default `false`.
- `use_edge_weight`: If `true`, consider the edge weights in the input graph (if available).
                     If `add_self_loops=true` the new weights will be set to 1. 
                     This option is ignored if the `edge_weight` is explicitly provided in the forward pass.
                     Default `false`.
# Examples

```jldoctest
julia> a3tgcn = A3TGCN(2 => 6)
A3TGCN(2 => 6)

julia> g, x = rand_graph(5, 10), rand(Float32, 2, 5);

julia> y = a3tgcn(g,x);

julia> size(y)
(6, 5)

julia> Flux.reset!(a3tgcn);

julia> y = a3tgcn(rand_graph(5, 10), rand(Float32, 2, 5, 20));

julia> size(y)
(6, 5)
```

!!! warning "Batch size changes"
    Failing to call `reset!` when the input batch size changes can lead to unexpected behavior.
"""
struct A3TGCN <: GNNLayer
    tgcn::Flux.Recur{TGCNCell}
    dense1::Dense
    dense2::Dense
    in::Int
    out::Int
end

Flux.@layer A3TGCN

function A3TGCN(ch::Pair{Int, Int},
                  bias::Bool = true,
                  init = Flux.glorot_uniform,
                  init_state = Flux.zeros32,
                  add_self_loops = false,
                  use_edge_weight = true)
    in, out = ch
    tgcn = TGCN(in => out; bias, init, init_state, add_self_loops, use_edge_weight)
    dense1 = Dense(out, out)
    dense2 = Dense(out, out)
    return A3TGCN(tgcn, dense1, dense2, in, out)
end

function (a3tgcn::A3TGCN)(g::GNNGraph, x::AbstractArray)
    h = a3tgcn.tgcn(g, x)
    e = a3tgcn.dense1(h)
    e = a3tgcn.dense2(e)
    a = softmax(e, dims = 3)
    c = sum(a .* h , dims = 3)
    if length(size(c)) == 3
        c = dropdims(c, dims = 3)
    end
    return c
end

function Base.show(io::IO, a3tgcn::A3TGCN)
    print(io, "A3TGCN($(a3tgcn.in) => $(a3tgcn.out))")
end

struct GConvGRUCell <: GNNLayer
    conv_x_r::ChebConv
    conv_h_r::ChebConv
    conv_x_z::ChebConv
    conv_h_z::ChebConv
    conv_x_h::ChebConv
    conv_h_h::ChebConv
    k::Int
    state0
    in::Int
    out::Int
end

Flux.@layer GConvGRUCell

function GConvGRUCell(ch::Pair{Int, Int}, k::Int, n::Int;
                   bias::Bool = true,
                   init = Flux.glorot_uniform,
                   init_state = Flux.zeros32)
    in, out = ch
    # reset gate
    conv_x_r = ChebConv(in => out, k; bias, init)
    conv_h_r = ChebConv(out => out, k; bias, init)
    # update gate
    conv_x_z = ChebConv(in => out, k; bias, init)
    conv_h_z = ChebConv(out => out, k; bias, init)
    # new gate
    conv_x_h = ChebConv(in => out, k; bias, init)
    conv_h_h = ChebConv(out => out, k; bias, init)
    state0 = init_state(out, n)
    return GConvGRUCell(conv_x_r, conv_h_r, conv_x_z, conv_h_z, conv_x_h, conv_h_h, k, state0, in, out)
end

function (ggru::GConvGRUCell)(h, g::GNNGraph, x)
    r = ggru.conv_x_r(g, x) .+ ggru.conv_h_r(g, h)
    r = Flux.sigmoid_fast(r)
    z = ggru.conv_x_z(g, x) .+ ggru.conv_h_z(g, h)
    z = Flux.sigmoid_fast(z)
    h̃ = ggru.conv_x_h(g, x) .+ ggru.conv_h_h(g, r .* h)
    h̃ = Flux.tanh_fast(h̃)
    h = (1 .- z) .* h̃ .+ z .* h 
    return h, h
end

function Base.show(io::IO, ggru::GConvGRUCell)
    print(io, "GConvGRUCell($(ggru.in) => $(ggru.out))")
end

"""
    GConvGRU(in => out, k, n; [bias, init, init_state])

Graph Convolutional Gated Recurrent Unit (GConvGRU) recurrent layer from the paper [Structured Sequence Modeling with Graph Convolutional Recurrent Networks](https://arxiv.org/pdf/1612.07659).

Performs a layer of ChebConv to model spatial dependencies, followed by a Gated Recurrent Unit (GRU) cell to model temporal dependencies.

# Arguments

- `in`: Number of input features.
- `out`: Number of output features.
- `k`: Chebyshev polynomial order.
- `n`: Number of nodes in the graph.
- `bias`: Add learnable bias. Default `true`.
- `init`: Weights' initializer. Default `glorot_uniform`.
- `init_state`: Initial state of the hidden stat of the GRU layer. Default `zeros32`.

# Examples

```jldoctest
julia> g1, x1 = rand_graph(5, 10), rand(Float32, 2, 5);

julia> ggru = GConvGRU(2 => 5, 2, g1.num_nodes);

julia> y = ggru(g1, x1);

julia> size(y)
(5, 5)

julia> g2, x2 = rand_graph(5, 10), rand(Float32, 2, 5, 30);

julia> z = ggru(g2, x2);

julia> size(z)
(5, 5, 30)
```
""" 
GConvGRU(ch, k, n; kwargs...) = Flux.Recur(GConvGRUCell(ch, k, n; kwargs...))
Flux.Recur(ggru::GConvGRUCell) = Flux.Recur(ggru, ggru.state0)

(l::Flux.Recur{GConvGRUCell})(g::GNNGraph) = GNNGraph(g, ndata = l(g, node_features(g)))
_applylayer(l::Flux.Recur{GConvGRUCell}, g::GNNGraph, x) = l(g, x)
_applylayer(l::Flux.Recur{GConvGRUCell}, g::GNNGraph) = l(g)

struct GConvLSTMCell <: GNNLayer
    conv_x_i::ChebConv
    conv_h_i::ChebConv
    w_i
    b_i
    conv_x_f::ChebConv
    conv_h_f::ChebConv
    w_f
    b_f
    conv_x_c::ChebConv
    conv_h_c::ChebConv
    w_c
    b_c
    conv_x_o::ChebConv
    conv_h_o::ChebConv
    w_o
    b_o
    k::Int
    state0
    in::Int
    out::Int
end

Flux.@layer GConvLSTMCell

function GConvLSTMCell(ch::Pair{Int, Int}, k::Int, n::Int;
                        bias::Bool = true,
                        init = Flux.glorot_uniform,
                        init_state = Flux.zeros32)
    in, out = ch
    # input gate
    conv_x_i = ChebConv(in => out, k; bias, init)
    conv_h_i = ChebConv(out => out, k; bias, init)
    w_i = init(out, 1)
    b_i = bias ? Flux.create_bias(w_i, true, out) : false
    # forget gate
    conv_x_f = ChebConv(in => out, k; bias, init)
    conv_h_f = ChebConv(out => out, k; bias, init)
    w_f = init(out, 1)
    b_f = bias ? Flux.create_bias(w_f, true, out) : false
    # cell state
    conv_x_c = ChebConv(in => out, k; bias, init)
    conv_h_c = ChebConv(out => out, k; bias, init)
    w_c = init(out, 1)
    b_c = bias ? Flux.create_bias(w_c, true, out) : false
    # output gate
    conv_x_o = ChebConv(in => out, k; bias, init)
    conv_h_o = ChebConv(out => out, k; bias, init)
    w_o = init(out, 1)
    b_o = bias ? Flux.create_bias(w_o, true, out) : false
    state0 = (init_state(out, n), init_state(out, n))
    return GConvLSTMCell(conv_x_i, conv_h_i, w_i, b_i,
                         conv_x_f, conv_h_f, w_f, b_f,
                         conv_x_c, conv_h_c, w_c, b_c,
                         conv_x_o, conv_h_o, w_o, b_o,
                         k, state0, in, out)
end

function (gclstm::GConvLSTMCell)((h, c), g::GNNGraph, x)
    # input gate
    i = gclstm.conv_x_i(g, x) .+ gclstm.conv_h_i(g, h) .+ gclstm.w_i .* c .+ gclstm.b_i 
    i = Flux.sigmoid_fast(i)
    # forget gate
    f = gclstm.conv_x_f(g, x) .+ gclstm.conv_h_f(g, h) .+ gclstm.w_f .* c .+ gclstm.b_f
    f = Flux.sigmoid_fast(f)
    # cell state
    c = f .* c .+ i .* Flux.tanh_fast(gclstm.conv_x_c(g, x) .+ gclstm.conv_h_c(g, h) .+ gclstm.w_c .* c .+ gclstm.b_c)
    # output gate
    o = gclstm.conv_x_o(g, x) .+ gclstm.conv_h_o(g, h) .+ gclstm.w_o .* c .+ gclstm.b_o
    o = Flux.sigmoid_fast(o)
    h =  o .* Flux.tanh_fast(c)
    return (h,c), h
end

function Base.show(io::IO, gclstm::GConvLSTMCell)
    print(io, "GConvLSTMCell($(gclstm.in) => $(gclstm.out))")
end

"""
    GConvLSTM(in => out, k, n; [bias, init, init_state])

Graph Convolutional Long Short-Term Memory (GConvLSTM) recurrent layer from the paper [Structured Sequence Modeling with Graph Convolutional Recurrent Networks](https://arxiv.org/pdf/1612.07659). 

Performs a layer of ChebConv to model spatial dependencies, followed by a Long Short-Term Memory (LSTM) cell to model temporal dependencies.

# Arguments

- `in`: Number of input features.
- `out`: Number of output features.
- `k`: Chebyshev polynomial order.
- `n`: Number of nodes in the graph.
- `bias`: Add learnable bias. Default `true`.
- `init`: Weights' initializer. Default `glorot_uniform`.
- `init_state`: Initial state of the hidden stat of the LSTM layer. Default `zeros32`.

# Examples

```jldoctest
julia> g1, x1 = rand_graph(5, 10), rand(Float32, 2, 5);

julia> gclstm = GConvLSTM(2 => 5, 2, g1.num_nodes);

julia> y = gclstm(g1, x1);

julia> size(y)
(5, 5)

julia> g2, x2 = rand_graph(5, 10), rand(Float32, 2, 5, 30);

julia> z = gclstm(g2, x2);

julia> size(z)
(5, 5, 30)
```
"""
GConvLSTM(ch, k, n; kwargs...) = Flux.Recur(GConvLSTMCell(ch, k, n; kwargs...))
Flux.Recur(tgcn::GConvLSTMCell) = Flux.Recur(tgcn, tgcn.state0)

(l::Flux.Recur{GConvLSTMCell})(g::GNNGraph) = GNNGraph(g, ndata = l(g, node_features(g)))
_applylayer(l::Flux.Recur{GConvLSTMCell}, g::GNNGraph, x) = l(g, x)
_applylayer(l::Flux.Recur{GConvLSTMCell}, g::GNNGraph) = l(g)

struct DCGRUCell
    in::Int
    out::Int
    state0
    k::Int
    dconv_u::DConv
    dconv_r::DConv
    dconv_c::DConv
end

Flux.@layer DCGRUCell

function DCGRUCell(ch::Pair{Int,Int}, k::Int, n::Int; bias = true, init = glorot_uniform, init_state = Flux.zeros32)
    in, out = ch
    dconv_u = DConv((in + out) => out, k; bias=bias, init=init)
    dconv_r = DConv((in + out) => out, k; bias=bias, init=init)
    dconv_c = DConv((in + out) => out, k; bias=bias, init=init)
    state0 = init_state(out, n)
    return DCGRUCell(in, out, state0, k, dconv_u, dconv_r, dconv_c)
end

function (dcgru::DCGRUCell)(h, g::GNNGraph, x)
    h̃ = vcat(x, h)
    z = dcgru.dconv_u(g, h̃)
    z = NNlib.sigmoid_fast.(z)
    r = dcgru.dconv_r(g, h̃)
    r = NNlib.sigmoid_fast.(r)
    ĥ = vcat(x, h .* r)
    c = dcgru.dconv_c(g, ĥ)
    c = tanh.(c)
    h = z.* h + (1 .- z) .* c
    return h, h
end

function Base.show(io::IO, dcgru::DCGRUCell)
    print(io, "DCGRUCell($(dcgru.in) => $(dcgru.out), $(dcgru.k))")
end

"""
    DCGRU(in => out, k, n; [bias, init, init_state])

Diffusion Convolutional Recurrent Neural Network (DCGRU) layer from the paper [Diffusion Convolutional Recurrent Neural
Network: Data-driven Traffic Forecasting](https://arxiv.org/pdf/1707.01926).

Performs a Diffusion Convolutional layer to model spatial dependencies, followed by a Gated Recurrent Unit (GRU) cell to model temporal dependencies.

# Arguments

- `in`: Number of input features.
- `out`: Number of output features.
- `k`: Diffusion step.
- `n`: Number of nodes in the graph.
- `bias`: Add learnable bias. Default `true`.
- `init`: Weights' initializer. Default `glorot_uniform`.
- `init_state`: Initial state of the hidden stat of the LSTM layer. Default `zeros32`.

# Examples

```jldoctest
julia> g1, x1 = rand_graph(5, 10), rand(Float32, 2, 5);

julia> dcgru = DCGRU(2 => 5, 2, g1.num_nodes);

julia> y = dcgru(g1, x1);

julia> size(y)
(5, 5)

julia> g2, x2 = rand_graph(5, 10), rand(Float32, 2, 5, 30);

julia> z = dcgru(g2, x2);

julia> size(z)
(5, 5, 30)
```
"""
DCGRU(ch, k, n; kwargs...) = Flux.Recur(DCGRUCell(ch, k, n; kwargs...))
Flux.Recur(dcgru::DCGRUCell) = Flux.Recur(dcgru, dcgru.state0)

(l::Flux.Recur{DCGRUCell})(g::GNNGraph) = GNNGraph(g, ndata = l(g, node_features(g)))
_applylayer(l::Flux.Recur{DCGRUCell}, g::GNNGraph, x) = l(g, x)
_applylayer(l::Flux.Recur{DCGRUCell}, g::GNNGraph) = l(g)

"""
    EvolveGCNO(ch; bias = true, init = glorot_uniform, init_state = Flux.zeros32)

Evolving Graph Convolutional Network (EvolveGCNO) layer from the paper [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](https://arxiv.org/pdf/1902.10191).

Perfoms a Graph Convolutional layer with parameters derived from a Long Short-Term Memory (LSTM) layer across the snapshots of the temporal graph.


# Arguments

- `in`: Number of input features.
- `out`: Number of output features.
- `bias`: Add learnable bias. Default `true`.
- `init`: Weights' initializer. Default `glorot_uniform`.
- `init_state`: Initial state of the hidden stat of the LSTM layer. Default `zeros32`.

# Examples

```jldoctest
julia> tg = TemporalSnapshotsGNNGraph([rand_graph(10,20; ndata = rand(4,10)), rand_graph(10,14; ndata = rand(4,10)), rand_graph(10,22; ndata = rand(4,10))])
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10]
  num_edges: [20, 14, 22]
  num_snapshots: 3

julia> ev = EvolveGCNO(4 => 5)
EvolveGCNO(4 => 5)

julia> size(ev(tg, tg.ndata.x))
(3,)

julia> size(ev(tg, tg.ndata.x)[1])
(5, 10)
```
"""
struct EvolveGCNO
    conv
    W_init
    init_state
    in::Int
    out::Int
    Wf
    Uf
    Bf
    Wi
    Ui
    Bi
    Wo
    Uo
    Bo
    Wc
    Uc
    Bc
end
 
Flux.@functor EvolveGCNO

function EvolveGCNO(ch; bias = true, init = glorot_uniform, init_state = Flux.zeros32)
    in, out = ch
    W = init(out, in)
    conv = GCNConv(ch; bias = bias, init = init)
    Wf = init(out, in)
    Uf = init(out, in)
    Bf = bias ? init(out, in) : nothing
    Wi = init(out, in)
    Ui = init(out, in)
    Bi = bias ? init(out, in) : nothing
    Wo = init(out, in)
    Uo = init(out, in)
    Bo = bias ? init(out, in) : nothing
    Wc = init(out, in)
    Uc = init(out, in)
    Bc = bias ? init(out, in) : nothing
    return EvolveGCNO(conv, W, init_state, in, out, Wf, Uf, Bf, Wi, Ui, Bi, Wo, Uo, Bo, Wc, Uc, Bc)
end

function (egcno::EvolveGCNO)(tg::TemporalSnapshotsGNNGraph, x)
    H = egcno.init_state(egcno.out, egcno.in)
    C = egcno.init_state(egcno.out, egcno.in)
    W = egcno.W_init
    X = map(1:tg.num_snapshots) do i
        F = Flux.sigmoid_fast.(egcno.Wf .* W + egcno.Uf .* H + egcno.Bf)
        I = Flux.sigmoid_fast.(egcno.Wi .* W + egcno.Ui .* H + egcno.Bi)
        O = Flux.sigmoid_fast.(egcno.Wo .* W + egcno.Uo .* H + egcno.Bo)
        C̃ = Flux.tanh_fast.(egcno.Wc .* W + egcno.Uc .* H + egcno.Bc)
        C = F .* C + I .* C̃
        H = O .* tanh_fast.(C)
        W = H
        egcno.conv(tg.snapshots[i], x[i]; conv_weight = H)
    end
    return X
end
 
function Base.show(io::IO, egcno::EvolveGCNO)
    print(io, "EvolveGCNO($(egcno.in) => $(egcno.out))")
end

function (l::GINConv)(tg::TemporalSnapshotsGNNGraph, x::AbstractVector)
    return l.(tg.snapshots, x)
end

function (l::ChebConv)(tg::TemporalSnapshotsGNNGraph, x::AbstractVector)
    return l.(tg.snapshots, x)
end

function (l::GATConv)(tg::TemporalSnapshotsGNNGraph, x::AbstractVector)
    return l.(tg.snapshots, x)
end

function (l::GATv2Conv)(tg::TemporalSnapshotsGNNGraph, x::AbstractVector)
    return l.(tg.snapshots, x)
end

function (l::GatedGraphConv)(tg::TemporalSnapshotsGNNGraph, x::AbstractVector)
    return l.(tg.snapshots, x)
end

function (l::CGConv)(tg::TemporalSnapshotsGNNGraph, x::AbstractVector)
    return l.(tg.snapshots, x)
end

function (l::SGConv)(tg::TemporalSnapshotsGNNGraph, x::AbstractVector)
    return l.(tg.snapshots, x)
end

function (l::TransformerConv)(tg::TemporalSnapshotsGNNGraph, x::AbstractVector)
    return l.(tg.snapshots, x)
end

function (l::GCNConv)(tg::TemporalSnapshotsGNNGraph, x::AbstractVector)
    return l.(tg.snapshots, x)
end

function (l::ResGatedGraphConv)(tg::TemporalSnapshotsGNNGraph, x::AbstractVector)
    return l.(tg.snapshots, x)
end

function (l::SAGEConv)(tg::TemporalSnapshotsGNNGraph, x::AbstractVector)
    return l.(tg.snapshots, x)
end

function (l::GraphConv)(tg::TemporalSnapshotsGNNGraph, x::AbstractVector)
    return l.(tg.snapshots, x)
end
