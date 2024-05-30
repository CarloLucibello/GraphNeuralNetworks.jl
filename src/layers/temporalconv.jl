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

Flux.@functor TGCNCell

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

Flux.@functor A3TGCN

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

Flux.@functor GConvGRUCell

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
julia> g, x = rand_graph(5, 10), rand(Float32, 2, 5);

julia> ggru = GConvGRU(2 => 5, 2, g.num_nodes);

julia> y = ggru(g, x);

julia> size(y)
(5, 5)
```
""" 
GConvGRU(ch, k, n; kwargs...) = Flux.Recur(GConvGRUCell(ch, k, n; kwargs...))
Flux.Recur(ggru::GConvGRUCell) = Flux.Recur(ggru, ggru.state0)

(l::Flux.Recur{GConvGRUCell})(g::GNNGraph) = GNNGraph(g, ndata = l(g, node_features(g)))
_applylayer(l::Flux.Recur{GConvGRUCell}, g::GNNGraph, x) = l(g, x)
_applylayer(l::Flux.Recur{GConvGRUCell}, g::GNNGraph) = l(g)

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
