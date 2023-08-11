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
    TGCN(in => out; [bias, init, add_self_loops, use_edge_weight])

Temporal Graph Convolutional Network (T-GCN) recurrent layer from the paper [T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction](https://arxiv.org/pdf/1811.05320.pdf).

Performs a layer of GCNConv to model spatial dependencies, followed by a Gated Recurrent Unit (GRU) cell to model temporal dependencies.

# Arguments

- `in`: Number of input features.
- `out`: Number of output features.
- `bias`: Add learnable bias. Default `true`.
- `init`: Weights' initializer. Default `glorot_uniform`.
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

struct A3TGCN
    tgcn::Flux.Recur{TGCNCell}
    dense1::Dense
    dense2::Dense
    in::Int
    out::Int
end

Flux.@functor A3TGCN

function A3TGCN(ch::Pair{Int, Int};
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
    c= dropdims(c, dims = 3)
    return c
end

function Base.show(io::IO, a3tgcn::A3TGCN)
    print(io, "A3TGCN($(a3tgcn.in) => $(a3tgcn.out))")
end