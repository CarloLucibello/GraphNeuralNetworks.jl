function (m::Flux.Recur)(g, x)
    m.state, y = m.cell(m.state, g, x)
    return y
    end
    
function (m::Flux.Recur)(g::GNNGraph, x::AbstractArray{T, 3}) where T
h = [m(g, x_t) for x_t in Flux.eachlastdim(x)]
sze = size(h[1])
reshape(reduce(hcat, h), sze[1], sze[2], length(h))
end

"""
    TGCNCell(in => out; [bias, init, add_self_loops, use_edge_weight])

Temporal Graph Convolutional Network (T-GCN) cell from the paper [T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction](https://arxiv.org/pdf/1811.05320.pdf).

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
"""
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

TGCN(ch; kwargs...) = Flux.Recur(TGCNCell(ch; kwargs...))
Flux.Recur(tgcn::TGCNCell) = Flux.Recur(tgcn, tgcn.state0)
