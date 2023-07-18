function _tgcn_set_hidden_state(x::AbstractMatrix, h::Union{AbstractMatrix, Nothing}, out::Int)
    if h === nothing
        h = zeros(eltype(x), out, size(x, 2))
    end
    return h
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
    in::Int
    out::Int
end

Flux.@functor TGCNCell

function TGCNCell(ch::Pair{Int, Int};
                  bias::Bool = true,
                  init = Flux.glorot_uniform,
                  add_self_loops = false,
                  use_edge_weight = true)
    in, out = ch
    conv = GCNConv(in => out, sigmoid; init, bias, add_self_loops,
                   use_edge_weight)
    gru = Flux.GRUv3Cell(out, out)
    return TGCNCell(conv, gru, in,
                    out)
end

function (tgcn::TGCNCell)(g::GNNGraph, x::AbstractArray; h = nothing)
    h = _tgcn_set_hidden_state(x, h, tgcn.out)
    x̃ = tgcn.conv(g, x)
    h, _ = tgcn.gru(h, x̃)
    return h
end

function Base.show(io::IO, tgcn::TGCNCell)
    print(io, "TGCNCell($(tgcn.in) => $(tgcn.out))")
end
