
@doc raw"""
    GraphConv(in => out, σ=identity; aggr=+, bias=true, init=glorot_uniform)

Graph convolution layer from Reference: [Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks](https://arxiv.org/abs/1810.02244).

Performs:
```math
\mathbf{x}_i' = W_1 \mathbf{x}_i + \square_{j \in \mathcal{N}(i)} W_2 \mathbf{x}_j
```

where the aggregation type is selected by `aggr`.

# Arguments

- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `σ`: Activation function.
- `aggr`: Aggregation operator for the incoming messages (e.g. `+`, `*`, `max`, `min`, and `mean`).
- `bias`: Add learnable bias.
- `init`: Weights' initializer.

# Examples

```julia
# create data
s = [1,1,2,3]
t = [2,3,1,1]
in_channel = 3
out_channel = 5
g = GNNGraph(s, t)
x = randn(Float32, 3, g.num_nodes)

# create layer
l = GraphConv(in_channel => out_channel, relu, bias = false, aggr = mean)

# forward pass
y = l(g, x)       
```
"""
@concrete struct GraphConv <: AbstractExplicitLayer
    in_dims::Int
    out_dims::Int
    use_bias::Bool
    init_weight::Function
    init_bias::Function
    σ
    aggr
end


function GraphConv(ch::Pair{Int, Int}, σ = identity;
            aggr = +,
            init_weight = glorot_uniform,
            init_bias = zeros32, 
            use_bias::Bool = true, 
            allow_fast_activation::Bool = true)
    in_dims, out_dims = ch
    σ = allow_fast_activation ? NNlib.fast_act(σ) : σ
    return GraphConv(in_dims, out_dims, use_bias, init_weight, init_bias, σ, aggr)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::GraphConv)
    weight1 = l.init_weight(rng, l.out_dims, l.in_dims)
    weight2 = l.init_weight(rng, l.out_dims, l.in_dims)
    if l.use_bias
        bias = l.init_bias(rng, l.out_dims)
    else
        bias = false
    end
    return (; weight1, weight2, bias)
end

function LuxCore.parameterlength(l::GraphConv)
    if l.use_bias
        return 2 * l.in_dims * l.out_dims + l.out_dims
    else
        return 2 * l.in_dims * l.out_dims
    end
end

statelength(d::GraphConv) = 0
outputsize(d::GraphConv) = (d.out_dims,)

function Base.show(io::IO, l::GraphConv)
    print(io, "GraphConv(", l.in_dims, " => ", l.out_dims)
    (l.σ == identity) || print(io, ", ", l.σ)
    (l.aggr == +) || print(io, ", aggr=", l.aggr)
    l.use_bias || print(io, ", use_bias=false")
    print(io, ")")
end

(l::GraphConv)(g::GNNGraph, x, ps, st) = GNNlib.graph_conv(l, g, x, ps), st
