@doc raw"""
    GCNConv(in => out, σ=identity; bias=true, init=glorot_uniform)

Graph convolutional layer from paper [Semi-supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907).

Performs the operation
```math
\mathbf{x}'_i = \sum_{j\in \{i\} \cup N(i)} \frac{1}{c_{ij}} W \mathbf{x}_j
```
where ``c_{ij} = \sqrt{(1+|N(i)|)(1+|N(j)|)}``.

The input to the layer is a node feature array `X` 
of size `(num_features, num_nodes)`.

# Arguments

- `in`: Number of input features.
- `out`: Number of output features.
- `σ`: Activation function.
- `bias`: Add learnable bias.
- `init`: Weights' initializer.
"""
struct GCNConv{A<:AbstractMatrix, B, F} <: GNNLayer
    weight::A
    bias::B
    σ::F
end

@functor GCNConv

function GCNConv(ch::Pair{Int,Int}, σ=identity;
                 init=glorot_uniform, bias::Bool=true)
    in, out = ch
    W = init(out, in)
    b = Flux.create_bias(W, bias, out)
    GCNConv(W, b, σ)
end

## Matrix operations are more performant, 
## but cannot compute the normalized laplacian of sparse cuda matrices yet,
## therefore fallback to message passing framework on gpu for the time being
 
function (l::GCNConv)(g::GNNGraph, x::AbstractMatrix{T}) where T
    Ã = normalized_adjacency(g, T; dir=:out, add_self_loops=true)
    l.σ.(l.weight * x * Ã .+ l.bias)
end

compute_message(l::GCNConv, xi, xj, eij) = xj
update_node(l::GCNConv, m, x) = m

function (l::GCNConv)(g::GNNGraph, x::CuMatrix{T}) where T
    g = add_self_loops(g)
    c = 1 ./ sqrt.(degree(g, T, dir=:in))
    x = x .* c'
    x, _ = propagate(l, g, +, x)
    x = x .* c'
    return l.σ.(l.weight * x .+ l.bias)
end

function Base.show(io::IO, l::GCNConv)
    out, in = size(l.weight)
    print(io, "GCNConv($in => $out")
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ")")
end


@doc raw"""
    ChebConv(in => out, k; bias=true, init=glorot_uniform)

Chebyshev spectral graph convolutional layer from
paper [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375).

Implements

```math
X' = \sum^{K-1}_{k=0}  W^{(k)} Z^{(k)}
```

where ``Z^{(k)}`` is the ``k``-th term of Chebyshev polynomials, and can be calculated by the following recursive form:

```math
Z^{(0)} = X \\
Z^{(1)} = \hat{L} X \\
Z^{(k)} = 2 \hat{L} Z^{(k-1)} - Z^{(k-2)}
```

with ``\hat{L}`` the [`scaled_laplacian`](@ref).

# Arguments

- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `k`: The order of Chebyshev polynomial.
- `bias`: Add learnable bias.
- `init`: Weights' initializer.
"""
struct ChebConv{A<:AbstractArray{<:Number,3}, B} <: GNNLayer
    weight::A
    bias::B
    k::Int
end

function ChebConv(ch::Pair{Int,Int}, k::Int;
                  init=glorot_uniform, bias::Bool=true)
    in, out = ch
    W = init(out, in, k)
    b = Flux.create_bias(W, bias, out)
    ChebConv(W, b, k)
end

@functor ChebConv

function (c::ChebConv)(g::GNNGraph, X::AbstractMatrix{T}) where T
    check_num_nodes(g, X)
    @assert size(X, 1) == size(c.weight, 2) "Input feature size must match input channel size."
    
    L̃ = scaled_laplacian(g, eltype(X))    

    Z_prev = X
    Z = X * L̃
    Y = view(c.weight,:,:,1) * Z_prev
    Y += view(c.weight,:,:,2) * Z
    for k = 3:c.k
        Z, Z_prev = 2*Z*L̃ - Z_prev, Z
        Y += view(c.weight,:,:,k) * Z
    end
    return Y .+ c.bias
end

function Base.show(io::IO, l::ChebConv)
    out, in, k = size(l.weight)
    print(io, "ChebConv(", in, " => ", out)
    print(io, ", k=", k)
    print(io, ")")
end


@doc raw"""
    GraphConv(in => out, σ=identity, aggr=+; bias=true, init=glorot_uniform)

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
"""
struct GraphConv{A<:AbstractMatrix, B} <: GNNLayer
    weight1::A
    weight2::A
    bias::B
    σ
    aggr
end

@functor GraphConv

function GraphConv(ch::Pair{Int,Int}, σ=identity, aggr=+;
                   init=glorot_uniform, bias::Bool=true)
    in, out = ch
    W1 = init(out, in)
    W2 = init(out, in)
    b = Flux.create_bias(W1, bias, out)
    GraphConv(W1, W2, b, σ, aggr)
end

compute_message(l::GraphConv, x_i, x_j, e_ij) =  x_j
update_node(l::GraphConv, m, x) = l.σ.(l.weight1 * x .+ l.weight2 * m .+ l.bias)

function (l::GraphConv)(g::GNNGraph, x::AbstractMatrix)
    check_num_nodes(g, x)
    x, _ = propagate(l, g, l.aggr, x)
    x
end

function Base.show(io::IO, l::GraphConv)
    in_channel = size(l.weight1, ndims(l.weight1))
    out_channel = size(l.weight1, ndims(l.weight1)-1)
    print(io, "GraphConv(", in_channel, " => ", out_channel)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end


@doc raw"""
    GATConv(in => out, , σ=identity;
            heads=1,
            concat=true,
            init=glorot_uniform    
            bias=true, 
            negative_slope=0.2f0)

Graph attentional layer from the paper [Graph Attention Networks](https://arxiv.org/abs/1710.10903).

Implements the operation
```math
\mathbf{x}_i' = \sum_{j \in N(i)} \alpha_{ij} W \mathbf{x}_j
```
where the attention coefficient ``\alpha_{ij}`` is given by
```math
\alpha_{ij} = \frac{1}{z_i} \exp(LeakyReLU(\mathbf{a}^T [W \mathbf{x}_i || W \mathbf{x}_j]))
```
with ``z_i`` a normalization factor.

# Arguments

- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `bias::Bool`: Keyword argument, whether to learn the additive bias.
- `heads`: Number attention heads 
- `concat`: Concatenate layer output or not. If not, layer output is averaged over the heads.
- `negative_slope::Real`: Keyword argument, the parameter of LeakyReLU.
"""
struct GATConv{T, A<:AbstractMatrix{T}, B} <: GNNLayer
    weight::A
    bias::B
    a::A
    σ
    negative_slope::T
    channel::Pair{Int, Int}
    heads::Int
    concat::Bool
end

@functor GATConv
Flux.trainable(l::GATConv) = (l.weight, l.bias, l.a)

function GATConv(ch::Pair{Int,Int}, σ=identity;
                 heads::Int=1, concat::Bool=true, negative_slope=0.2f0,
                 init=glorot_uniform, bias::Bool=true)
    in, out = ch             
    W = init(out*heads, in)
    b = Flux.create_bias(W, bias, out*heads)
    a = init(2*out, heads)
    GATConv(W, b, a, σ, negative_slope, ch, heads, concat)
end

function compute_message(l::GATConv, Wxi, Wxj)
    aWW = sum(l.a .* cat(Wxi, Wxj, dims=1), dims=1)   # 1 × nheads × nedges
    α = exp.(leakyrelu.(aWW, l.negative_slope))       
    return (α = α, m = α .* Wxj)
end

update_node(l::GATConv, d̄, x) = d̄.m ./ d̄.α

function (l::GATConv)(g::GNNGraph, x::AbstractMatrix)
    check_num_nodes(g, x)
    g = add_self_loops(g)
    chin, chout = l.channel
    heads = l.heads

    Wx = l.weight * x
    Wx = reshape(Wx, chout, heads, :)                   # chout × nheads × nnodes
    
    x, _ = propagate(l, g, +, Wx)                 ## chout × nheads × nnodes

    b = reshape(l.bias, chout, heads)
    x = l.σ.(x .+ b)                                      
    if !l.concat
        x = sum(x, dims=2)
    end

    # We finally return a matrix
    return reshape(x, :, size(x, 3)) 
end


function Base.show(io::IO, l::GATConv)
    in_channel = size(l.weight, ndims(l.weight))
    out_channel = size(l.weight, ndims(l.weight)-1)
    print(io, "GATConv(", in_channel, "=>", out_channel)
    print(io, ", LeakyReLU(λ=", l.negative_slope)
    print(io, "))")
end


@doc raw"""
    GatedGraphConv(out, num_layers; aggr=+, init=glorot_uniform)

Gated graph convolution layer from [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493).

Implements the recursion
```math
\mathbf{h}^{(0)}_i = \mathbf{x}_i || \mathbf{0} \\
\mathbf{h}^{(l)}_i = GRU(\mathbf{h}^{(l-1)}_i, \square_{j \in N(i)} W \mathbf{h}^{(l-1)}_j)
```

 where ``\mathbf{h}^{(l)}_i`` denotes the ``l``-th hidden variables passing through GRU. The dimension of input ``\mathbf{x}_i`` needs to be less or equal to `out`.

# Arguments

- `out`: The dimension of output features.
- `num_layers`: The number of gated recurrent unit.
- `aggr`: Aggregation operator for the incoming messages (e.g. `+`, `*`, `max`, `min`, and `mean`).
- `init`: Weight initialization function.
"""
struct GatedGraphConv{A<:AbstractArray{<:Number,3}, R} <: GNNLayer
    weight::A
    gru::R
    out_ch::Int
    num_layers::Int
    aggr
end

@functor GatedGraphConv


function GatedGraphConv(out_ch::Int, num_layers::Int;
                        aggr=+, init=glorot_uniform)
    w = init(out_ch, out_ch, num_layers)
    gru = GRUCell(out_ch, out_ch)
    GatedGraphConv(w, gru, out_ch, num_layers, aggr)
end


compute_message(l::GatedGraphConv, x_i, x_j, e_ij) = x_j
update_node(l::GatedGraphConv, m, x) = m

# remove after https://github.com/JuliaDiff/ChainRules.jl/pull/521
@non_differentiable fill!(x...)

function (l::GatedGraphConv)(g::GNNGraph, H::AbstractMatrix{S}) where {T<:AbstractVector,S<:Real}
    check_num_nodes(g, H)
    m, n = size(H)
    @assert (m <= l.out_ch) "number of input features must less or equals to output features."
    if m < l.out_ch
        Hpad = similar(H, S, l.out_ch - m, n)
        H = vcat(H, fill!(Hpad, 0))
    end
    for i = 1:l.num_layers
        M = view(l.weight, :, :, i) * H
        M, _ = propagate(l, g, l.aggr, M)
        H, _ = l.gru(H, M)
    end
    H
end

function Base.show(io::IO, l::GatedGraphConv)
    print(io, "GatedGraphConv(($(l.out_ch) => $(l.out_ch))^$(l.num_layers)")
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end


@doc raw"""
    EdgeConv(f; aggr=max)

Edge convolutional layer from paper [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829).

Performs the operation
```math
\mathbf{x}_i' = \square_{j \in N(i)} f(\mathbf{x}_i || \mathbf{x}_j - \mathbf{x}_i)
```

where `f` typically denotes a learnable function, e.g. a linear layer or a multi-layer perceptron.

# Arguments

- `f`: A (possibly learnable) function acting on edge features. 
- `aggr`: Aggregation operator for the incoming messages (e.g. `+`, `*`, `max`, `min`, and `mean`).
"""
struct EdgeConv <: GNNLayer
    nn
    aggr
end

@functor EdgeConv

EdgeConv(nn; aggr=max) = EdgeConv(nn, aggr)

compute_message(l::EdgeConv, x_i, x_j, e_ij) = l.nn(vcat(x_i, x_j .- x_i))

update_node(l::EdgeConv, m, x) = m

function (l::EdgeConv)(g::GNNGraph, X::AbstractMatrix)
    check_num_nodes(g, X)
    X, _ = propagate(l, g, l.aggr, X)
    X
end

function Base.show(io::IO, l::EdgeConv)
    print(io, "EdgeConv(", l.nn)
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end


@doc raw"""
    GINConv(f; eps = 0f0)

Graph Isomorphism convolutional layer from paper [How Powerful are Graph Neural Networks?](https://arxiv.org/pdf/1810.00826.pdf)


```math
\mathbf{x}_i' = f\left((1 + \epsilon) \mathbf{x}_i + \sum_{j \in N(i)} \mathbf{x}_j \right)
```
where `f` typically denotes a learnable function, e.g. a linear layer or a multi-layer perceptron.

# Arguments

- `f`: A (possibly learnable) function acting on node features. 
- `eps`: Weighting factor.
"""
struct GINConv{R<:Real} <: GNNLayer
    nn
    eps::R
end

@functor GINConv
Flux.trainable(l::GINConv) = (nn=l.nn,)

function GINConv(nn; eps=0f0)
    GINConv(nn, eps)
end

compute_message(l::GINConv, x_i, x_j, e_ij) = x_j 
update_node(l::GINConv, m, x) = l.nn((1 + l.eps) * x + m)

function (l::GINConv)(g::GNNGraph, X::AbstractMatrix)
    check_num_nodes(g, X)
    X, _ = propagate(l, g, +, X)
    X
end
