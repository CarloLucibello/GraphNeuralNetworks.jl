using DataStructures: nlargest

@doc raw"""
    GlobalPool(aggr)

Global pooling layer for graph neural networks.
Takes a graph and feature nodes as inputs
and performs the operation

```math
\mathbf{u}_V = \square_{i \in V} \mathbf{x}_i
```

where ``V`` is the set of nodes of the input graph and 
the type of aggregation represented by ``\square`` is selected by the `aggr` argument. 
Commonly used aggregations are `mean`, `max`, and `+`.

See also [`reduce_nodes`](@ref).

# Examples

```julia
using Flux, GraphNeuralNetworks, Graphs

pool = GlobalPool(mean)

g = GNNGraph(erdos_renyi(10, 4))
X = rand(32, 10)
pool(g, X) # => 32x1 matrix


g = Flux.batch([GNNGraph(erdos_renyi(10, 4)) for _ in 1:5])
X = rand(32, 50)
pool(g, X) # => 32x5 matrix
```
"""
struct GlobalPool{F} <: GNNLayer
    aggr::F
end

function (l::GlobalPool)(g::GNNGraph, x::AbstractArray)
    return reduce_nodes(l.aggr, g, x)
end

(l::GlobalPool)(g::GNNGraph) = GNNGraph(g, gdata = l(g, node_features(g)))

@doc raw"""
    GlobalAttentionPool(fgate, ffeat=identity)

Global soft attention layer from the [Gated Graph Sequence Neural
Networks](https://arxiv.org/abs/1511.05493) paper

```math
\mathbf{u}_V = \sum_{i\in V} \alpha_i\, f_{feat}(\mathbf{x}_i)
```

where the coefficients ``\alpha_i`` are given by a [`softmax_nodes`](@ref)
operation:

```math
\alpha_i = \frac{e^{f_{gate}(\mathbf{x}_i)}}
                {\sum_{i'\in V} e^{f_{gate}(\mathbf{x}_{i'})}}.
```

# Arguments

- `fgate`: The function ``f_{gate}: \mathbb{R}^{D_{in}} \to \mathbb{R}``. 
           It is tipically expressed by a neural network.

- `ffeat`: The function ``f_{feat}: \mathbb{R}^{D_{in}} \to \mathbb{R}^{D_{out}}``. 
           It is tipically expressed by a neural network.

# Examples

```julia
chin = 6
chout = 5    

fgate = Dense(chin, 1)
ffeat = Dense(chin, chout)
pool = GlobalAttentionPool(fgate, ffeat)

g = Flux.batch([GNNGraph(random_regular_graph(10, 4), 
                         ndata=rand(Float32, chin, 10)) 
                for i=1:3])

u = pool(g, g.ndata.x)

@assert size(u) == (chout, g.num_graphs)
```
"""
struct GlobalAttentionPool{G, F}
    fgate::G
    ffeat::F
end

@functor GlobalAttentionPool

GlobalAttentionPool(fgate) = GlobalAttentionPool(fgate, identity)

function (l::GlobalAttentionPool)(g::GNNGraph, x::AbstractArray)
    α = softmax_nodes(g, l.fgate(x))
    feats = α .* l.ffeat(x)
    u = reduce_nodes(+, g, feats)
    return u
end

(l::GlobalAttentionPool)(g::GNNGraph) = GNNGraph(g, gdata = l(g, node_features(g)))

"""
    TopKPool(adj, k, in_channel)

Top-k pooling layer.

# Arguments

- `adj`: Adjacency matrix  of a graph.
- `k`: Top-k nodes are selected to pool together.
- `in_channel`: The dimension of input channel.
"""
struct TopKPool{T, S}
    A::AbstractMatrix{T}
    k::Int
    p::AbstractVector{S}
    Ã::AbstractMatrix{T}
end

function TopKPool(adj::AbstractMatrix, k::Int, in_channel::Int; init = glorot_uniform)
    TopKPool(adj, k, init(in_channel), similar(adj, k, k))
end

function (t::TopKPool)(X::AbstractArray)
    y = t.p' * X / norm(t.p)
    idx = topk_index(y, t.k)
    t.Ã .= view(t.A, idx, idx)
    X_ = view(X, :, idx) .* σ.(view(y, idx)')
    return X_
end

function topk_index(y::AbstractVector, k::Int)
    v = nlargest(k, y)
    return collect(1:length(y))[y .>= v[end]]
end

topk_index(y::Adjoint, k::Int) = topk_index(y', k)


@doc raw"""
    Set2Set(n_in, n_iters, n_layers = 1)

Set2Set layer from the paper [Order Matters: Sequence to sequence for sets](https://arxiv.org/abs/1511.06391).

For each graph in the batch, the layer computes an output vector of size `2*n_in` by iterating the following steps `n_iters` times:
```math
\mathbf{q} = \mathrm{LSTM}(\mathbf{q}_{t-1}^*)
\alpha_{i} = \frac{\exp(\mathbf{q}^T \mathbf{x}_i)}{\sum_{j=1}^N \exp(\mathbf{q}^T \mathbf{x}_j)} 
\mathbf{r} = \sum_{i=1}^N \alpha_{i} \mathbf{x}_i
\mathbf{q}^*_t = [\mathbf{q}; \mathbf{r}]
```
where `N` is the number of nodes in the graph, `LSTM` is a Long-Short-Term-Memory network with `n_layers` layers, 
input size `2*n_in` and output size `n_in`.

Given a batch of graphs `g` and node features `x`, the layer returns a matrix of size `(2*n_in, n_graphs)`.
```
"""
struct Set2Set{L} <: GNNLayer
    lstm::L
    num_iters::Int
end

@functor Set2Set

function Set2Set(n_in::Int, n_iters::Int, n_layers::Int = 1)
    @assert n_layers >= 1
    n_out = 2 * n_in

    if n_layers == 1
        lstm = LSTM(n_out => n_in)
    else
        layers = [LSTM(n_out => n_in)]
        for _ in 2:n_layers
            push!(layers, LSTM(n_in => n_in))
        end
        lstm = Chain(layers...)
    end

    return Set2Set(lstm, n_iters)
end

function (l::Set2Set)(g::GNNGraph, x::AbstractMatrix)
    n_in = size(x, 1)
    Flux.reset!(l.lstm)
    qstar = zeros_like(x, (2*n_in, g.num_graphs))
    for t in 1:l.num_iters
        q = l.lstm(qstar)                            # [n_in, n_graphs]
        qn = broadcast_nodes(g, q)                    # [n_in, n_nodes]
        α = softmax_nodes(g, sum(qn .* x, dims = 1))  # [1, n_nodes]
        r = reduce_nodes(+, g, x .* α)               # [n_in, n_graphs]
        qstar = vcat(q, r)                           # [2*n_in, n_graphs]
    end
    return qstar
end

(l::Set2Set)(g::GNNGraph) = GNNGraph(g, gdata = l(g, node_features(g)))
