# Message Passing

A generic message passing on graph takes the form

```math
\begin{aligned}
\mathbf{m}_{j\to i} &= \phi(\mathbf{x}_i, \mathbf{x}_j, \mathbf{e}_{j\to i}) \\
\bar{\mathbf{m}}_{i} &= \square_{j\in N(i)}  \mathbf{m}_{j\to i} \\
\mathbf{x}_{i}' &= \gamma_x(\mathbf{x}_{i}, \bar{\mathbf{m}}_{i})\\
\mathbf{e}_{j\to i}^\prime &=  \gamma_e(\mathbf{e}_{j \to i},\mathbf{m}_{j \to i})
\end{aligned}
```

where we refer to ``\phi`` as to the message function, 
and to ``\gamma_x`` and ``\gamma_e`` as to the node update and edge update function
respectively. The aggregation ``\square`` is over the neighborhood ``N(i)`` of node ``i``, 
and it is usually set to summation ``\sum``, a max or a mean operation. 

In GNN.jl, the function [`propagate`](@ref) takes care of materializing the
node features on each edge, applying the message function, performing the
aggregation, and returning ``\bar{\mathbf{m}}``. 
It is then left to the user to perform further node and edge updates,
manypulating arrays of size ``D_{node} \times num_nodes`` and   
``D_{edge} \times num_edges``.

As part of the [`propagate`](@ref) pipeline, we have the function
[`apply_edges`](@ref). It can be independently used to materialize 
node features on edges and perform edge-related computation without
the following neighborhood aggregation one finds in `propagate`.

The whole propagation mechanism internally relies on the [`NNlib.gather`](@ref) 
and [`NNlib.scatter`](@ref) methods.


## Examples

### Basic use propagate and apply_edges 



### Implementing a custom Graph Convolutional Layer

Let's implement a simple graph convolutional layer using the message passing framework.
The convolution reads 

```math
\mathbf{x}'_i = W \cdot \sum_{j \in N(i)}  \mathbf{x}_j
```
We will also add a bias and an activation function.

```julia
using Flux, LightGraphs, GraphNeuralNetworks

struct GCN{A<:AbstractMatrix, B, F} <: GNNLayer
    weight::A
    bias::B
    σ::F
end

Flux.@functor GCN # allow collecting params, gpu movement, etc...

function GCN(ch::Pair{Int,Int}, σ=identity)
    in, out = ch
    W = Flux.glorot_uniform(out, in)
    b = zeros(Float32, out)
    GCN(W, b, σ)
end

function (l::GCN)(g::GNNGraph, x::AbstractMatrix{T}) where T
    @assert size(x, 2) == g.num_nodes

    # Computes messages from source/neighbour nodes (j) to target/root nodes (i).
    # The message function will have to handle matrices of size (*, num_edges).
    # In this simple case we just let the neighbor features go through.
    message(xi, xj, e) = xj 

    # The + operator gives the sum aggregation.
    # `mean`, `max`, `min`, and `*` are other possibilities.
    x = propagate(message, g, +, xj=x) 

    return l.σ.(l.weight * x .+ l.bias)
end
```

See the [`GATConv`](@ref) implementation [here](https://github.com/CarloLucibello/GraphNeuralNetworks.jl/blob/master/src/layers/conv.jl) for a more complex example.
