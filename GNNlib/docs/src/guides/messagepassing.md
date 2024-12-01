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
and it is usually equal either to ``\sum``, to `max` or to a `mean` operation. 

In GNNlib.jl, the message passing mechanism is exposed by the [`propagate`](@ref) function.
[`propagate`](@ref) takes care of materializing the node features on each edge, applying the message function, performing the
aggregation, and returning ``\bar{\mathbf{m}}``. 
It is then left to the user to perform further node and edge updates,
manipulating arrays of size ``D_{node} \times num\_nodes`` and   
``D_{edge} \times num\_edges``.

[`propagate`](@ref) is composed of two steps, also available as two independent methods:

1. [`apply_edges`](@ref) materializes node features on edges and applies the message function. 
2. [`aggregate_neighbors`](@ref) applies a reduction operator on the messages coming from the neighborhood of each node.

The whole propagation mechanism internally relies on the [`NNlib.gather`](@extref) 
and [`NNlib.scatter`](@extref) methods.


## Examples

### Basic use of apply_edges and propagate

The function [`apply_edges`](@ref) can be used to broadcast node data
on each edge and produce new edge data.
```julia
julia> using GNNlib, Graphs, Statistics

julia> g = rand_graph(10, 20)
GNNGraph:
    num_nodes = 10
    num_edges = 20

julia> x = ones(2,10);

julia> z = 2ones(2,10);

# Return an edge features arrays (D × num_edges)
julia> apply_edges((xi, xj, e) -> xi .+ xj, g, xi=x, xj=z)
2×20 Matrix{Float64}:
 3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0
 3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0  3.0

# now returning a named tuple
julia> apply_edges((xi, xj, e) -> (a=xi .+ xj, b=xi .- xj), g, xi=x, xj=z)
(a = [3.0 3.0 … 3.0 3.0; 3.0 3.0 … 3.0 3.0], b = [-1.0 -1.0 … -1.0 -1.0; -1.0 -1.0 … -1.0 -1.0])

# Here we provide a named tuple input
julia> apply_edges((xi, xj, e) -> xi.a + xi.b .* xj, g, xi=(a=x,b=z), xj=z)
2×20 Matrix{Float64}:
 5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0
 5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0  5.0
```

The function [`propagate`](@ref) instead performs the [`apply_edges`](@ref) operation
but then also applies a reduction over each node's neighborhood (see [`aggregate_neighbors`](@ref)).
```julia
julia> propagate((xi, xj, e) -> xi .+ xj, g, +, xi=x, xj=z)
2×10 Matrix{Float64}:
 3.0  6.0  9.0  9.0  0.0  6.0  6.0  3.0  15.0  3.0
 3.0  6.0  9.0  9.0  0.0  6.0  6.0  3.0  15.0  3.0

# Previous output can be understood by looking at the degree
julia> degree(g)
10-element Vector{Int64}:
 1
 2
 3
 3
 0
 2
 2
 1
 5
 1
```

### Implementing a custom Graph Convolutional Layer using Flux.jl

Let's implement a simple graph convolutional layer using the message passing framework using the machine learning framework Flux.jl.
The convolution reads 

```math
\mathbf{x}'_i = W \cdot \sum_{j \in N(i)}  \mathbf{x}_j
```
We will also add a bias and an activation function.

```julia
using Flux, Graphs, GraphNeuralNetworks

struct GCN{A<:AbstractMatrix, B, F} <: GNNLayer
    weight::A
    bias::B
    σ::F
end

Flux.@layer GCN # allow gpu movement, select trainable params etc...

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

See the `GATConv` implementation [here](https://juliagraphs.org/GraphNeuralNetworks.jl/graphneuralnetworks/api/conv/) for a more complex example.


## Built-in message functions

In order to exploit optimized specializations of the [`propagate`](@ref), it is recommended 
to use built-in message functions such as [`copy_xj`](@ref) whenever possible. 
