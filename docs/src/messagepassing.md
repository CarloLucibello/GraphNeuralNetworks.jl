# Message Passing

The message passing is initiated by the [`propagate`](@ref) function
and generally takes the form

```math
\begin{aligned}
\mathbf{m}_{j\to i} &= \phi(\mathbf{x}_i, \mathbf{x}_j, \mathbf{e}_{j\to i}) \\
\bar{\mathbf{m}}_{i} &= \square_{j\in N(i)}  \mathbf{m}_{j\to i} \\
\mathbf{x}_{i}' &= \gamma_x(\mathbf{x}_{i}, \bar{\mathbf{m}}_{i})\\
\mathbf{e}_{j\to i}^\prime &=  \gamma_e(\mathbf{e}_{j \to i},\mathbf{m}_{j \to i})
\end{aligned}
```

where we refer to ``\phi`` as to the message function, 
and to ``\gamma_x`` and ``\gamma_e`` as the node update and edge update function
respectively. The generic aggregation ``\square`` usually is given by a summation
``\sum``, a max or a mean operation. 

The message propagation mechanism internally relies on the [`NNlib.gather`](@ref) 
and [`NNlib.scatter`](@ref) methods.

## An example: implementing the GCNConv

Let's (re-)implement the [`GCNConv`](@ref) layer use the message passing framework.
The convolution reads 

```math
\mathbf{x}'_i = \sum_{j \in N(i)} \frac{1}{c_{ij}} W \mathbf{x}_j
```
where ``c_{ij} = \sqrt{|N(i)||N(j)|}``. We will also add a bias and an activation function.

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
    c = 1 ./ sqrt.(degree(g, T, dir=:in))
    x = x .* c'
    x = propagate((xi, xj, e) -> l.weight * xj, g, +, xj=x)
    x = x .* c'
    return l.σ.(x .+ l.bias)
end
```

See the [`GATConv`](@ref) implementation [here](https://github.com/CarloLucibello/GraphNeuralNetworks.jl/blob/master/src/layers/conv.jl) for a more complex example.
