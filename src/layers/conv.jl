"""
    GCNConv(in => out, σ=identity; bias=true, init=glorot_uniform)

Graph convolutional layer.

# Arguments

- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `σ`: Activation function.
- `bias`: Add learnable bias.
- `init`: Weights' initializer.

The input to the layer is a node feature array `X` 
of size `(num_features, num_nodes)`.
"""
struct GCNConv{A<:AbstractMatrix, B, F} <: MessagePassing
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
 
function (l::GCNConv)(fg::FeaturedGraph, x::AbstractMatrix)
    Ã = normalized_adjacency(fg, eltype(x); dir=:out, add_self_loops=true)
    l.σ.(l.weight * x * Ã .+ l.bias)
end

message(l::GCNConv, xi, xj) = xj
update(l::GCNConv, m, x) = m

function (l::GCNConv)(fg::FeaturedGraph, x::CuMatrix)
    fg = add_self_loops(fg; add_to_existing=true)
    T = eltype(l.weight)
    # cout = sqrt.(degree(fg, dir=:out))
    cin = 1 ./ reshape(sqrt.(T.(degree(fg, dir=:in))), 1, :)
    x = cin .* x
    _, x = propagate(l, fg, nothing, x, nothing, +)
    x = cin .* x
    return l.σ.(l.weight * x .+ l.bias)
end

(l::GCNConv)(fg::FeaturedGraph) = FeaturedGraph(fg, nf = l(fg, node_feature(fg)))

function Base.show(io::IO, l::GCNConv)
    out, in = size(l.weight)
    print(io, "GCNConv($in => $out")
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ")")
end


"""
    ChebConv(in=>out, k; bias=true, init=glorot_uniform)

Chebyshev spectral graph convolutional layer.

# Arguments

- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `k`: The order of Chebyshev polynomial.
- `bias`: Add learnable bias.
- `init`: Weights' initializer.
"""
struct ChebConv{A<:AbstractArray{<:Number,3}, B}
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

function (c::ChebConv)(fg::FeaturedGraph, X::AbstractMatrix{T}) where T
    check_num_nodes(fg, X)
    @assert size(X, 1) == size(c.weight, 2) "Input feature size must match input channel size."
    
    L̃ = scaled_laplacian(fg, eltype(X))    

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

(l::ChebConv)(fg::FeaturedGraph) = FeaturedGraph(fg, nf = l(fg, node_feature(fg)))

function Base.show(io::IO, l::ChebConv)
    out, in, k = size(l.weight)
    print(io, "ChebConv(", in, " => ", out)
    print(io, ", k=", k)
    print(io, ")")
end


"""
    GraphConv(in => out, σ=identity, aggr=+; bias=true, init=glorot_uniform)

Graph neural network layer.

# Arguments

- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `σ`: Activation function.
- `aggr`: An aggregate function applied to the result of message function. `+`, `-`,
`*`, `/`, `max`, `min` and `mean` are available.
- `bias`: Add learnable bias.
- `init`: Weights' initializer.
"""
struct GraphConv{A<:AbstractMatrix, B} <: MessagePassing
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

message(gc::GraphConv, x_i, x_j, e_ij) =  x_j
update(gc::GraphConv, m, x) = gc.σ.(gc.weight1 * x .+ gc.weight2 * m .+ gc.bias)

function (gc::GraphConv)(fg::FeaturedGraph, x::AbstractMatrix)
    check_num_nodes(fg, x)
    _, x = propagate(gc, fg, nothing, x, nothing, +)
    x
end

(l::GraphConv)(fg::FeaturedGraph) = FeaturedGraph(fg, nf = l(fg, node_feature(fg)))

function Base.show(io::IO, l::GraphConv)
    in_channel = size(l.weight1, ndims(l.weight1))
    out_channel = size(l.weight1, ndims(l.weight1)-1)
    print(io, "GraphConv(", in_channel, " => ", out_channel)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end


"""
    GATConv(in => out;
            heads=1,
            concat=true,
            init=glorot_uniform    
            bias=true, 
            negative_slope=0.2)

Graph attentional layer.

# Arguments

- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `bias::Bool`: Keyword argument, whether to learn the additive bias.
- `heads`: Number attention heads 
- `concat`: Concatenate layer output or not. If not, layer output is averaged over the heads.
- `negative_slope::Real`: Keyword argument, the parameter of LeakyReLU.
"""
struct GATConv{T, A<:AbstractMatrix{T}, B} <: MessagePassing
    weight::A
    bias::B
    a::A
    negative_slope::T
    channel::Pair{Int, Int}
    heads::Int
    concat::Bool
end

@functor GATConv

function GATConv(ch::Pair{Int,Int};
                 heads::Int=1, concat::Bool=true, negative_slope=0.2f0,
                 init=glorot_uniform, bias::Bool=true)
    in, out = ch             
    W = init(out*heads, in)
    b = Flux.create_bias(W, bias, out*heads)
    a = init(2*out, heads)
    GATConv(W, b, a, negative_slope, ch, heads, concat)
end

function (gat::GATConv)(fg::FeaturedGraph, X::AbstractMatrix)
    check_num_nodes(fg, X)
    fg = add_self_loops(fg)
    chin, chout = gat.channel
    heads = gat.heads

    source, target = edge_index(fg)
    Wx = gat.weight*X
    Wx = reshape(Wx, chout, heads, :)                   # chout × nheads × nnodes
    Wxi = NNlib.gather(Wx, target)                      # chout × nheads × nedges
    Wxj = NNlib.gather(Wx, source)

    # Edge Message
    # Computing softmax. TODO make it numerically stable
    aWW = sum(gat.a .* cat(Wxi, Wxj, dims=1), dims=1)   # 1 × nheads × nedges
    α = exp.(leakyrelu.(aWW, gat.negative_slope))       
    m̄ =  NNlib.scatter(+, α .* Wxj, target)             # chout × nheads × nnodes 
    ᾱ = NNlib.scatter(+, α, target)                     # 1 × nheads × nnodes
    
    # Node update
    b = reshape(gat.bias, chout, heads)
    X = m̄ ./ ᾱ .+ b                                     # chout × nheads × nnodes 
    if !gat.concat
        X = sum(X, dims=2)
    end

    # We finally return a matrix
    return reshape(X, :, size(X, 3)) 
end

(l::GATConv)(fg::FeaturedGraph) = FeaturedGraph(fg, nf = l(fg, node_feature(fg)))


function Base.show(io::IO, l::GATConv)
    in_channel = size(l.weight, ndims(l.weight))
    out_channel = size(l.weight, ndims(l.weight)-1)
    print(io, "GATConv(", in_channel, "=>", out_channel)
    print(io, ", LeakyReLU(λ=", l.negative_slope)
    print(io, "))")
end


"""
    GatedGraphConv(out, num_layers; aggr=+, init=glorot_uniform)

Gated graph convolution layer.

# Arguments

- `out`: The dimension of output features.
- `num_layers`: The number of gated recurrent unit.
- `aggr`: An aggregate function applied to the result of message function. `+`, `-`,
`*`, `/`, `max`, `min` and `mean` are available.
"""
struct GatedGraphConv{A<:AbstractArray{<:Number,3}, R} <: MessagePassing
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


message(l::GatedGraphConv, x_i, x_j, e_ij) = x_j
update(l::GatedGraphConv, m, x) = m

function (ggc::GatedGraphConv)(fg::FeaturedGraph, H::AbstractMatrix{S}) where {T<:AbstractVector,S<:Real}
    check_num_nodes(fg, H)
    m, n = size(H)
    @assert (m <= ggc.out_ch) "number of input features must less or equals to output features."
    if m < ggc.out_ch
        Hpad = similar(H, S, ggc.out_ch - m, n)
        H = vcat(H, fill!(Hpad, 0))
    end
    for i = 1:ggc.num_layers
        M = view(ggc.weight, :, :, i) * H
        _, M = propagate(ggc, fg, nothing, M, nothing, +)
        H, _ = ggc.gru(H, M)
    end
    H
end

(l::GatedGraphConv)(fg::FeaturedGraph) = FeaturedGraph(fg, nf = l(fg, node_feature(fg)))

function Base.show(io::IO, l::GatedGraphConv)
    print(io, "GatedGraphConv(($(l.out_ch) => $(l.out_ch))^$(l.num_layers)")
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end



"""
    EdgeConv(nn; aggr=max)

Edge convolutional layer.

# Arguments

- `nn`: A neural network (e.g. a Dense layer or a MLP). 
- `aggr`: An aggregate function applied to the result of message function. `+`, `max` and `mean` are available.
"""
struct EdgeConv <: MessagePassing
    nn
    aggr
end

@functor EdgeConv

EdgeConv(nn; aggr=max) = EdgeConv(nn, aggr)

message(ec::EdgeConv, x_i, x_j, e_ij) = ec.nn(vcat(x_i, x_j .- x_i))

update(ec::EdgeConv, m, x) = m

function (ec::EdgeConv)(fg::FeaturedGraph, X::AbstractMatrix)
    check_num_nodes(fg, X)
    _, X = propagate(ec, fg, nothing, X, nothing, ec.aggr)
    X
end

(l::EdgeConv)(fg::FeaturedGraph) = FeaturedGraph(fg, nf = l(fg, node_feature(fg)))

function Base.show(io::IO, l::EdgeConv)
    print(io, "EdgeConv(", l.nn)
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end


"""
    GINConv(nn; eps = 0f0)

Graph Isomorphism Network.

# Arguments

- `nn`: A neural network/layer.
- `eps`: Weighting factor.

The definition of this is as defined in the original paper,
Xu et. al. (2018) https://arxiv.org/abs/1810.00826.
"""
struct GINConv{R<:Real} <: MessagePassing
    nn
    eps::R
end

@functor GINConv
Flux.trainable(g::GINConv) = (nn=g.nn,)

function GINConv(nn; eps=0f0)
    GINConv(nn, eps)
end

message(g::GINConv, x_i, x_j) = x_j 
update(g::GINConv, m, x) = g.nn((1 + g.eps) * x + m)

function (g::GINConv)(fg::FeaturedGraph, X::AbstractMatrix)
    check_num_nodes(fg, X)
    _, X = propagate(g, fg, nothing, X, nothing, +)
    X
end

(l::GINConv)(fg::FeaturedGraph) = FeaturedGraph(fg, nf = l(fg, node_feature(fg)))
