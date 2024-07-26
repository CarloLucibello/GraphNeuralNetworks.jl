# Missing Layers

# | Layer                       |Sparse Ops|Edge Weight|Edge Features| Heterograph  | TemporalSnapshotsGNNGraphs |
# | :--------                   |  :---:   |:---:      |:---:        |  :---:       | :---:                      |
# | [`AGNNConv`](@ref)          |          |           |     ✓       |              |                    |                          
# | [`CGConv`](@ref)            |          |           |     ✓       |       ✓      |             ✓             | 
# | [`EGNNConv`](@ref)          |          |           |     ✓       |              |                           |
# | [`EdgeConv`](@ref)          |          |           |             |       ✓      |                            |  
# | [`GATConv`](@ref)           |          |           |     ✓       |       ✓      |              ✓             |
# | [`GATv2Conv`](@ref)         |          |           |     ✓       |       ✓      |             ✓              |
# | [`GatedGraphConv`](@ref)    |     ✓    |           |             |              |            ✓               |
# | [`GINConv`](@ref)           |     ✓    |           |             |       ✓      |               ✓           |
# | [`GMMConv`](@ref)           |          |           |     ✓       |              |                            |
# | [`MEGNetConv`](@ref)        |          |           |     ✓       |              |                            |              
# | [`NNConv`](@ref)            |          |           |     ✓       |              |                            |
# | [`ResGatedGraphConv`](@ref) |          |           |             |       ✓      |               ✓             |
# | [`SAGEConv`](@ref)          |     ✓    |           |             |       ✓      |             ✓               |
# | [`SGConv`](@ref)            |     ✓    |           |             |              |             ✓             |
# | [`TransformerConv`](@ref)   |          |           |     ✓       |              |                           |


@concrete struct GCNConv <: GNNLayer
    in_dims::Int
    out_dims::Int
    use_bias::Bool
    add_self_loops::Bool
    use_edge_weight::Bool
    init_weight
    init_bias
    σ
end

function GCNConv(ch::Pair{Int, Int}, σ = identity;
                 init_weight = glorot_uniform,
                init_bias = zeros32,
                use_bias::Bool = true,
                add_self_loops::Bool = true,
                use_edge_weight::Bool = false,
                allow_fast_activation::Bool = true)
    in_dims, out_dims = ch
    σ = allow_fast_activation ? NNlib.fast_act(σ) : σ
    return GCNConv(in_dims, out_dims, use_bias, add_self_loops, use_edge_weight, init_weight, init_bias, σ)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::GCNConv)
    weight = l.init_weight(rng, l.out_dims, l.in_dims)
    if l.use_bias
        bias = l.init_bias(rng, l.out_dims)
    else
        bias = false
    end
    return (; weight, bias)
end

LuxCore.parameterlength(l::GCNConv) = l.use_bias ? l.in_dims * l.out_dims + l.out_dims : l.in_dims * l.out_dims
LuxCore.statelength(d::GCNConv) = 0
LuxCore.outputsize(d::GCNConv) = (d.out_dims,)

function Base.show(io::IO, l::GCNConv)
    print(io, "GCNConv(", l.in_dims, " => ", l.out_dims)
    l.σ == identity || print(io, ", ", l.σ)
    l.use_bias || print(io, ", use_bias=false")
    l.add_self_loops || print(io, ", add_self_loops=false")
    l.use_edge_weight || print(io, ", use_edge_weight=true")
    print(io, ")")
end

# TODO norm_fn should be keyword argument
(l::GCNConv)(g, x, ps, st; conv_weight=nothing, edge_weight=nothing, norm_fn= d -> 1 ./ sqrt.(d)) = 
    l(g, x, edge_weight, norm_fn, ps, st; conv_weight)
(l::GCNConv)(g, x, edge_weight, ps, st; conv_weight=nothing, norm_fn = d -> 1 ./ sqrt.(d)) = 
    l(g, x, edge_weight, norm_fn, ps, st; conv_weight)
(l::GCNConv)(g, x, edge_weight, norm_fn, ps, st; conv_weight=nothing) = 
    GNNlib.gcn_conv(l, g, x, edge_weight, norm_fn, conv_weight, ps), st

@concrete struct ChebConv <: GNNLayer
    in_dims::Int
    out_dims::Int
    use_bias::Bool
    k::Int
    init_weight
    init_bias
    σ
end

function ChebConv(ch::Pair{Int, Int}, k::Int, σ = identity;
                  init_weight = glorot_uniform,
                  init_bias = zeros32,
                  use_bias::Bool = true,
                  allow_fast_activation::Bool = true)
    in_dims, out_dims = ch
    σ = allow_fast_activation ? NNlib.fast_act(σ) : σ
    return ChebConv(in_dims, out_dims, use_bias, k, init_weight, init_bias, σ)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::ChebConv)
    weight = l.init_weight(rng, l.out_dims, l.in_dims, l.k)
    if l.use_bias
        bias = l.init_bias(rng, l.out_dims)
        return (; weight, bias)
    else
        return (; weight)
    end
end

LuxCore.parameterlength(l::ChebConv) = l.use_bias ? l.in_dims * l.out_dims * l.k + l.out_dims : 
                                                    l.in_dims * l.out_dims * l.k
LuxCore.statelength(d::ChebConv) = 0
LuxCore.outputsize(d::ChebConv) = (d.out_dims,)

function Base.show(io::IO, l::ChebConv)
    print(io, "ChebConv(", l.in_dims, " => ", l.out_dims, ", K=", l.K)
    l.σ == identity || print(io, ", ", l.σ)
    l.use_bias || print(io, ", use_bias=false")
    print(io, ")")
end

(l::ChebConv)(g, x, ps, st) = GNNlib.cheb_conv(l, g, x, ps), st

@concrete struct GraphConv <: GNNLayer
    in_dims::Int
    out_dims::Int
    use_bias::Bool
    init_weight
    init_bias
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
        return (; weight1, weight2, bias)
    else
        return (; weight1, weight2)
    end
end

function LuxCore.parameterlength(l::GraphConv)
    if l.use_bias
        return 2 * l.in_dims * l.out_dims + l.out_dims
    else
        return 2 * l.in_dims * l.out_dims
    end
end

LuxCore.statelength(d::GraphConv) = 0
LuxCore.outputsize(d::GraphConv) = (d.out_dims,)

function Base.show(io::IO, l::GraphConv)
    print(io, "GraphConv(", l.in_dims, " => ", l.out_dims)
    (l.σ == identity) || print(io, ", ", l.σ)
    (l.aggr == +) || print(io, ", aggr=", l.aggr)
    l.use_bias || print(io, ", use_bias=false")
    print(io, ")")
end

(l::GraphConv)(g, x, ps, st) = GNNlib.graph_conv(l, g, x, ps), st
