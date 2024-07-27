# Missing Layers

# | Layer                       |Sparse Ops|Edge Weight|Edge Features| Heterograph  | TemporalSnapshotsGNNGraphs |
# | :--------                   |  :---:   |:---:      |:---:        |  :---:       | :---:                      |
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

_getbias(ps) = hasproperty(ps, :bias) ? getproperty(ps, :bias) : false

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
        return (; weight, bias)
    else
        return (; weight)
    end
end

LuxCore.parameterlength(l::GCNConv) = l.use_bias ? l.in_dims * l.out_dims + l.out_dims : l.in_dims * l.out_dims
LuxCore.statelength(d::GCNConv) = 0
LuxCore.outputsize(d::GCNConv) = (d.out_dims,)

function Base.show(io::IO, l::GCNConv)
    print(io, "GCNConv(", l.in_dims, " => ", l.out_dims)
    l.σ == identity || print(io, ", ", l.σ)
    l.use_bias || print(io, ", use_bias=false")
    l.add_self_loops || print(io, ", add_self_loops=false")
    !l.use_edge_weight || print(io, ", use_edge_weight=true")
    print(io, ")")
end

(l::GCNConv)(g, x, ps, st; conv_weight=nothing, edge_weight=nothing, norm_fn= d -> 1 ./ sqrt.(d)) = 
    l(g, x, edge_weight, ps, st; conv_weight, norm_fn)

function (l::GCNConv)(g, x, edge_weight, ps, st; 
            norm_fn = d -> 1 ./ sqrt.(d),
            conv_weight=nothing, )

    m = (; ps.weight, bias = _getbias(ps), 
           l.add_self_loops, l.use_edge_weight, l.σ)
    y = GNNlib.gcn_conv(m, g, x, edge_weight, norm_fn, conv_weight)
    return y, st
end

@concrete struct ChebConv <: GNNLayer
    in_dims::Int
    out_dims::Int
    use_bias::Bool
    k::Int
    init_weight
    init_bias
end

function ChebConv(ch::Pair{Int, Int}, k::Int;
                  init_weight = glorot_uniform,
                  init_bias = zeros32,
                  use_bias::Bool = true)
    in_dims, out_dims = ch
    return ChebConv(in_dims, out_dims, use_bias, k, init_weight, init_bias)
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
    print(io, "ChebConv(", l.in_dims, " => ", l.out_dims, ", k=", l.k)
    l.use_bias || print(io, ", use_bias=false")
    print(io, ")")
end

function (l::ChebConv)(g, x, ps, st)
    m = (; ps.weight, bias = _getbias(ps), l.k)
    y = GNNlib.cheb_conv(m, g, x)
    return y, st

end

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

function (l::GraphConv)(g, x, ps, st)
    m = (; ps.weight1, ps.weight2, bias = _getbias(ps), 
          l.σ, l.aggr)
    return GNNlib.graph_conv(m, g, x), st
end


@concrete struct AGNNConv <: GNNLayer
    init_beta <: AbstractVector
    add_self_loops::Bool
    trainable::Bool
end

function AGNNConv(; init_beta = 1.0f0, add_self_loops = true, trainable = true)
    return AGNNConv([init_beta], add_self_loops, trainable)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::AGNNConv)
    if l.trainable
        return (; β = l.init_beta)
    else
        return (;)
    end
end

LuxCore.parameterlength(l::AGNNConv) = l.trainable ? 1 : 0
LuxCore.statelength(d::AGNNConv) = 0

function Base.show(io::IO, l::AGNNConv)
    print(io, "AGNNConv(", l.init_beta)
    l.add_self_loops || print(io, ", add_self_loops=false")
    l.trainable || print(io, ", trainable=false")
    print(io, ")")
end

function (l::AGNNConv)(g, x::AbstractMatrix, ps, st)
    β = l.trainable ? ps.β : l.init_beta
    m = (;  β, l.add_self_loops)
    return GNNlib.agnn_conv(m, g, x), st
end

@concrete struct CGConv <: GNNContainerLayer{(:dense_f, :dense_s)}
    in_dims::NTuple{2, Int}
    out_dims::Int
    dense_f
    dense_s
    residual::Bool
    init_weight
    init_bias
end

CGConv(ch::Pair{Int, Int}, args...; kws...) = CGConv((ch[1], 0) => ch[2], args...; kws...)

function CGConv(ch::Pair{NTuple{2, Int}, Int}, act = identity; residual = false,
                use_bias = true, init_weight = glorot_uniform, init_bias = zeros32, 
                allow_fast_activation = true)
    (nin, ein), out = ch
    dense_f = Dense(2nin + ein => out, sigmoid; use_bias, init_weight, init_bias, allow_fast_activation)
    dense_s = Dense(2nin + ein => out, act; use_bias, init_weight, init_bias, allow_fast_activation)
    return CGConv((nin, ein), out, dense_f, dense_s, residual, init_weight, init_bias)
end

(l::CGConv)(g, x, ps, st) = l(g, x, nothing, ps, st)

function (l::CGConv)(g, x, e, ps, st)
    dense_f = StatefulLuxLayer(l.dense_f, ps.dense_f)
    dense_s = StatefulLuxLayer(l.dense_s, ps.dense_s)
    m = (; dense_f, dense_s, l.residual)
    return GNNlib.cg_conv(m, g, x, e), st
end
