_getbias(ps) = hasproperty(ps, :bias) ? getproperty(ps, :bias) : false
_getstate(st, name) = hasproperty(st, name) ? getproperty(st, name) : NamedTuple()
_getstate(s::StatefulLuxLayer{true}) = s.st
_getstate(s::StatefulLuxLayer{false}) = s.st_any


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

LuxCore.outputsize(l::CGConv) = (l.out_dims,)

(l::CGConv)(g, x, ps, st) = l(g, x, nothing, ps, st)

function (l::CGConv)(g, x, e, ps, st)
    dense_f = StatefulLuxLayer{true}(l.dense_f, ps.dense_f, _getstate(st, :dense_f))
    dense_s = StatefulLuxLayer{true}(l.dense_s, ps.dense_s, _getstate(st, :dense_s))
    m = (; dense_f, dense_s, l.residual)
    return GNNlib.cg_conv(m, g, x, e), st
end

@concrete struct EdgeConv <: GNNContainerLayer{(:nn,)}
    nn <: AbstractExplicitLayer
    aggr
end

EdgeConv(nn; aggr = max) = EdgeConv(nn, aggr)

function Base.show(io::IO, l::EdgeConv)
    print(io, "EdgeConv(", l.nn)
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end


function (l::EdgeConv)(g::AbstractGNNGraph, x, ps, st)
    nn = StatefulLuxLayer{true}(l.nn, ps, st)
    m = (; nn, l.aggr)
    y = GNNlib.edge_conv(m, g, x)
    stnew = _getstate(nn)
    return y, stnew
end


@concrete struct EGNNConv <: GNNContainerLayer{(:ϕe, :ϕx, :ϕh)}
    ϕe
    ϕx
    ϕh
    num_features
    residual::Bool
end

function EGNNConv(ch::Pair{Int, Int}, hidden_size = 2 * ch[1]; residual = false)
    return EGNNConv((ch[1], 0) => ch[2]; hidden_size, residual)
end

#Follows reference implementation at https://github.com/vgsatorras/egnn/blob/main/models/egnn_clean/egnn_clean.py
function EGNNConv(ch::Pair{NTuple{2, Int}, Int}; hidden_size::Int = 2 * ch[1][1],
                  residual = false)
    (in_size, edge_feat_size), out_size = ch
    act_fn = swish

    # +1 for the radial feature: ||x_i - x_j||^2
    ϕe = Chain(Dense(in_size * 2 + edge_feat_size + 1 => hidden_size, act_fn),
               Dense(hidden_size => hidden_size, act_fn))

    ϕh = Chain(Dense(in_size + hidden_size => hidden_size, swish),
               Dense(hidden_size => out_size))

    ϕx = Chain(Dense(hidden_size => hidden_size, swish),
               Dense(hidden_size => 1, use_bias = false))

    num_features = (in = in_size, edge = edge_feat_size, out = out_size,
                    hidden = hidden_size)
    if residual
        @assert in_size==out_size "Residual connection only possible if in_size == out_size"
    end
    return EGNNConv(ϕe, ϕx, ϕh, num_features, residual)
end

LuxCore.outputsize(l::EGNNConv) = (l.num_features.out,)

(l::EGNNConv)(g, h, x, ps, st) = l(g, h, x, nothing, ps, st)

function (l::EGNNConv)(g, h, x, e, ps, st)
    ϕe = StatefulLuxLayer{true}(l.ϕe, ps.ϕe, _getstate(st, :ϕe))
    ϕx = StatefulLuxLayer{true}(l.ϕx, ps.ϕx, _getstate(st, :ϕx))
    ϕh = StatefulLuxLayer{true}(l.ϕh, ps.ϕh, _getstate(st, :ϕh))
    m = (; ϕe, ϕx, ϕh, l.residual, l.num_features)
    return GNNlib.egnn_conv(m, g, h, x, e), st
end

function Base.show(io::IO, l::EGNNConv)
    ne = l.num_features.edge
    nin = l.num_features.in
    nout = l.num_features.out
    nh = l.num_features.hidden
    print(io, "EGNNConv(($nin, $ne) => $nout; hidden_size=$nh")
    if l.residual
        print(io, ", residual=true")
    end
    print(io, ")")
end

@concrete struct DConv <: GNNLayer
    in_dims::Int
    out_dims::Int
    k::Int
    init_weight
    init_bias
    use_bias::Bool
end

function DConv(ch::Pair{Int, Int}, k::Int; 
        init_weight = glorot_uniform, 
        init_bias = zeros32,
        use_bias = true)
    in, out = ch
    return DConv(in, out, k, init_weight, init_bias, use_bias)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::DConv)
    weights = l.init_weight(rng, 2, l.k, l.out_dims, l.in_dims)
    if l.use_bias
        bias = l.init_bias(rng, l.out_dims)
        return (; weights, bias)
    else
        return (; weights)
    end
end

LuxCore.parameterlength(l::DConv) = l.use_bias ? l.in_dims * l.out_dims * l.k + l.out_dims : 
                                                l.in_dims * l.out_dims * l.k

function (l::DConv)(g, x, ps, st)
    m = (; ps.weights, bias = _getbias(ps), l.k)
    return GNNlib.d_conv(m, g, x), st
end

function Base.show(io::IO, l::DConv)
    print(io, "DConv($(l.in) => $(l.out), k=$(l.k))")
end

@concrete struct GATConv <: GNNLayer
    dense_x
    dense_e
    init_weight
    init_bias
    use_bias::Bool
    σ
    negative_slope
    channel::Pair{NTuple{2, Int}, Int}
    heads::Int
    concat::Bool
    add_self_loops::Bool
    dropout
end


GATConv(ch::Pair{Int, Int}, args...; kws...) = GATConv((ch[1], 0) => ch[2], args...; kws...)

function GATConv(ch::Pair{NTuple{2, Int}, Int}, σ = identity;
                 heads::Int = 1, concat::Bool = true, negative_slope = 0.2,
                 init_weight = glorot_uniform, init_bias = zeros32,
                 use_bias::Bool = true, 
                 add_self_loops = true, dropout=0.0)
    (in, ein), out = ch
    if add_self_loops
        @assert ein==0 "Using edge features and setting add_self_loops=true at the same time is not yet supported."
    end

    dense_x = Dense(in => out * heads, use_bias = false)
    dense_e = ein > 0 ? Dense(ein => out * heads, use_bias = false) : nothing
    negative_slope = convert(Float32, negative_slope)
    return GATConv(dense_x, dense_e, init_weight, init_bias, use_bias, 
                 σ, negative_slope, ch, heads, concat, add_self_loops, dropout)
end

# Flux.trainable(l::GATConv) = (dense_x = l.dense_x, dense_e = l.dense_e, bias = l.bias, a = l.a)
function LuxCore.initialparameters(rng::AbstractRNG, l::GATConv)
    (in, ein), out = l.channel
    dense_x = initialparameters(rng, l.dense_x)
    a = init_weight(ein > 0 ? 3out : 2out, heads)
    ps = (; dense_x, a)
    if ein > 0
        ps = (ps..., dense_e = initialparameters(rng, l.dense_e))
    end
    if use_bias
        ps = (ps..., bias = l.init_bias(rng, concat ? out * l.heads : out))
    end
    return ps
end

(l::GATConv)(g, x, ps, st) = l(g, x, nothing, ps, st)

function (l::GATConv)(g, x, e, ps, st) 
    return GNNlib.gat_conv(l, g, x, e), st
end

function Base.show(io::IO, l::GATConv)
    (in, ein), out = l.channel
    print(io, "GATConv(", ein == 0 ? in : (in, ein), " => ", out ÷ l.heads)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ", negative_slope=", l.negative_slope)
    print(io, ")")
end
