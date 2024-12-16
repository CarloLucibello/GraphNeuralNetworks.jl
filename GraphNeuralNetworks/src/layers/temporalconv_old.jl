struct TGCNCell{C,G} <: GNNLayer
    conv::C
    gru::G
    in::Int
    out::Int
end

Flux.@layer :noexpand TGCNCell

function TGCNCell(ch::Pair{Int, Int};
                  bias::Bool = true,
                  init = Flux.glorot_uniform,
                  add_self_loops = false)
    in, out = ch
    conv = GCNConv(in => out, sigmoid; init, bias, add_self_loops)
    gru = GRUCell(out => out)
    return TGCNCell(conv, gru, in, out)
end

Flux.initialstates(cell::TGCNCell) = initialstates(cell.gru)
(cell::TGCNCell)(g::GNNGraph, x::AbstractVecOrMat) = cell(g, x, initialstates(cell))

function (cell::TGCNCell)(g::GNNGraph, x::AbstractVecOrMat, h::AbstractVecOrMat)
    x = cell.conv(g, x)
    h = cell.gru(x, h)
    return h
end

function Base.show(io::IO, cell::TGCNCell)
    print(io, "TGCNCell($(cell.in) => $(cell.out))")
end

"""
    TGCN(in => out; [bias, init, add_self_loops])

Temporal Graph Convolutional Network (T-GCN) recurrent layer from the paper [T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction](https://arxiv.org/pdf/1811.05320.pdf).

Performs a layer of GCNConv to model spatial dependencies, followed by a Gated Recurrent Unit (GRU) cell to model temporal dependencies.

# Arguments

- `in`: Number of input features.
- `out`: Number of output features.
- `bias`: Add learnable bias. Default `true`.
- `init`: Convolution's weights initializer. Default `glorot_uniform`.
- `add_self_loops`: Add self loops to the graph before performing the convolution. Default `false`.

# Forward 

    tgcn(g::GNNGraph, x, [h])

- `g`: The input graph.
- `x`: The input to the TGCN. It should be a matrix size `in x timesteps` or an array of size `in x timesteps x num_nodes`.
- `h`: The initial hidden state of the GRU cell. If given, it is a vector of size `out` or a matrix of size `out x num_nodes`.
       If not provided, it is assumed to be a vector of zeros.

# Examples

```jldoctest
julia> in, out = 2, 3;

julia> tgcn = TGCN(in => out)
TGCN(
  TGCNCell(
    GCNConv(2 => 3, σ),                 # 9 parameters
    GRUCell(3 => 3),                    # 63 parameters
  ),
)                   # Total: 5 arrays, 72 parameters, 560 bytes.

julia> num_nodes = 5; num_edges = 10; timesteps = 4;

julia> g = rand_graph(num_nodes, num_edges);

julia> x = rand(Float32, in, timesteps, num_nodes);

julia> tgcn(g, x) |> size
(3, 4, 5)
```
"""
struct TGCN{C<:TGCNCell} <: GNNLayer
    cell::C
end

Flux.@layer TGCN

TGCN(ch::Pair{Int, Int}; kws...) = TGCN(TGCNCell(ch; kws...))

initialstates(tgcn::TGCN) = initialstates(tgcn.cell)

(tgcn::TGCN)(g::GNNGraph, x) = tgcn(g, x, initialstates(tgcn))

function (tgcn::TGCN)(g::GNNGraph, x::AbstractArray, h) 
    return scan(tgcn.cell, g, x, h)
end

Base.show(io::IO, tgcn::TGCN) = print(io, "TGCN($(tgcn.cell.in) => $(tgcn.cell.out))")

####### TO BE PORTED TO FLUX v0.15 from here ############################

"""
    A3TGCN(in => out; [bias, init, add_self_loops])

Attention Temporal Graph Convolutional Network (A3T-GCN) model from the paper [A3T-GCN: Attention Temporal Graph
Convolutional Network for Traffic Forecasting](https://arxiv.org/pdf/2006.11583.pdf).

Performs a TGCN layer, followed by a soft attention layer.

# Arguments

- `in`: Number of input features.
- `out`: Number of output features.
- `bias`: Add learnable bias. Default `true`.
- `init`: Convolution's weights initializer. Default `glorot_uniform`.
- `add_self_loops`: Add self loops to the graph before performing the convolution. Default `false`.

# Examples

```jldoctest
julia> in, out = 2, 3;

julia> model = A3TGCN(in => out)
TGCN(
  TGCNCell(
    GCNConv(2 => 3, σ),                 # 9 parameters
    GRUCell(3 => 3),                    # 63 parameters
  ),
)                   # Total: 5 arrays, 72 parameters, 560 bytes.

julia> num_nodes = 5; num_edges = 10; timesteps = 4;

julia> g = rand_graph(num_nodes, num_edges);

julia> x = rand(Float32, in, timesteps, num_nodes);

julia> model(g, x) |> size
(3, 4, 5)
```
"""
struct A3TGCN <: GNNLayer
    tgcn::TGCN
    dense1::Dense
    dense2::Dense
    in::Int
    out::Int
end

Flux.@layer A3TGCN

function A3TGCN(ch::Pair{Int, Int}; kws...)
    in, out = ch
    tgcn = TGCN(in => out; kws...)
    dense1 = Dense(out => out)
    dense2 = Dense(out => out)
    return A3TGCN(tgcn, dense1, dense2, in, out)
end

function (a3tgcn::A3TGCN)(g::GNNGraph, x::AbstractArray, h)
    h = a3tgcn.tgcn(g, x, h)  # [out, timesteps, num_nodes]
    logits = a3tgcn.dense1(h)
    logits = a3tgcn.dense2(logits)      # [out, timesteps, num_nodes]
    a = softmax(logits, dims=2)     # TODO handle multiple graphs
    c = sum(a .* h, dims=2)
    c = dropdims(c, dims=2)    # [out, num_nodes]
    return c
end

function Base.show(io::IO, a3tgcn::A3TGCN)
    print(io, "A3TGCN($(a3tgcn.in) => $(a3tgcn.out))")
end


"""
    EvolveGCNO(ch; bias = true, init = glorot_uniform, init_state = Flux.zeros32)

Evolving Graph Convolutional Network (EvolveGCNO) layer from the paper [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](https://arxiv.org/pdf/1902.10191).

Perfoms a Graph Convolutional layer with parameters derived from a Long Short-Term Memory (LSTM) layer across the snapshots of the temporal graph.


# Arguments

- `in`: Number of input features.
- `out`: Number of output features.
- `bias`: Add learnable bias. Default `true`.
- `init`: Weights' initializer. Default `glorot_uniform`.
- `init_state`: Initial state of the hidden stat of the LSTM layer. Default `zeros32`.

# Examples

```jldoctest
julia> tg = TemporalSnapshotsGNNGraph([rand_graph(10,20; ndata = rand(4,10)), rand_graph(10,14; ndata = rand(4,10)), rand_graph(10,22; ndata = rand(4,10))])
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10]
  num_edges: [20, 14, 22]
  num_snapshots: 3

julia> ev = EvolveGCNO(4 => 5)
EvolveGCNO(4 => 5)

julia> size(ev(tg, tg.ndata.x))
(3,)

julia> size(ev(tg, tg.ndata.x)[1])
(5, 10)
```
"""
struct EvolveGCNO
    conv
    W_init
    init_state
    in::Int
    out::Int
    Wf
    Uf
    Bf
    Wi
    Ui
    Bi
    Wo
    Uo
    Bo
    Wc
    Uc
    Bc
end
 
function EvolveGCNO(ch; bias = true, init = glorot_uniform, init_state = Flux.zeros32)
    in, out = ch
    W = init(out, in)
    conv = GCNConv(ch; bias = bias, init = init)
    Wf = init(out, in)
    Uf = init(out, in)
    Bf = bias ? init(out, in) : nothing
    Wi = init(out, in)
    Ui = init(out, in)
    Bi = bias ? init(out, in) : nothing
    Wo = init(out, in)
    Uo = init(out, in)
    Bo = bias ? init(out, in) : nothing
    Wc = init(out, in)
    Uc = init(out, in)
    Bc = bias ? init(out, in) : nothing
    return EvolveGCNO(conv, W, init_state, in, out, Wf, Uf, Bf, Wi, Ui, Bi, Wo, Uo, Bo, Wc, Uc, Bc)
end

function (egcno::EvolveGCNO)(tg::TemporalSnapshotsGNNGraph, x)
    H = egcno.init_state(egcno.out, egcno.in)
    C = egcno.init_state(egcno.out, egcno.in)
    W = egcno.W_init
    X = map(1:tg.num_snapshots) do i
        F = Flux.sigmoid_fast.(egcno.Wf .* W + egcno.Uf .* H + egcno.Bf)
        I = Flux.sigmoid_fast.(egcno.Wi .* W + egcno.Ui .* H + egcno.Bi)
        O = Flux.sigmoid_fast.(egcno.Wo .* W + egcno.Uo .* H + egcno.Bo)
        C̃ = Flux.tanh_fast.(egcno.Wc .* W + egcno.Uc .* H + egcno.Bc)
        C = F .* C + I .* C̃
        H = O .* tanh_fast.(C)
        W = H
        egcno.conv(tg.snapshots[i], x[i]; conv_weight = H)
    end
    return X
end
 
function Base.show(io::IO, egcno::EvolveGCNO)
    print(io, "EvolveGCNO($(egcno.in) => $(egcno.out))")
end
