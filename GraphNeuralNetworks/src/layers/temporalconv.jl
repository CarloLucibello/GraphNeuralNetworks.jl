function scan(cell, g::GNNGraph, x::AbstractArray{T,3}, state) where {T}
    y = []
    for x_t in eachslice(x, dims = 2)
        yt, state = cell(g, x_t, state)
        y = vcat(y, [yt])
    end
    return stack(y, dims = 2)
end

"""
    GConvGRUCell(in => out, k; [bias, init])

Graph Convolutional Gated Recurrent Unit (GConvGRU) recurrent cell from the paper 
[Structured Sequence Modeling with Graph Convolutional Recurrent Networks](https://arxiv.org/abs/1612.07659).

Uses [`ChebConv`](@ref) to model spatial dependencies, 
followed by a Gated Recurrent Unit (GRU) cell to model temporal dependencies.

# Arguments

- `in => out`: A pair  where `in` is the number of input node features and `out` 
  is the number of output node features.
- `k`: Chebyshev polynomial order.
- `bias`: Add learnable bias. Default `true`.
- `init`: Weights' initializer. Default `glorot_uniform`.

# Forward 

    cell(g::GNNGraph, x, [h])

- `g`: The input graph.
- `x`: The node features. It should be a matrix of size `in x num_nodes`.
- `h`: The initial hidden state of the GRU cell. If given, it is a matrix of size `out x num_nodes`.
       If not provided, it is assumed to be a matrix of zeros.

Performs one recurrence step and returns a tuple `(h, h)`, 
where `h` is the updated hidden state of the GRU cell.

# Examples

```jldoctest
julia> using GraphNeuralNetworks, Flux

julia> num_nodes, num_edges = 5, 10;

julia> d_in, d_out = 2, 3;

julia> timesteps = 5;

julia> g = rand_graph(num_nodes, num_edges);

julia> x = [rand(Float32, d_in, num_nodes) for t in 1:timesteps];

julia> cell = GConvGRUCell(d_in => d_out, 2);

julia> state = Flux.initialstates(cell);

julia> y = state;

julia> for xt in x
           y, state = cell(g, xt, state)
       end

julia> size(y) # (d_out, num_nodes)
(3, 5)
```
"""
@concrete struct GConvGRUCell <: GNNLayer
    conv_x_r
    conv_h_r
    conv_x_z
    conv_h_z
    conv_x_h
    conv_h_h
    k::Int
    in::Int
    out::Int
end

Flux.@layer :noexpand GConvGRUCell

function GConvGRUCell(ch::Pair{Int, Int}, k::Int;
                   bias::Bool = true,
                   init = Flux.glorot_uniform,
                   )
    in, out = ch
    # reset gate
    conv_x_r = ChebConv(in => out, k; bias, init)
    conv_h_r = ChebConv(out => out, k; bias, init)
    # update gate
    conv_x_z = ChebConv(in => out, k; bias, init)
    conv_h_z = ChebConv(out => out, k; bias, init)
    # new gate
    conv_x_h = ChebConv(in => out, k; bias, init)
    conv_h_h = ChebConv(out => out, k; bias, init)
    return GConvGRUCell(conv_x_r, conv_h_r, conv_x_z, conv_h_z, conv_x_h, conv_h_h, k, in, out)
end

function Flux.initialstates(cell::GConvGRUCell)
    zeros_like(cell.conv_x_r.weight, cell.out)
end

(cell::GConvGRUCell)(g::GNNGraph, x::AbstractMatrix) = cell(g, x, initialstates(cell))

function (cell::GConvGRUCell)(g::GNNGraph, x::AbstractMatrix, h::AbstractVector)
    h = repeat(h, 1, g.num_nodes)
    return cell(g, x, h)
end

function (cell::GConvGRUCell)(g::GNNGraph, x::AbstractMatrix, h::AbstractMatrix)
    # reset gate
    r = cell.conv_x_r(g, x) .+ cell.conv_h_r(g, h)
    r = Flux.sigmoid_fast(r)
    # update gate
    z = cell.conv_x_z(g, x) .+ cell.conv_h_z(g, h)
    z = Flux.sigmoid_fast(z)
    # new gate
    h̃ = cell.conv_x_h(g, x) .+ cell.conv_h_h(g, r .* h)
    h̃ = Flux.tanh_fast(h̃)
    h = (1 .- z) .* h̃ .+ z .* h 
    return h, h
end

function Base.show(io::IO, cell::GConvGRUCell)
    print(io, "GConvGRUCell($(cell.in) => $(cell.out), $(cell.k))")
end

"""
    GConvGRU(in => out, k; kws...)

The recurrent layer corresponding to the [`GConvGRUCell`](@ref) cell, 
used to process an entire temporal sequence of node features at once.

The arguments are the same as for [`GConvGRUCell`](@ref).

# Forward 

    layer(g::GNNGraph, x, [h])

- `g`: The input graph.
- `x`: The time-varying node features. It should be an array of size `in x timesteps x num_nodes`.
- `h`: The initial hidden state of the GRU cell. If given, it is a matrix of size `out x num_nodes`.
       If not provided, it is assumed to be a matrix of zeros.

Applies the recurrent cell to each timestep of the input sequence and returns the output as
an array of size `out x timesteps x num_nodes`.

# Examples

```jldoctest
julia> num_nodes, num_edges = 5, 10;

julia> d_in, d_out = 2, 3;

julia> timesteps = 5;

julia> g = rand_graph(num_nodes, num_edges);

julia> x = rand(Float32, d_in, timesteps, num_nodes);

julia> layer = GConvGRU(d_in => d_out, 2);

julia> y = layer(g, x);

julia> size(y) # (d_out, timesteps, num_nodes)
(3, 5, 5)
```
""" 
struct GConvGRU{G <: GConvGRUCell} <: GNNLayer
    cell::G
end

Flux.@layer GConvGRU

function GConvGRU(ch::Pair{Int,Int}, k::Int; kws...)
    return GConvGRU(GConvGRUCell(ch, k; kws...))
end

Flux.initialstates(rnn::GConvGRU) = Flux.initialstates(rnn.cell)

function (rnn::GConvGRU)(g::GNNGraph, x::AbstractArray)
    return scan(rnn.cell, g, x, initialstates(rnn))
end

function Base.show(io::IO, rnn::GConvGRU)
    print(io, "GConvGRU($(rnn.cell.in) => $(rnn.cell.out), $(rnn.cell.k))")
end

