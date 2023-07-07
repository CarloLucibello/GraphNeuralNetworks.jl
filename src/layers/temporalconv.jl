function set_hidden_state(x::AbstractMatrix, h::Union{AbstractMatrix, Nothing}, out::Int)
    if h === nothing
        h = zeros(eltype(x), out, size(x, 2))
    end
    return h
end

function compute_hidden_state(u::AbstractMatrix, h::AbstractMatrix, c::AbstractMatrix)
    h = u .* h .+ (1 .- u) .* c
    return h
end

struct TGCNCell <: GNNLayer
    conv::GCNConv
    dense_update_gate::Dense
    dense_reset_gate::Dense
    dense_candidate_state::Dense
    out::Int
end

Flux.@functor TGCNCell

function TGCNCell(ch::Pair{Int, Int}; init = Flux.glorot_uniform,
                  bias::Bool = true,
                  add_self_loops = false,
                  use_edge_weight = true)
    in, out = ch
    conv = GCNConv(in => out, sigmoid; init, bias, add_self_loops,
                   use_edge_weight)
    dense_update_gate = Dense(2 * out => out, sigmoid)
    dense_reset_gate = Dense(2 * out => out, sigmoid)
    dense_candidate_state = Dense(2 * out => out, tanh)
    return TGCNCell(conv, dense_update_gate, dense_reset_gate, dense_candidate_state, out)
end

function (tgcn::TGCNCell)(g::GNNGraph, x::AbstractArray; h = nothing)
    h = set_hidden_state(x, h, tgcn.out)
    x̃ = tgcn.conv(g, x)
    u = tgcn.dense_update_gate([x̃; h])
    r = tgcn.dense_reset_gate([x̃; h])
    c = tgcn.dense_candidate_state([x̃; h .* r])
    h = compute_hidden_state(u, h, c)
    return h
end

