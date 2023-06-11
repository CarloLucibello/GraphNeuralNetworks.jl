using GraphNeuralNetworks
using Flux
using Random, Statistics, LinearAlgebra
using SparseArrays

"""
A type representing a conjunctive normal form.
"""
struct CNF
    N::Int # num variables
    M::Int # num factors
    clauses::Vector{Vector{Int}}
end

function CNF(clauses::Vector{Vector{Int}})
    M = length(clauses)
    N = maximum(maximum(abs.(c)) for c in clauses)
    return CNF(N, M, clauses)
end

"""
    randomcnf(; N=100, k=3, α=0.1, seed=-1, planted = Vector{Vector{Int}}())

Generates a random instance of the k-SAT problem, with `N` variables and `αN` clauses.
Any configuration in `planted` is guaranteed to be a solution of the problem.
"""
function randomcnf(; N::Int = 100, k::Int = 3, α::Float64 = 0.1, seed::Int=-1,
                    planted = Vector{Vector{Int}}())
    seed > 0 && Random.seed!(seed)
    M = round(Int, N*α)
    clauses = Vector{Vector{Int}}()
    for p in planted
        @assert length(p) == N   "Wrong size for planted configurations ($N != $(lenght(p)) )"
    end
    for a=1:M
        while true
            c = rand(1:N, k)
            length(union(c)) != k && continue
            c = c .* rand([-1,1], k)

            # reject if not satisfies the planted solutions
            sat = Bool[any(i -> i>0, sol[abs.(c)] .* c) for sol in planted]
            !all(sat) && continue

            push!(clauses, c)
            break
        end
    end
    return CNF(N, M, clauses)
end


function to_edge_index(cnf::CNF)
    N = cnf.N
    srcV, dstF = Vector{Int}(), Vector{Int}()
    srcF, dstV = Vector{Int}(), Vector{Int}()
    for (a, c) in enumerate(cnf.clauses)
        for v in c
            negated = v < 0
            push!(srcV, abs(v) + N*negated)
            push!(dstF, a)
            push!(srcF, a)
            push!(dstV, abs(v) + N*negated)
        end
    end
    return srcV, dstF,srcV, dstF
end

function to_adjacency_matrix(cnf::CNF)
    M, N = cnf.M, cnf.N
    A = spzeros(Int, M, 2*N)
    for (a, c) in enumerate(cnf.clauses)
        for v in c
            negated = v < 0
            A[a, abs(v) + N*negated] = 1
        end
    end
    return A
end

function flip_literals(X::AbstractMatrix)
    n = size(X, 2) ÷ 2
    return hcat(X[:,n+1:2n], X[:,1:n])
end

## Layer
struct NeuroSAT
    Xv0
    Xf0
    MLPv
    MLPf
    MLPout
    LSTMv
    LSTMf
end

# A  # rectangular adjacency matrix
Flux.@functor NeuroSAT

# Optimisers.trainable(m::NeuroSAT) = (; m.MLPv, m.MLPf, m.MLPout, m.LSTMv, m.LSTMf)

function NeuroSAT(D::Int)
    Xv0 = randn(Float32, D)
    Xf0 = randn(Float32, D)
    MLPv = Chain(Dense(D => 4D, relu), Dense(4D => D))
    MLPf = Chain(Dense(D => 4D, relu), Dense(4D => D))
    MLPout = Chain(Dense(D => 4D, relu), Dense(4D => 1))
    LSTMv = LSTM(2D => D)
    LSTMf = LSTM(D => D)
    return NeuroSAT(Xv0, Xf0, MLPv, MLPf, MLPout, LSTMv, LSTMf)
end

function (m::NeuroSAT)(A::AbstractArray, Tmax)
    Xv = repeat(m.Xv0, 1, size(A, 2))
    # Xf = repeat(m.Xf0, 1, size(A, 1))
    
    for t = 1:Tmax
        Xv = m.MLPv(Xv)
        Mf = Xv * A'
        Xf = m.MLPf(m.LSTMf(Mf))
        Mv = Xf * A
        Xv = m.LSTMv(vcat(Mv, flip_literals(Xv)))
    end
    return mean(m.MLPout(Xv))
end

# function Base.show(io, m::NeuroSAT)
#     D = size(m.Xv0, 1) 
#     print(io, "NeuroSAT($(D))")
# end

N = 100 
cnf = randomcnf(; N, k=3, α=1.5, seed=-1)
M = cnf.M
D = 32     # 128 nel paper
Xv = randn(Float32, D, 2*N)
Xf = randn(Float32, D, M)

srcV, dstF, srcF, dstV = to_edge_index(cnf)
A = to_adjacency_matrix(cnf)


model = NeuroSAT(D)

m_vtof = GNNGraphs._gather(Xv, srcV)
m_ftov = GNNGraphs._gather(Xf, srcF)


