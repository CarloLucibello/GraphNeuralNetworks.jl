
const COO_T = Tuple{T, T, V} where {T <: AbstractVector{<:Integer}, V}
const ADJLIST_T = AbstractVector{T} where {T <: AbstractVector{<:Integer}}
const ADJMAT_T = AbstractMatrix
const SPARSE_T = AbstractSparseMatrix # subset of ADJMAT_T
const CUMAT_T = Union{CUDA.AnyCuMatrix, CUDA.CUSPARSE.CuSparseMatrix}

const AVecI = AbstractVector{<:Integer}

# All concrete graph types should be subtypes of AbstractGNNGraph{T}.
# GNNGraph and GNNHeteroGraph are the two concrete types.
abstract type AbstractGNNGraph{T} <: AbstractGraph{Int} end
