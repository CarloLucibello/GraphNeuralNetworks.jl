module GNNlibAMDGPUExt

using AMDGPU: AnyROCMatrix
using Random, Statistics, LinearAlgebra
using GNNlib: GNNlib, propagate, copy_xj, e_mul_xj, w_mul_xj
using GNNGraphs: GNNGraph, COO_T, SPARSE_T

###### PROPAGATE SPECIALIZATIONS ####################

## COPY_XJ 

## avoid the fast path on gpu until we have better cuda support
function GNNlib.propagate(::typeof(copy_xj), g::GNNGraph{<:Union{COO_T, SPARSE_T}}, ::typeof(+),
        xi, xj::AnyROCMatrix, e)
    propagate((xi, xj, e) -> copy_xj(xi, xj, e), g, +, xi, xj, e)
end

## E_MUL_XJ 

## avoid the fast path on gpu until we have better cuda support
function GNNlib.propagate(::typeof(e_mul_xj), g::GNNGraph{<:Union{COO_T, SPARSE_T}}, ::typeof(+),
        xi, xj::AnyROCMatrix, e::AbstractVector)
    propagate((xi, xj, e) -> e_mul_xj(xi, xj, e), g, +, xi, xj, e)
end

## W_MUL_XJ 

## avoid the fast path on gpu until we have better support
function GNNlib.propagate(::typeof(w_mul_xj), g::GNNGraph{<:Union{COO_T, SPARSE_T}}, ::typeof(+),
        xi, xj::AnyROCMatrix, e::Nothing)
    propagate((xi, xj, e) -> w_mul_xj(xi, xj, e), g, +, xi, xj, e)
end

end #module
