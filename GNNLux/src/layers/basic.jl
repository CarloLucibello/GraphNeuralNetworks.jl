"""
    abstract type GNNLayer <: AbstractExplicitLayer end

An abstract type from which graph neural network layers are derived.
It is Derived from Lux's `AbstractExplicitLayer` type.

See also [`GNNChain`](@ref GNNLux.GNNChain).
"""
abstract type GNNLayer <: AbstractExplicitLayer end
