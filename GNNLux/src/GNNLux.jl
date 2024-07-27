module GNNLux
using ConcreteStructs: @concrete
using NNlib: NNlib, sigmoid, relu
using LuxCore: LuxCore, AbstractExplicitLayer, AbstractExplicitContainerLayer
using Lux: Lux, Dense, glorot_uniform, zeros32
using Reexport: @reexport
using Random: AbstractRNG
using GNNlib: GNNlib
@reexport using GNNGraphs

include("layers/basic.jl")
export GNNLayer, 
       GNNContainerLayer, 
       GNNChain

include("layers/conv.jl")
export AGNNConv,
       CGConv,
       ChebConv,
       GCNConv,
       GraphConv

end #module
 