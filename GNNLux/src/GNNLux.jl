module GNNLux
using ConcreteStructs: @concrete
using NNlib: NNlib
using LuxCore: LuxCore, AbstractExplicitLayer, AbstractExplicitContainerLayer
using Lux: Lux, glorot_uniform, zeros32
using Reexport: @reexport
using Random: AbstractRNG
using GNNlib: GNNlib
@reexport using GNNGraphs

include("layers/basic.jl")
export GNNLayer, 
       GNNContainerLayer, 
       GNNChain

include("layers/conv.jl")
export GCNConv,
       ChebConv,
       GraphConv

end #module
 