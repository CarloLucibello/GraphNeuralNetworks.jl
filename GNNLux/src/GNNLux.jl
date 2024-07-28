module GNNLux
using ConcreteStructs: @concrete
using NNlib: NNlib, sigmoid, relu
using LuxCore: LuxCore, AbstractExplicitLayer, AbstractExplicitContainerLayer
using Lux: Lux, Dense, glorot_uniform, zeros32, StatefulLuxLayer
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
       EdgeConv,
       # EGNNConv,
       # DConv,
       # GATConv,
       # GATv2Conv,
       # GatedGraphConv,
       GCNConv,
       # GINConv,
       # GMMConv,
       GraphConv
       # MEGNetConv,
       # NNConv,
       # ResGatedGraphConv,
       # SAGEConv,
       # SGConv,
       # TAGConv,
       # TransformerConv
   

end #module
 