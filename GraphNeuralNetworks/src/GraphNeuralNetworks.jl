module GraphNeuralNetworks

using Statistics: mean
using LinearAlgebra, Random
using Flux
using Flux: glorot_uniform, leakyrelu, GRUCell, batch, initialstates
using MacroTools: @forward
using NNlib
using ChainRulesCore
using Reexport: @reexport
using MLUtils: zeros_like
using ConcreteStructs: @concrete

using GNNGraphs:  COO_T, ADJMAT_T, SPARSE_T,
                  check_num_nodes, check_num_edges,
                  EType, NType # for heteroconvs

@reexport using GNNGraphs
@reexport using GNNlib

include("layers/basic.jl")
export GNNLayer,
       GNNChain,
       WithGraph,
       DotDecoder

include("layers/conv.jl")
export AGNNConv,
       CGConv,
       ChebConv,
       DConv,
       EdgeConv,
       EGNNConv,
       GATConv,
       GATv2Conv,
       GatedGraphConv,
       GCNConv,
       GINConv,
       GMMConv,
       GraphConv,
       MEGNetConv,
       NNConv,
       ResGatedGraphConv,
       SAGEConv,
       SGConv,
       TAGConv,
       TransformerConv

include("layers/heteroconv.jl")
export HeteroGraphConv

include("layers/temporalconv.jl")
export GNNRecurrence,
       GConvGRU, GConvGRUCell,
       GConvLSTM, GConvLSTMCell,
       DCGRU, DCGRUCell,
       EvolveGCNO, EvolveGCNOCell,
       TGCN, TGCNCell

include("layers/pool.jl")
export GlobalPool,
       GlobalAttentionPool,
       Set2Set,
       TopKPool,
       topk_index

include("deprecations.jl")

end
