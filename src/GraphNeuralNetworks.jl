module GraphNeuralNetworks

using Statistics: mean
using LinearAlgebra, Random
using Flux
using Flux: glorot_uniform, leakyrelu, GRUCell, @functor, batch
using MacroTools: @forward
using NNlib
using NNlib: scatter, gather
using ChainRulesCore
using Reexport
using DataStructures: nlargest

@reexport using GNNGraphs
using GNNGraphs: COO_T, ADJMAT_T, SPARSE_T,
                  check_num_nodes, check_num_edges,
                  EType, NType # for heteroconvs

export
# utils
      reduce_nodes,
      reduce_edges,
      softmax_nodes,
      softmax_edges,
      broadcast_nodes,
      broadcast_edges,
      softmax_edge_neighbors,

# msgpass
      apply_edges,
      aggregate_neighbors,
      propagate,
      copy_xj,
      copy_xi,
      xi_dot_xj,
      xi_sub_xj,
      xj_sub_xi,
      e_mul_xj,
      w_mul_xj,

# layers/basic
      GNNLayer,
      GNNChain,
      WithGraph,
      DotDecoder,

# layers/conv
      AGNNConv,
      CGConv,
      ChebConv,
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
      TransformerConv,
      DConv,

# layers/heteroconv
      HeteroGraphConv,

# layers/temporalconv
      TGCN,
      A3TGCN,
      GConvLSTM,
      GConvGRU,

# layers/pool
      GlobalPool,
      GlobalAttentionPool,
      Set2Set,
      TopKPool,
      topk_index,

# mldatasets
      mldataset2gnngraph

include("utils.jl")
include("layers/basic.jl")
include("layers/conv.jl")
include("layers/heteroconv.jl")
include("layers/temporalconv.jl")
include("layers/pool.jl")
include("msgpass.jl")
include("mldatasets.jl")
include("deprecations.jl")

end
