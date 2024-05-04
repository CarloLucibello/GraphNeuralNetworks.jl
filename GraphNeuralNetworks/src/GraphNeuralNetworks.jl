module GraphNeuralNetworks

using Statistics: mean
using LinearAlgebra, Random
using Flux
using Flux: glorot_uniform, leakyrelu, GRUCell, @functor, batch
using MacroTools: @forward
using Reexport
using SparseArrays, Graphs # not needed but if removed Documenter will complain

@reexport using GNNlib

using .GNNGraphs: COO_T, ADJMAT_T, SPARSE_T,
                  EType, NType # for heteroconvs

export
# GNNlib/utils
      reduce_nodes,
      reduce_edges,
      softmax_nodes,
      softmax_edges,
      broadcast_nodes,
      broadcast_edges,
      softmax_edge_neighbors,

# GNNlib/msgpass
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
      TransformerConv,

# layers/heteroconv
      HeteroGraphConv,

# layers/temporalconv
      TGCN,
      A3TGCN,

# layers/pool
      GlobalPool,
      GlobalAttentionPool,
      Set2Set,
      TopKPool

include("layers/basic.jl")
include("layers/conv.jl")
include("layers/heteroconv.jl")
include("layers/temporalconv.jl")
include("layers/pool.jl")
include("deprecations.jl")

end
