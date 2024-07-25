module GNNlib

using Statistics: mean
using LinearAlgebra, Random
using MLUtils: zeros_like
using NNlib
using NNlib: scatter, gather
using DataStructures: nlargest
using ChainRulesCore: @non_differentiable
using GNNGraphs
using .GNNGraphs: COO_T, ADJMAT_T, SPARSE_T,
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
      w_mul_xj

## The following methods are defined but not exported

# # layers/basic
#       dot_decoder,

# # layers/conv
#       agnn_conv,
#       cg_conv,
#       cheb_conv,
#       edge_conv,
#       egnn_conv,
#       gat_conv,
#       gatv2_conv,
#       gated_graph_conv,
#       gcn_conv,
#       gin_conv,
#       gmm_conv,
#       graph_conv,
#       megnet_conv,
#       nn_conv,
#       res_gated_graph_conv,
#       sage_conv,
#       sg_conv,
#       transformer_conv,

# # layers/temporalconv
#       a3tgcn_conv,

# # layers/pool
#       global_pool,
#       global_attention_pool,
#       set2set_pool,
#       topk_pool,
#       topk_index,


include("utils.jl")
include("layers/basic.jl")
include("layers/conv.jl")
# include("layers/heteroconv.jl") # no functional part at the moment
include("layers/temporalconv.jl")
include("layers/pool.jl")
include("msgpass.jl")

end #module
 