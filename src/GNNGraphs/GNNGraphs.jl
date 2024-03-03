module GNNGraphs

using SparseArrays
using Functors: @functor
import Graphs
using Graphs: AbstractGraph, outneighbors, inneighbors, adjacency_matrix, degree,
              has_self_loops, is_directed
import Flux
using Flux: batch
import NearestNeighbors
import NNlib
import StatsBase
import KrylovKit
using ChainRulesCore
using LinearAlgebra, Random, Statistics
import MLUtils
using MLUtils: getobs, numobs, ones_like, zeros_like
import Functors

include("chainrules.jl") # hacks for differentiability

include("datastore.jl")
export DataStore

include("abstracttypes.jl")
export AbstractGNNGraph

include("gnngraph.jl")
export GNNGraph,
       node_features,
       edge_features,
       graph_features

include("gnnheterograph.jl")
export GNNHeteroGraph,
       num_edge_types,
       num_node_types,
       edge_type_subgraph

include("temporalsnapshotsgnngraph.jl")
export TemporalSnapshotsGNNGraph,
       add_snapshot,
       # add_snapshot!,
       remove_snapshot
       # remove_snapshot!

include("query.jl")
export adjacency_list,
       edge_index,
       get_edge_weight,
       graph_indicator,
       has_multi_edges,
       is_directed,
       is_bidirected,
       normalized_laplacian,
       scaled_laplacian,
       laplacian_lambda_max,
# from Graphs
       adjacency_matrix,
       degree,
       has_self_loops,
       has_isolated_nodes,
       inneighbors,
       outneighbors,
       khop_adj

include("transform.jl")
export add_nodes,
       add_edges,
       add_self_loops,
       getgraph,
       negative_sample,
       rand_edge_split,
       remove_self_loops,
       remove_multi_edges,
       set_edge_weight,
       to_bidirected,
       to_unidirected,
       random_walk_pe,
# from Flux
       batch,
       unbatch,
# from SparseArrays
       blockdiag

include("generate.jl")
export rand_graph,
       rand_heterograph,
       rand_bipartite_heterograph,
       knn_graph,
       radius_graph,
       rand_temporal_radius_graph,
       rand_temporal_hyperbolic_graph

include("sampling.jl")
export sample_neighbors

include("operators.jl")
# Base.intersect

include("convert.jl")
include("utils.jl")

include("gatherscatter.jl")
# _gather, _scatter

end #module
