module GNNGraphs

using SparseArrays
using Functors: @functor
using CUDA 
import Graphs
using Graphs: AbstractGraph, outneighbors, inneighbors, adjacency_matrix, degree, has_self_loops, is_directed
import Flux
using Flux: batch
import NearestNeighbors
import NNlib
import LearnBase
import StatsBase
import KrylovKit
using ChainRulesCore
using LinearAlgebra, Random

include("gnngraph.jl")
export GNNGraph, 
       node_features, 
       edge_features, 
       graph_features
    
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
       # from Graphs
       adjacency_matrix, 
       degree, 
       has_self_loops,
       inneighbors,
       outneighbors 

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
       # from Flux
       batch,
       unbatch,
       # from SparseArrays
       blockdiag

include("generate.jl")
export rand_graph, 
       knn_graph

include("sampling.jl")
export sample_neighbors

include("operators.jl")
# Base.intersect

include("convert.jl")
include("utils.jl")

    
end #module
