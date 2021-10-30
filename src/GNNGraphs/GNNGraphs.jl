module GNNGraphs

using SparseArrays
using Functors: @functor
using CUDA 
import Graphs
using Graphs: AbstractGraph, outneighbors, inneighbors, adjacency_matrix, degree
import Flux
using Flux: batch
import NNlib
import LearnBase
import StatsBase
using LearnBase: getobs
import KrylovKit
using ChainRulesCore
using LinearAlgebra, Random

include("gnngraph.jl")
export GNNGraph, node_features, edge_features, graph_features
    
include("query.jl")
export  edge_index, adjacency_list, normalized_laplacian, scaled_laplacian,
        graph_indicator
    
include("transform.jl")
export add_edges, add_self_loops, remove_self_loops, getgraph

include("generate.jl")
export rand_graph


include("convert.jl")
include("utils.jl")

export 
    # from Graphs
    adjacency_matrix, degree, outneighbors, inneighbors,
    # from SparseArrays
    sprand, sparse, blockdiag,
    # from Flux
    batch

end #module
