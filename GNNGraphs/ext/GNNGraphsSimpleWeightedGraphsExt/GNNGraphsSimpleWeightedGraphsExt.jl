module GNNGraphsSimpleWeightedGraphsExt

using Graphs
using GNNGraphs
using SimpleWeightedGraphs

function GNNGraphs.GNNGraph(g::T; kws...) where 
                 {T <: Union{SimpleWeightedGraph, SimpleWeightedDiGraph}}
   return GNNGraph(g.weights, kws...)
end

end #module