module GNNlibSimpleWeightedGraphsExt

using GNNlib
using Graphs
using SimpleWeightedGraphs

function GNNlib.GNNGraph(g::T; kws...) where 
                 {T <: Union{SimpleWeightedGraph, SimpleWeightedDiGraph}}
   return GNNGraph(g.weights, kws...)
end

end #module