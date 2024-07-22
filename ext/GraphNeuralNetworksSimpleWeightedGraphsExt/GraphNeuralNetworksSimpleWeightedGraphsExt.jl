module GraphNeuralNetworksSimpleWeightedGraphsExt

using GraphNeuralNetworks
using Graphs
using SimpleWeightedGraphs

function GraphNeuralNetworks.GNNGraph(g::T; kws...) where 
                 {T <: Union{SimpleWeightedGraph, SimpleWeightedDiGraph}}
   return GNNGraph(g.weights, kws...)
end

end #module