__precompile__(false)

function GraphNeuralNetworks.GNNGraph(g::T; edge_weight = nothing, kws...) where 
                 {T <: Union{SimpleWeightedGraph, SimpleWeightedDiGraph, 
                       GraphNeuralNetworks.AbstractGraph}}
    s = Graphs.src.(Graphs.edges(g))
    t = Graphs.dst.(Graphs.edges(g))
    if g isa Union{SimpleWeightedGraph, SimpleWeightedDiGraph}
        w = filter(!iszero, g.weights |> vec) |> collect
    else
        w = edge_weight
    end
    if !Graphs.is_directed(g)
        # add reverse edges since GNNGraph is directed
        s, t = [s; t], [t; s]
        if !isnothing(w) & !(g isa Union{SimpleWeightedGraph, SimpleWeightedDiGraph})
            @assert length(w) == Graphs.ne(g) "edge_weight must have length equal to the number of undirected edges"
            w = [w; w]
        end
    end
    num_nodes = Graphs.nv(g)
   return GNNGraph((s, t, w); num_nodes = num_nodes, kws...)
end