# Heterogeneous Graphs

Heterogeneus graphs (also called heterographs), are graphs where each node has a type,
that we denote with symbols such as `:user` and `:movie`,
and edges also represent different relations identified
by a triple of symbols, `(source_nodes, edge_type, target_nodes)`, as in `(:user, :rate, :movie)`.

Different node/edge types can store different group of features
and this makes heterographs a very flexible modeling tools 
and data containers.

In GraphNeuralNetworks.jl heterographs are implemented in 
the type [`GNNHeteroGraph`](@ref).

