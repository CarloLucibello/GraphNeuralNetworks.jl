# GNNGraphs.jl

GNNGraphs.jl is a package that provides graph data structures and helper functions specifically designed for working with graph neural networks. This package allows to store not only the graph structure, but also features associated with nodes, edges, and the graph itself. It is the core foundation for the GNNlib.jl, GraphNeuralNetworks.jl, and GNNLux.jl packages.

It supports three types of graphs: 

- **Static graph** is the basic graph type represented by [`GNNGraph`](@ref), where each node and edge can have associated features. This type of graph is used in typical graph neural network applications, where neural networks operate on both the structure of the graph and the features stored in it. It can be used to represent a graph where the structure does not change over time, but the features of the nodes and edges can change over time.

- **Heterogeneous graph** is a graph that supports multiple types of nodes and edges, and is represented by [`GNNHeteroGraph`](@ref). Each type can have its own properties and relationships. This is useful in scenarios with different entities and interactions, such as in citation graphs or multi-relational data.

- **Temporal graph** is a graph that changes over time, and is represented by [`TemporalSnapshotsGNNGraph`](@ref). Edges and features can change dynamically. This type of graph is useful for applications that involve tracking time-dependent relationships, such as social networks.



This package depends on the package [Graphs.jl] (https://github.com/JuliaGraphs/Graphs.jl).



## Installation

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:
    
```julia
pkg> add GNNGraphs
```

