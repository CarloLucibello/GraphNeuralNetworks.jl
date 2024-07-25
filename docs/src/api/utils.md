```@meta
CurrentModule = GraphNeuralNetworks
```

# Utility Functions

## Index

```@index
Order = [:type, :function]
Pages   = ["utils.md"]
```

## Docs


### Graph-wise operations 

```@docs
GraphNeuralNetworks.reduce_nodes
GraphNeuralNetworks.reduce_edges
GraphNeuralNetworks.softmax_nodes
GraphNeuralNetworks.softmax_edges
GraphNeuralNetworks.broadcast_nodes
GraphNeuralNetworks.broadcast_edges
```

### Neighborhood operations

```@docs
GraphNeuralNetworks.softmax_edge_neighbors
```

### NNlib

Primitive functions implemented in NNlib.jl:

- [`gather!`](https://fluxml.ai/NNlib.jl/stable/reference/#NNlib.gather!)
- [`gather`](https://fluxml.ai/NNlib.jl/stable/reference/#NNlib.gather)
- [`scatter!`](https://fluxml.ai/NNlib.jl/stable/reference/#NNlib.scatter!)
- [`scatter`](https://fluxml.ai/NNlib.jl/stable/reference/#NNlib.scatter)
