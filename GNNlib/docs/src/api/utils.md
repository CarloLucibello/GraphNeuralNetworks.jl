```@meta
CurrentModule = GNNlib
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
reduce_nodes
reduce_edges
softmax_nodes
softmax_edges
broadcast_nodes
broadcast_edges
```

### Neighborhood operations

```@docs
softmax_edge_neighbors
```

### NNlib

Primitive functions implemented in NNlib.jl:

- [`gather!`](https://fluxml.ai/NNlib.jl/stable/reference/#NNlib.gather!)
- [`gather`](https://fluxml.ai/NNlib.jl/stable/reference/#NNlib.gather)
- [`scatter!`](https://fluxml.ai/NNlib.jl/stable/reference/#NNlib.scatter!)
- [`scatter`](https://fluxml.ai/NNlib.jl/stable/reference/#NNlib.scatter)
