```@meta
CurrentModule = GNNlib
CollapsedDocStrings = true
```

# Utility Functions

## Graph-wise operations 

```@docs
reduce_nodes
reduce_edges
softmax_nodes
softmax_edges
broadcast_nodes
broadcast_edges
```

## Neighborhood operations

```@docs
softmax_edge_neighbors
```

## NNlib's gather and scatter functions

Primitive functions for message passing implemented in [NNlib.jl](https://fluxml.ai/NNlib.jl/stable/reference/#Gather-and-Scatter):

- [`gather!`](https://fluxml.ai/NNlib.jl/stable/reference/#NNlib.gather!)
- [`gather`](https://fluxml.ai/NNlib.jl/stable/reference/#NNlib.gather)
- [`scatter!`](https://fluxml.ai/NNlib.jl/stable/reference/#NNlib.scatter!)
- [`scatter`](https://fluxml.ai/NNlib.jl/stable/reference/#NNlib.scatter)
