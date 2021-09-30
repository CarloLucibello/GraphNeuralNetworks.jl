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

### NNlib

Primitive functions implemented in NNlib.jl.

```@docs
NNlib.gather!
NNlib.gather
NNlib.scatter!
NNlib.scatter
```