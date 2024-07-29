```@meta
CurrentModule = GraphNeuralNetworks
```

# Message Passing

## Index

```@index
Order = [:type, :function]
Pages   = ["messagepassing.md"]
```

## Interface

```@docs
GNNlib.apply_edges
GNNlib.aggregate_neighbors
GNNlib.propagate
```

## Built-in message functions

```@docs
GNNlib.copy_xi
GNNlib.copy_xj
GNNlib.xi_dot_xj
GNNlib.xi_sub_xj
GNNlib.xj_sub_xi
GNNlib.e_mul_xj
GNNlib.w_mul_xj
```
