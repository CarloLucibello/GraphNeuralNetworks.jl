# Models

## Explicit modeling

```julia
using Flux, GraphNeuralNetworks
using Flux: @functor

struct GNN
    conv1
    bn
    conv2
    dropout
    dense
end

@functor GNN

function GNN(din, d, dout)    
    GNN(GCNConv(din => d),
        BatchNorm(d),
        GraphConv(d => d, relu),
        Dropout(0.5),
        Dense(d, dout))
end

function (model::GNN)(g::GNNGraph, x)
    x = model.conv1(g, x)
    x = relu.(model.bn(x))
    x = model.conv2(g, x)
    x = model.dropout(x)
    x = model.dense(x)
    return x 
end

din, d, dout = 3, 4, 2 
g = GNNGraph(random_regular_graph(10, 4), graph_type=GRAPH_T)
X = randn(Float32, din, 10)
model = GNN(din, d, dout)
y = model(g, X)
```

## Compact modeling with GNNChains

```julia
model = GNNChain(GCNConv(din => d),
                 BatchNorm(d),
                 x -> relu.(x),
                 GraphConv(d => d, relu),
                 Dropout(0.5),
                 Dense(d, dout))
```
