# Models

GNNLux.jl provides common graph convolutional layers by which you can assemble arbitrarily deep or complex models. GNN layers are compatible with 
Lux.jl ones, therefore expert Lux users are promptly able to define and train 
their models. 

In what follows, we discuss two different styles for model creation:
the *explicit modeling* style, more verbose but more flexible, 
and the *implicit modeling* style based on [`GNNLux.GNNChain`](@ref), more concise but less flexible.

## Explicit modeling

In the explicit modeling style, the model is created according to the following steps:

1. Define a new type for your model (`GNN` in the example below). Refer to the
 [Lux Manual](https://lux.csail.mit.edu/dev/manual/interface#lux-interface) for the
 definition of the type.
2. Define a convenience constructor for your model.
4. Define the forward pass by implementing the call method for your type.
5. Instantiate the model. 

Here is an example of this construction:
```julia
using Lux, GNNLux
using Zygote
using Random, Statistics

struct GNN <: AbstractLuxContainerLayer{(:conv1, :bn, :conv2, :dropout, :dense)} # step 1
    conv1
    bn
    conv2
    dropout
    dense
end

function GNN(din::Int, d::Int, dout::Int) # step 2
    GNN(GraphConv(din => d),
        BatchNorm(d),
        GraphConv(d => d, relu),
        Dropout(0.5),
        Dense(d, dout))
end

function (model::GNN)(g::GNNGraph, x, ps, st) # step 3
    x, st_conv1 = model.conv1(g, x, ps.conv1, st.conv1)
    x, st_bn = model.bn(x, ps.bn, st.bn)
    x = relu.(x)
    x, st_conv2 = model.conv2(g, x, ps.conv2, st.conv2)
    x, st_drop = model.dropout(x, ps.dropout, st.dropout)
    x, st_dense = model.dense(x, ps.dense, st.dense)
    return x, (conv1=st_conv1, bn=st_bn, conv2=st_conv2, dropout=st_drop, dense=st_dense)
end

din, d, dout = 3, 4, 2 
model = GNN(din, d, dout)                 # step 4
rng = Random.default_rng()
ps, st = Lux.setup(rng, model)
g = rand_graph(rng, 10, 30)
X = randn(Float32, din, 10) 

st = Lux.testmode(st)
y, st = model(g, X, ps, st) 
st = Lux.trainmode(st)
grad = Zygote.gradient(ps -> mean(model(g, X, ps, st)[1]), ps)[1]
```

## Implicit modeling with GNNChains

While very flexible, the way in which we defined `GNN` model definition in last section is a bit verbose.
In order to simplify things, we provide the [`GNNLux.GNNChain`](@ref) type. It is very similar 
to Lux's well known `Chain`. It allows to compose layers in a sequential fashion as Chain
does, propagating the output of each layer to the next one. In addition, `GNNChain` 
 propagates the input graph as well, providing it as a first argument
to layers subtyping the [`GNNLux.GNNLayer`](@ref) abstract type. 

Using `GNNChain`, the model definition becomes more concise:

```julia
model = GNNChain(GraphConv(din => d),
                 BatchNorm(d),
                 x -> relu.(x),
                 GraphConv(d => d, relu),
                 Dropout(0.5),
                 Dense(d, dout))
```

The `GNNChain` only propagates the graph and the node features. More complex scenarios, e.g. when also edge features are updated, have to be handled using the explicit definition of the forward pass. 
