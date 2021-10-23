# Models

GraphNeuralNetworks.jl provides common graph convolutional layers by which you can assemble arbitrarily deep or complex models. GNN layers are compatible with 
Flux.jl ones, therefore expert Flux's users should be immediately able to define and train 
their models. 

In what follows, we discuss two different styles for model creation:
the *explicit modeling* style, more verbose but more flexible, 
and the *implicit modeling* style based on [`GNNChain`](@ref), more concise but less flexible.

## Explicit modeling

In the explicit modeling style, the model is created according to the following steps:

1. Define a new type for your model (`GNN` in the example below). Layers and submodels are fields.
2. Apply `Flux.@functor` to the new type to make it Flux's compatible (parameters' collection, gpu movement, etc...)
3. Optionally define a convenience constructor for your model.
4. Define the forward pass by implementing the call method for your type.
5. Instantiate the model. 

Here is an example of this construction:
```julia
using Flux, Graphs, GraphNeuralNetworks

struct GNN                                # step 1
    conv1
    bn
    conv2
    dropout
    dense
end

Flux.@functor GNN                              # step 2

function GNN(din::Int, d::Int, dout::Int) # step 3    
    GNN(GCNConv(din => d),
        BatchNorm(d),
        GraphConv(d => d, relu),
        Dropout(0.5),
        Dense(d, dout))
end

function (model::GNN)(g::GNNGraph, x)     # step 4
    x = model.conv1(g, x)
    x = relu.(model.bn(x))
    x = model.conv2(g, x)
    x = model.dropout(x)
    x = model.dense(x)
    return x 
end

din, d, dout = 3, 4, 2 
model = GNN(din, d, dout)                 # step 5

g = GNNGraph(random_regular_graph(10, 4))
X = randn(Float32, din, 10) 

y = model(g, X)  # output size: (dout, g.num_nodes)
gs = gradient(() -> sum(model(g, X)), Flux.params(model))
```

## Implicit modeling with GNNChains

While very flexible, the way in which we defined `GNN` model definition in last section is a bit verbose.
In order to simplify things, we provide the [`GNNChain`](@ref) type. It is very similar 
to Flux's well known `Chain`. It allows to compose layers in a sequential fashion as Chain
does, propagating the output of each layer to the next one. In addition, `GNNChain` 
handles propagates the input graph as well, providing it as a first argument
to layers subtyping the [`GNNLayer`](@ref) abstract type. 

Using `GNNChain`, the previous example becomes

```julia
using Flux, Graphs, GraphNeuralNetworks

din, d, dout = 3, 4, 2 
g = GNNGraph(random_regular_graph(10, 4))
X = randn(Float32, din, 10)

model = GNNChain(GCNConv(din => d),
                 BatchNorm(d),
                 x -> relu.(x),
                 GCNConv(d => d, relu),
                 Dropout(0.5),
                 Dense(d, dout))
```

The `GNNChain` only propagates the graph and the node features. More complex scenarios, e.g. when also edge features are updated, have to be handled using the explicit definition of the forward pass. 

A `GNNChain` oppurtunely propagates the graph into the branches created by the `Flux.Parallel` layer:

```julia
AddResidual(l) = Parallel(+, identity, l)  # implementing a skip/residual connection

model = GNNChain( ResGatedGraphConv(din => d, relu),
                  AddResidual(ResGatedGraphConv(d => d, relu)),
                  AddResidual(ResGatedGraphConv(d => d, relu)),
                  AddResidual(ResGatedGraphConv(d => d, relu)),
                  GlobalPooling(mean),
                  Dense(d, dout))

y = model(g, X) # output size: (dout, g.num_graphs)
```
