# Models

GraphNeuralNetworks.jl provides common graph convolutional layers by which you can assemble arbitrarily deep or complex models. GNN layers are compatible with 
Flux.jl ones, therefore expert Flux users are promptly able to define and train 
their models. 

In what follows, we discuss two different styles for model creation:
the *explicit modeling* style, more verbose but more flexible, 
and the *implicit modeling* style based on [`GraphNeuralNetworks.GNNChain`](@ref), more concise but less flexible.

## Explicit modeling

In the explicit modeling style, the model is created according to the following steps:

1. Define a new type for your model (`GNN` in the example below). Layers and submodels are fields.
2. Apply `Flux.@layer` to the new type to make it Flux's compatible (parameters' collection, gpu movement, etc...)
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

Flux.@layer GNN                         # step 2

function GNN(din::Int, d::Int, dout::Int) # step 3    
    GNN(GraphConv(din => d),
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

g = rand_graph(10, 30)
X = randn(Float32, din, 10) 

y = model(g, X)  # output size: (dout, g.num_nodes)
grad = gradient(model -> sum(model(g, X)), model)
```

## Implicit modeling with GNNChains

While very flexible, the way in which we defined `GNN` model definition in last section is a bit verbose.
In order to simplify things, we provide the [`GraphNeuralNetworks.GNNChain`](@ref) type. It is very similar 
to Flux's well known `Chain`. It allows to compose layers in a sequential fashion as Chain
does, propagating the output of each layer to the next one. In addition, `GNNChain` propagates the input graph as well, providing it as a first argument
to layers subtyping the [`GraphNeuralNetworks.GNNLayer`](@ref) abstract type. 

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

A `GNNChain` opportunely propagates the graph into the branches created by the `Flux.Parallel` layer:

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

## Embedding a graph in the model

Sometimes it is useful to consider a specific graph as a part of a model instead of 
its input. GraphNeuralNetworks.jl provides the [`WithGraph`](@ref) type to deal with this scenario.

```julia
chain = GNNChain(GCNConv(din => d, relu),
                 GCNConv(d => d))


g = rand_graph(10, 30)

model = WithGraph(chain, g)

X = randn(Float32, din, 10)

# Pass only X as input, the model already contains the graph.
y = model(X) 
```

An example of `WithGraph` usage is given in the graph neural ODE script in the [examples](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/tree/master/examples) folder.
