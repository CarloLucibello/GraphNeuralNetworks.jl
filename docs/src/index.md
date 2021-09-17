# GraphNeuralNetworks

This is the documentation page for the [GraphNeuralNetworks.jl](https://github.com/CarloLucibello/GraphNeuralNetworks.jl) library.

A graph neural network library for Julia based on the deep learning framework [Flux.jl](https://github.com/FluxML/Flux.jl).
Its most relevant features are:
* Provides CUDA support.
* It's integrated with the JuliaGraphs ecosystem.
* Implements many common graph convolutional layers.
* Performs fast operations on batched graphs. 
* Makes it easy to define custom graph convolutional layers.


## Package overview

Let's give a brief overview of the package solving a  
graph regression problem on fake data. 

Usage examples on real datasets can be found in the [examples](https://github.com/CarloLucibello/GraphNeuralNetworks.jl/tree/master/examples) folder. 

### Data preparation

First, we create our dataset consisting in multiple random graphs and associated data features. 
that we batch together into a unique graph.

```juliarepl
julia> using GraphNeuralNetworks, LightGraphs, Flux, CUDA, Statistics

julia> all_graphs = GNNGraph[];

julia> for _ in 1:1000
           g = GNNGraph(random_regular_graph(10, 4),  
                       ndata=(; x = randn(Float32, 16,10)),  # input node features
                       gdata=(; y = randn(Float32)))         # regression target   
           push!(all_graphs, g)
       end

julia> gbatch = Flux.batch(all_graphs)
GNNGraph:
    num_nodes = 10000
    num_edges = 20000
    num_graphs = 1000
    ndata:
        x => (16, 10000)
    edata:
    gdata:
        y => (1000,)
```


### Model building 

We concisely define our model using as a [`GNNChain`](@ref) containing 2 graph convolutaional 
layers. If CUDA is available, our model will leave on the gpu.

```juliarepl
julia> device = CUDA.functional() ? Flux.gpu : Flux.cpu;

julia> model = GNNChain(GCNConv(16 => 64),
                        BatchNorm(64),
                        x -> relu.(x),
                        GCNConv(64 => 64, relu),
                        GlobalPool(mean),
                        Dense(64, 1)) |> device;

julia> ps = Flux.params(model);

julia> opt = ADAM(1f-4);
```

### Training 

```juliarepl
gtrain, _ = getgraph(gbatch, 1:800)
gtest, _ = getgraph(gbatch, 801:gbatch.num_graphs)
train_loader = Flux.Data.DataLoader(gtrain, batchsize=32, shuffle=true)
test_loader = Flux.Data.DataLoader(gtest, batchsize=32, shuffle=false)

function loss(g::GNNGraph)
    mean((vec(model(g, g.ndata.x)) - g.gdata.y).^2)
end

loss(loader) = mean(loss(g |> device) for g in loader)

for epoch in 1:100
    for g in train_loader
        g = g |> gpu
        grad = gradient(() -> loss(g), ps)
        Flux.Optimise.update!(opt, ps, grad)
    end

    @info (; epoch, train_loss=loss(train_loader), test_loss=loss(test_loader))
end
```
