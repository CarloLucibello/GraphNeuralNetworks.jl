# GNNLux.jl 

GNNLux.jl is a work-in-progress package that implements stateless graph convolutional layers, fully compatible with the [Lux.jl](https://lux.csail.mit.edu/stable/) machine learning framework. It is built on top of the GNNGraphs.jl, GNNlib.jl, and Lux.jl packages.

## Package overview

Let's give a brief overview of the package by solving a graph regression problem with synthetic data. 

### Data preparation

We create a dataset consisting in multiple random graphs and associated data features. 

```julia
using GNNLux, Lux, Statistics, MLUtils, Random
using Zygote, Optimizers

all_graphs = GNNGraph[]

for _ in 1:1000
    g = rand_graph(10, 40,  
            ndata=(; x = randn(Float32, 16,10)),  # Input node features
            gdata=(; y = randn(Float32)))         # Regression target   
    push!(all_graphs, g)
end
```

### Model building 

We concisely define our model as a [`GNNLux.GNNChain`](@ref) containing two graph convolutional layers. If CUDA is available, our model will live on the gpu.

```julia
device = CUDA.functional() ? Lux.gpu_device() : Lux.cpu_device()
rng = Random.default_rng()

model = GNNChain(GCNConv(16 => 64),
                x -> relu.(x),     
                GCNConv(64 => 64, relu),
                GlobalMeanPool(),  # Aggregate node-wise features into graph-wise features
                Dense(64, 1)) 

ps, st = LuxCore.setup(rng, model)
```

### Training 


```julia
train_graphs, test_graphs = MLUtils.splitobs(all_graphs, at=0.8)

train_loader = MLUtils.DataLoader(train_graphs, 
                batchsize=32, shuffle=true, collate=true)
test_loader = MLUtils.DataLoader(test_graphs, 
                batchsize=32, shuffle=false, collate=true)

for epoch in 1:100
    for g in train_loader
        g = g |> device
        grad = gradient(model -> loss(model, g), model)
        Flux.update!(opt, model, grad[1])
    end

    @info (; epoch, train_loss=loss(model, train_loader), test_loss=loss(model, test_loader))
end

function train_model!(model, ps, st, train_loader)
    train_state = Lux.Training.TrainState(model, ps, st, Adam(0.001f0))

    for iter in 1:1000
        for g in train_loader
            _, loss, _, train_state = Lux.Training.single_train_step!(AutoZygote(), MSELoss(),
                ((g, g.x)...,g.y), train_state)
            if iter % 100 == 1 || iter == 1000
                @info "Iteration: %04d \t Loss: %10.9g\n" iter loss
            end
        end
    end

    return model, ps, st
end

train_model!(model, ps, st, train_loader)
```
