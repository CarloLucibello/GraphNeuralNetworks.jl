# GNNLux.jl 

GNNLux.jl is a work-in-progress package that implements stateless graph convolutional layers, fully compatible with the [Lux.jl](https://lux.csail.mit.edu/stable/) machine learning framework. It is built on top of the GNNGraphs.jl, GNNlib.jl, and Lux.jl packages.

## Package overview

Let's give a brief overview of the package by solving a graph regression problem with synthetic data. 

### Data preparation

We create a dataset consisting in multiple random graphs and associated data features. 

```julia
using GNNLux, Lux, Statistics, MLUtils, Random
using Zygote, Optimisers

all_graphs = GNNGraph[]

for _ in 1:1000
    g = rand_graph(10, 40,  
            ndata=(; x = randn(Float32, 16,10)),  # Input node features
            gdata=(; y = randn(Float32)))         # Regression target   
    push!(all_graphs, g)
end

train_graphs, test_graphs = MLUtils.splitobs(all_graphs, at=0.8)

# g = rand_graph(10, 40, ndata=(; x = randn(Float32, 16,10)), gdata=(; y = randn(Float32))) 

rng = Random.default_rng()

model = GNNChain(GCNConv(16 => 64),
                x -> relu.(x),    
                Dropout(0.6), 
                GCNConv(64 => 64, relu),
                x -> mean(x, dims=2),
                Dense(64, 1)) 

ps, st = LuxCore.setup(rng, model)

function custom_loss(model, ps,st,tuple)
    g,x,y = tuple
    y_pred,st = model(g, x, ps, st)  
    return MSELoss()(y_pred, y), (layers = st,), 0
end


function train_model!(model, ps, st, train_graphs, test_graphs)
    train_state = Lux.Training.TrainState(model, ps, st, Adam(0.0001f0))
    train_loss=0
    for iter in 1:100
        for g in train_graphs
            _, loss, _, train_state = Lux.Training.single_train_step!(AutoZygote(), custom_loss,(g, g.x, g.y), train_state)
            train_loss += loss
        end
        train_loss = train_loss/length(train_graphs)
        if iter % 10 == 0 || iter == 100
            st_ = Lux.testmode(train_state.states)
            test_loss =0
            for g in test_graphs
                ŷ, st_ = model(g, g.x, train_state.parameters, st_)
                st_ = (layers = st_,)
                test_loss += MSELoss()(g.y,ŷ)
            end
            test_loss = test_loss/length(test_graphs)
            @info (; iter, train_loss, test_loss)
        end
    end

    return model, ps, st
end

train_model!(model, ps, st, train_graphs, test_graphs)