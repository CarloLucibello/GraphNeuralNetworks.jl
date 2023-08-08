# Temporal graph neural networks for traffic prediction

# Load packages
using Flux
using Flux.Losses: mae, mse
using GraphNeuralNetworks
using MLDatasets: METRLA
using CUDA
using Statistics, Random
CUDA.allowscalar(false)

function getdataset()
    metrla = METRLA()
    g=metrla[1]
    features=[]
    targets=[]
    graph = GNNGraph(g.edge_index; edata = g.edge_data, g.num_nodes)
    for i in 1:1000
        push!(features, g.node_data.features[i])
        push!(targets,g.node_data.targets[i])
    end
    train_loader = zip(features, targets) 
    return train_loader, graph
end

Base.@kwdef mutable struct Args
    η = 1.0f-3             # learning rate
    batchsize = 32         # batch size (number of graphs in each batch)
    epochs = 10            # number of epochs
    seed = 17              # set seed > 0 for reproducibility
    usecuda = true         # if true use cuda (if available)
    nhidden = 128          # dimension of hidden features
    infotime = 10          # report every `infotime` epochs
end


lossfunction(y,ŷ) = Flux.mse(ŷ, y) 


struct mmodel 
    tgcn
    dense::Dense
end

Flux.@functor mmodel

function mmodel(ch::Pair{Int, Int};
                bias::Bool = true,
                add_self_loops = false,
                use_edge_weight = true)
    in, out = ch
    tgcn = TGCN(in => out; bias,init_state = CUDA.zeros, add_self_loops, use_edge_weight)
    dense = Dense(out,1)
    return mmodel(tgcn,dense)
end

function (m::mmodel)(g::GNNGraph, x)
    x = m.tgcn(g, x)
    x = m.dense(x)
    return x
end

function train(; kws...)
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    if args.usecuda && CUDA.functional()
        device = gpu
        args.seed > 0 && CUDA.seed!(args.seed)
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    model = mmodel(2 => 10) |> device

    opt = Flux.setup(Adam(args.η), model)

    train_loader, graph=getdataset() 
    graph = graph |> device
    train_loader = train_loader |> device
    for epoch in 1:(args.epochs)
        for (x, y) in train_loader
            x, y = (x, y)
            grads = Flux.gradient(model) do model
                ŷ = model(graph, x)
                lossfunction(y,ŷ)
            end
            Flux.update!(opt, model, grads[1])
        end
        error = mean([lossfunction(model(graph,x), y) for (x, y) in train_loader])
        println("$epoch :  $error")
    end
    return model
end

model = train()

