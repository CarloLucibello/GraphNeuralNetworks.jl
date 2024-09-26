# Example of using TGCN, a recurrent temporal graph convolutional network of the paper https://arxiv.org/pdf/1811.05320.pdf, for traffic prediction by training it on the METRLA dataset 

# Load packages
using Flux
using Flux.Losses: mae
using GraphNeuralNetworks
using MLDatasets: METRLA
using CUDA
using Statistics, Random
CUDA.allowscalar(false)

# Import dataset function
function getdataset()
    metrla = METRLA(; num_timesteps = 3)
    g = metrla[1]
    graph = GNNGraph(g.edge_index; edata = g.edge_data, g.num_nodes)
    features = g.node_data.features
    targets = g.node_data.targets
    train_loader = zip(features[1:2000], targets[1:2000])
    test_loader = zip(features[2001:2288], targets[2001:2288])
    return graph, train_loader, test_loader
end

# Loss and accuracy functions
lossfunction(ŷ, y)  = Flux.mae(ŷ, y) 
accuracy(ŷ, y) = 1 - Statistics.norm(y-ŷ)/Statistics.norm(y)

function eval_loss_accuracy(model, graph, data_loader)
    error = mean([lossfunction(model(graph,x), y) for (x, y) in data_loader])
    acc = mean([accuracy(model(graph,x), y) for (x, y) in data_loader])
    return (loss = round(error, digits = 4), acc = round(acc , digits = 4))
end

# Arguments for the train function
Base.@kwdef mutable struct Args
    η = 1.0f-3             # learning rate
    epochs = 100           # number of epochs
    seed = 17              # set seed > 0 for reproducibility
    usecuda = true         # if true use cuda (if available)
    nhidden = 100          # dimension of hidden features
    infotime = 20          # report every `infotime` epochs
end

# Train function
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

    # Define model
    model = GNNChain(TGCN(2 => args.nhidden), Dense(args.nhidden, 1)) |> device

    opt = Flux.setup(Adam(args.η), model)

    graph, train_loader, test_loader = getdataset() 
    graph = graph |> device
    train_loader = train_loader |> device
    test_loader = test_loader |> device

    function report(epoch)
        train_loss, train_acc = eval_loss_accuracy(model, graph, train_loader)
        test_loss, test_acc = eval_loss_accuracy(model, graph, test_loader)
        println("Epoch: $epoch  $((; train_loss, train_acc))  $((; test_loss, test_acc))")
    end

    report(0)
    for epoch in 1:(args.epochs)
        for (x, y) in train_loader
            x, y = (x, y)
            grads = Flux.gradient(model) do model
                ŷ = model(graph, x)
                lossfunction(y,ŷ)
            end
            Flux.update!(opt, model, grads[1])
        end

        args.infotime > 0 && epoch % args.infotime == 0 && report(epoch)

    end
    return model
end

train()

