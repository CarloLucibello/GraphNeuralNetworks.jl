# An example of graph classification

using Flux
using Flux: onecold, onehotbatch
using Flux.Losses: logitbinarycrossentropy
using Flux: DataLoader
using GraphNeuralNetworks
using MLDatasets: TUDataset
using Statistics, Random
using MLUtils
using CUDA
CUDA.allowscalar(false)

function eval_loss_accuracy(model, data_loader, device)
    loss = 0.0
    acc = 0.0
    ntot = 0
    for (g, y) in data_loader
        g, y = (g, y) |> device
        n = length(y)
        ŷ = model(g, g.ndata.x) |> vec
        loss += logitbinarycrossentropy(ŷ, y) * n
        acc += mean((ŷ .> 0) .== y) * n
        ntot += n
    end
    return (loss = round(loss / ntot, digits = 4),
            acc = round(acc * 100 / ntot, digits = 2))
end

function getdataset()
    tudata = TUDataset("MUTAG")
    display(tudata)
    graphs = mldataset2gnngraph(tudata)
    oh(x) = Float32.(onehotbatch(x, 0:6))
    graphs = [GNNGraph(g, ndata = oh(g.ndata.targets)) for g in graphs]
    y = (1 .+ Float32.(tudata.graph_data.targets)) ./ 2
    @assert all(∈([0, 1]), y) # binary classification 
    return graphs, y
end

# arguments for the `train` function 
Base.@kwdef mutable struct Args
    η = 1.0f-3             # learning rate
    batchsize = 32      # batch size (number of graphs in each batch)
    epochs = 200         # number of epochs
    seed = 17             # set seed > 0 for reproducibility
    usecuda = true      # if true use cuda (if available)
    nhidden = 128        # dimension of hidden features
    infotime = 10      # report every `infotime` epochs
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

    # LOAD DATA
    NUM_TRAIN = 150

    dataset = getdataset()
    train_data, test_data = splitobs(dataset, at = NUM_TRAIN, shuffle = true)

    train_loader = DataLoader(train_data; args.batchsize, shuffle = true, collate = true)
    test_loader = DataLoader(test_data; args.batchsize, shuffle = false, collate = true)

    # DEFINE MODEL

    nin = size(dataset[1][1].ndata.x, 1)
    nhidden = args.nhidden

    model = GNNChain(GraphConv(nin => nhidden, relu),
                     GraphConv(nhidden => nhidden, relu),
                     GlobalPool(mean),
                     Dense(nhidden, 1)) |> device

    opt = Flux.setup(Adam(args.η), model)

    # LOGGING FUNCTION

    function report(epoch)
        train = eval_loss_accuracy(model, train_loader, device)
        test = eval_loss_accuracy(model, test_loader, device)
        println("Epoch: $epoch   Train: $(train)   Test: $(test)")
    end

    # TRAIN

    report(0)
    for epoch in 1:(args.epochs)
        for (g, y) in train_loader
            g, y = (g, y) |> device
            grads = Flux.gradient(model) do model
                ŷ = model(g, g.ndata.x) |> vec
                logitbinarycrossentropy(ŷ, y)
            end
            Flux.update!(opt, model, grads[1])
        end
        epoch % args.infotime == 0 && report(epoch)
    end
end

train()
