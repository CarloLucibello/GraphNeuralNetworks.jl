# An example of semi-supervised node classification

using Flux
using Flux: onecold, onehotbatch
using Flux.Losses: logitcrossentropy
using GeometricFlux, GraphSignals
using MLDatasets: Cora
using Statistics, Random
using CUDA
CUDA.allowscalar(false)

function eval_loss_accuracy(X, y, ids, model)
    ŷ = model(X)
    l = logitcrossentropy(ŷ[:, ids], y[:, ids])
    acc = mean(onecold(ŷ[:, ids]) .== onecold(y[:, ids]))
    return (loss = round(l, digits = 4), acc = round(acc * 100, digits = 2))
end

# arguments for the `train` function 
Base.@kwdef mutable struct Args
    η = 1.0f-3             # learning rate
    epochs = 100          # number of epochs
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
    data = Cora.dataset()
    g = FeaturedGraph(data.adjacency_list) |> device
    X = data.node_features |> device
    y = onehotbatch(data.node_labels, 1:(data.num_classes)) |> device
    train_ids = data.train_indices |> device
    val_ids = data.val_indices |> device
    test_ids = data.test_indices |> device
    ytrain = y[:, train_ids]

    nin, nhidden, nout = size(X, 1), args.nhidden, data.num_classes

    ## DEFINE MODEL
    model = Chain(GCNConv(g, nin => nhidden, relu),
                  Dropout(0.5),
                  GCNConv(g, nhidden => nhidden, relu),
                  Dense(nhidden, nout)) |> device

    opt = Flux.setup(Adam(args.η), model)

    @info g

    ## LOGGING FUNCTION
    function report(epoch)
        train = eval_loss_accuracy(X, y, train_ids, model)
        test = eval_loss_accuracy(X, y, test_ids, model)
        println("Epoch: $epoch   Train: $(train)   Test: $(test)")
    end

    ## TRAINING
    report(0)
    for epoch in 1:(args.epochs)
        grad = Flux.gradient(model) do model
            ŷ = model(X)
            logitcrossentropy(ŷ[:, train_ids], ytrain)
        end

        Flux.update!(opt, model, grad[1])

        epoch % args.infotime == 0 && report(epoch)
    end
end

train(usecuda = false)
