using Flux
using Flux: onecold, onehotbatch
using Flux.Losses: logitcrossentropy
using GraphNeuralNetworks
using MLDatasets: Cora
using Statistics, Random
using CUDA
CUDA.allowscalar(false)

function eval_loss_accuracy(X, y, ids, model, g)
    ŷ = model(g, X)
    l = logitcrossentropy(ŷ[:,ids], y[:,ids])
    acc = mean(onecold(ŷ[:,ids] |> cpu) .== onecold(y[:,ids] |> cpu))
    return (loss = round(l, digits=4), acc = round(acc*100, digits=2))
end


# arguments for the `train` function 
Base.@kwdef mutable struct Args
    η = 5f-3             # learning rate
    epochs = 10         # number of epochs
    seed = 17             # set seed > 0 for reproducibility
    usecuda = false      # if true use cuda (if available)
    nhidden = 64        # dimension of hidden features
end

function train(Layer; verbose=false, kws...)
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)
    
    if args.usecuda && CUDA.functional()
        device = Flux.gpu
        args.seed > 0 && CUDA.seed!(args.seed)
    else
        device = Flux.cpu
    end

    # LOAD DATA
    data = Cora.dataset()
    g = GNNGraph(data.adjacency_list) |> device
    X = data.node_features |> device
    y = onehotbatch(data.node_labels, 1:data.num_classes) |> device
    train_ids = data.train_indices |> device
    val_ids = data.val_indices |> device
    test_ids = data.test_indices |> device
    ytrain = y[:,train_ids]

    nin, nhidden, nout = size(X,1), args.nhidden, data.num_classes 
    
    ## DEFINE MODEL
    model = GNNChain(Layer(nin, nhidden),
                    #  Dropout(0.5),
                     Layer(nhidden, nhidden), 
                     Dense(nhidden, nout))  |> device

    ps = Flux.params(model)
    opt = ADAM(args.η)
    

    ## TRAINING
    function report(epoch)
        train = eval_loss_accuracy(X, y, train_ids, model, g)
        test = eval_loss_accuracy(X, y, test_ids, model, g)        
        println("Epoch: $epoch   Train: $(train)   Test: $(test)")
    end
    
    verbose && report(0)
    for epoch in 1:args.epochs
        gs = Flux.gradient(ps) do
            ŷ = model(g, X)
            logitcrossentropy(ŷ[:,train_ids], ytrain)
        end
        Flux.Optimise.update!(opt, ps, gs)
        verbose && report(epoch)
    end

    train_res = eval_loss_accuracy(X, y, train_ids, model, g)
    test_res = eval_loss_accuracy(X, y, test_ids, model, g)        
    return train_res, test_res
end

for (layer, Layer) in [
            ("GCNConv", (nin, nout) -> GCNConv(nin => nout, relu)),
            ("GraphConv", (nin, nout) -> GraphConv(nin => nout, relu, aggr=mean)),
            ("SAGEConv", (nin, nout) -> SAGEConv(nin => nout, relu)),
            ("GATConv", (nin, nout) -> GATConv(nin => nout, relu)),
            ("GINConv", (nin, nout) -> GINConv(Dense(nin, nout, relu), 0.01, aggr=mean)),
            ("ChebConv", (nin, nout) -> ChebConv(nin => nout, 2)),
            ("ResGatedGraphConv", (nin, nout) -> ResGatedGraphConv(nin => nout, relu)),        
            # (nin, nout) -> NNConv(nin => nout),  # needs edge features
            # (nin, nout) -> GatedGraphConv(nout, 2),  # needs nin = nout
            # (nin, nout) -> EdgeConv(Dense(2nin, nout, relu)), # Fits the traning set but does not generalize well
              ]

    @show layer
    @time train_res, test_res = train(Layer, verbose=true)
    @test train_res.acc > 95
    @test test_res.acc > 70
end
