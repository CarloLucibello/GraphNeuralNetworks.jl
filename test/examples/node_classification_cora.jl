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
    acc = mean(onecold(ŷ[:,ids]) .== onecold(y[:,ids]))
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
    dataset = Cora()
    classes = dataset.metadata["classes"]
    g = mldataset2gnngraph(dataset) |> device
    X = g.ndata.features
    y = onehotbatch(g.ndata.targets |> cpu, classes) |> device # remove when https://github.com/FluxML/Flux.jl/pull/1959 tagged
    train_mask = g.ndata.train_mask
    test_mask = g.ndata.test_mask
    ytrain = y[:,train_mask]

    nin, nhidden, nout = size(X,1), args.nhidden, length(classes)
    
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
    @time for epoch in 1:args.epochs
        gs = Flux.gradient(ps) do
            ŷ = model(g, X)
            logitcrossentropy(ŷ[:,train_mask], ytrain)
        end
        Flux.Optimise.update!(opt, ps, gs)
        verbose && report(epoch)
    end

    train_res = eval_loss_accuracy(X, y, train_mask, model, g)
    test_res = eval_loss_accuracy(X, y, test_mask, model, g)        
    return train_res, test_res
end

function train_many(; usecuda=false)
    for (layer, Layer) in [
                ("GCNConv", (nin, nout) -> GCNConv(nin => nout, relu)),
                ("ResGatedGraphConv", (nin, nout) -> ResGatedGraphConv(nin => nout, relu)),        
                ("GraphConv", (nin, nout) -> GraphConv(nin => nout, relu, aggr=mean)),
                ("SAGEConv", (nin, nout) -> SAGEConv(nin => nout, relu)),
                ("GATConv", (nin, nout) -> GATConv(nin => nout, relu)),
                ("GINConv", (nin, nout) -> GINConv(Dense(nin, nout, relu), 0.01, aggr=mean)),
                ## ("ChebConv", (nin, nout) -> ChebConv(nin => nout, 2)), # not working on gpu
                ## ("NNConv", (nin, nout) -> NNConv(nin => nout)),  # needs edge features
                ## ("GatedGraphConv", (nin, nout) -> GatedGraphConv(nout, 2)),  # needs nin = nout
                ## ("EdgeConv",(nin, nout) -> EdgeConv(Dense(2nin, nout, relu))), # Fits the traning set but does not generalize well
                ]

        @show layer
        @time train_res, test_res = train(Layer; usecuda, verbose=false)
        @test train_res.acc > 94
        @test test_res.acc > 70
    end
end

train_many(usecuda=false)
if TEST_GPU
    train_many(usecuda=true)
end
