# An example of semi-supervised node classification

using Flux
using Flux: @functor, dropout, onecold, onehotbatch
using Flux.Losses: logitcrossentropy
using GraphNeuralNetworks
using MLDatasets: Cora
using Statistics, Random
using CUDA
CUDA.allowscalar(false)

struct GNN
    conv1
    conv2 
    dense
end

@functor GNN

function GNN(; nin, nhidden, nout)
    GNN(GCNConv(nin => nhidden, relu),
        GCNConv(nhidden => nhidden, relu), 
        Dense(nhidden, nout))
end

function (net::GNN)(fg, x)
    x = net.conv1(fg, x)
    x = dropout(x, 0.5)
    x = net.conv2(fg, x)
    x = net.dense(x)
    return x
end

function eval_loss_accuracy(X, y, ids, model, fg)
    ŷ = model(fg, X)
    l = logitcrossentropy(ŷ[:,ids], y[:,ids])
    acc = mean(onecold(ŷ[:,ids] |> cpu) .== onecold(y[:,ids] |> cpu))
    return (loss = round(l, digits=4), acc = round(acc*100, digits=2))
end

# arguments for the `train` function 
Base.@kwdef mutable struct Args
    η = 1f-3             # learning rate
    epochs = 100          # number of epochs
    seed = 17             # set seed > 0 for reproducibility
    use_cuda = true      # if true use cuda (if available)
    nhidden = 128        # dimension of hidden features
    infotime = 10 	     # report every `infotime` epochs
end

function train(; kws...)
    args = Args(; kws...)
    if args.seed > 0
        Random.seed!(args.seed)
        CUDA.seed!(args.seed)
    end
    
    if args.use_cuda && CUDA.functional()
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    data = Cora.dataset()
    fg = FeaturedGraph(data.adjacency_list) |> device
    X = data.node_features |> device
    y = onehotbatch(data.node_labels, 1:data.num_classes) |> device
    train_ids = data.train_indices |> device
    val_ids = data.val_indices |> device
    test_ids = data.test_indices |> device
    ytrain = y[:,train_ids]

    model = GNN(nin=size(X,1), 
                nhidden=args.nhidden, 
                nout=data.num_classes) |> device
    ps = Flux.params(model)
    opt = ADAM(args.η)

    @info "NUM NODES: $(fg.num_nodes)  NUM EDGES: $(fg.num_edges)"
    
    function report(epoch)
        train = eval_loss_accuracy(X, y, train_ids, model, fg)
        test = eval_loss_accuracy(X, y, test_ids, model, fg)        
        println("Epoch: $epoch   Train: $(train)   Test: $(test)")
    end
    
    ## TRAINING
    report(0)
    for epoch in 1:args.epochs
        gs = Flux.gradient(ps) do
            ŷ = model(fg, X)
            logitcrossentropy(ŷ[:,train_ids], ytrain)
        end

        Flux.Optimise.update!(opt, ps, gs)
        
        epoch % args.infotime == 0 && report(epoch)
    end
end

train()
