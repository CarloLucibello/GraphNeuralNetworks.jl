# An example of semi-supervised node classification

using Flux
using Flux: @functor, dropout, onecold, onehotbatch
using Flux.Losses: logitbinarycrossentropy
using GraphNeuralNetworks
using MLDatasets: TUDataset
using Statistics, Random
using CUDA
CUDA.allowscalar(false)

function eval_loss_accuracy(model, g, X, y)
    ŷ = model(g, X) |> vec
    l = logitbinarycrossentropy(ŷ, y)
    acc = mean((2 .* ŷ .- 1) .* (2 .* y .- 1) .> 0)
    return (loss = round(l, digits=4), acc = round(acc*100, digits=2))
end

struct GNNData
    g
    X
    y
end


function getdataset(idxs)
    data = TUDataset("MUTAG")[idxs]
    @info "MUTAG: num_nodes: $(data.num_nodes)  num_edges: $(data.num_edges)  num_graphs: $(data.num_graphs)"
    g = GNNGraph(data.source, data.target, num_nodes=data.num_nodes, graph_indicator=data.graph_indicator)
    X = Array{Float32}(onehotbatch(data.node_labels, 0:6))
    # E = Array{Float32}(onehotbatch(data.edge_labels, sort(unique(data.edge_labels))))
    y = (1 .+ Array{Float32}(data.graph_labels)) ./ 2
    @assert all(∈([0,1]), y) # binary classification 
    return GNNData(g, X, y)
end

# arguments for the `train` function 
Base.@kwdef mutable struct Args
    η = 1f-3             # learning rate
    epochs = 1000          # number of epochs
    seed = 17             # set seed > 0 for reproducibility
    use_cuda = false      # if true use cuda (if available)
    nhidden = 128        # dimension of hidden features
    infotime = 10 	     # report every `infotime` epochs
end

function train(; kws...)
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)
    
    if args.use_cuda && CUDA.functional()
        device = gpu
        args.seed > 0 && CUDA.seed!(args.seed)
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    # LOAD DATA
    
    permindx = randperm(188)
    ntrain = 150
    gtrain, Xtrain, ytrain = getdataset(permindx[1:ntrain]) 
    gtest, Xtest, ytest = getdataset(permindx[ntrain+1:end]) 
    
    # DEFINE MODEL

    nin = size(Xtrain,1)
    nhidden = args.nhidden
    
    model = GNNChain(GCNConv(nin => nhidden, relu),
                     Dropout(0.5),
                     GCNConv(nhidden => nhidden, relu),
                     GlobalPool(mean), 
                     Dense(nhidden, 1))  |> device

    ps = Flux.params(model)
    opt = ADAM(args.η)

    
    # LOGGING FUNCTION

    function report(epoch)
        train = eval_loss_accuracy(model, gtrain, Xtrain, ytrain)
        test = eval_loss_accuracy(model, gtest, Xtest, ytest)
        println("Epoch: $epoch   Train: $(train)   Test: $(test)")
    end
    
    # TRAIN
    
    report(0)
    for epoch in 1:args.epochs
        # for (g, X, y) in train_loader
            gs = Flux.gradient(ps) do
                ŷ = model(gtrain, Xtrain) |> vec
                logitbinarycrossentropy(ŷ, ytrain)
            end
            Flux.Optimise.update!(opt, ps, gs)
        # end
        
        epoch % args.infotime == 0 && report(epoch)
    end
end

# train()
