# An example of semi-supervised node classification

using Flux
using Flux: @functor, dropout, onecold, onehotbatch, getindex
using Flux.Losses: logitbinarycrossentropy
using Flux.Data: DataLoader
using GraphNeuralNetworks
using MLDatasets: TUDataset
using Statistics, Random
using CUDA
CUDA.allowscalar(false)

function eval_loss_accuracy(model, data_loader, device)
    loss = 0.
    acc = 0.
    ntot = 0
    for (g, X, y) in data_loader
        g, X, y = g |> device, X |> device, y |> device
        n = length(y) 
        ŷ = model(g, X) |> vec
        loss += logitbinarycrossentropy(ŷ, y) * n 
        acc += mean((2 .* ŷ .- 1) .* (2 .* y .- 1) .> 0) * n
        ntot += n
    end        
    return (loss = round(loss/ntot, digits=4), acc = round(acc*100/ntot, digits=2))
end

struct GNNData
    g
    X
    y
end

Base.getindex(data::GNNData, i::Int) = getindex(data, [i])

function Base.getindex(data::GNNData, i::AbstractVector)
    sg, nodemap = subgraph(data.g, i)
    return (sg, data.X[:,nodemap], data.y[i])
end

# Flux's Dataloader compatibility. Related PR https://github.com/FluxML/Flux.jl/pull/1683
Flux.Data._nobs(data::GNNData) = data.g.num_graphs
Flux.Data._getobs(data::GNNData, i) = data[i] 

function process_dataset(data)
    g = GNNGraph(data.source, data.target, num_nodes=data.num_nodes, graph_indicator=data.graph_indicator)
    X = Array{Float32}(onehotbatch(data.node_labels, 0:6))
    # The dataset also has edge features but we won't be using them
    # E = Array{Float32}(onehotbatch(data.edge_labels, sort(unique(data.edge_labels))))
    y = (1 .+ Array{Float32}(data.graph_labels)) ./ 2
    @assert all(∈([0,1]), y) # binary classification 
    return GNNData(g, X, y)
end

# arguments for the `train` function 
Base.@kwdef mutable struct Args
    η = 1f-3             # learning rate
    batchsize = 64      # batch size (number of graphs in each batch)
    epochs = 200         # number of epochs
    seed = 17             # set seed > 0 for reproducibility
    usecuda = true      # if true use cuda (if available)
    nhidden = 128        # dimension of hidden features
    infotime = 10 	     # report every `infotime` epochs
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
    full_data = TUDataset("MUTAG")
    
    @info "MUTAG DATASET
            num_nodes: $(full_data.num_nodes)  
            num_edges: $(full_data.num_edges)  
            num_graphs: $(full_data.num_graphs)"
    
    perm = randperm(full_data.num_graphs)
    dtrain = process_dataset(full_data[perm[1:NUM_TRAIN]]) 
    dtest = process_dataset(full_data[perm[NUM_TRAIN+1:end]]) 
    train_loader = DataLoader(dtrain, batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader(dtest, batchsize=args.batchsize, shuffle=false)
    
    # DEFINE MODEL

    nin = size(dtrain.X, 1)
    nhidden = args.nhidden
    
    model = GNNChain(GraphConv(nin => nhidden, relu),
                     Dropout(0.5),
                     GraphConv(nhidden => nhidden, relu),
                     GlobalPool(mean), 
                     Dense(nhidden, 1))  |> device

    ps = Flux.params(model)
    opt = ADAM(args.η)

    
    # LOGGING FUNCTION

    function report(epoch)
        train = eval_loss_accuracy(model, train_loader, device)
        test = eval_loss_accuracy(model, test_loader, device)
        println("Epoch: $epoch   Train: $(train)   Test: $(test)")
    end
    
    # TRAIN
    
    report(0)
    for epoch in 1:args.epochs
        for (g, X, y) in train_loader
            g, X, y = g |> device, X |> device, y |> device
            gs = Flux.gradient(ps) do
                ŷ = model(g, X) |> vec
                logitbinarycrossentropy(ŷ, y)
            end
            Flux.Optimise.update!(opt, ps, gs)
        end
        
        epoch % args.infotime == 0 && report(epoch)
    end
end

train()
