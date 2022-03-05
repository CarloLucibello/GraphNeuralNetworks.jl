# An example of graph classification

using Flux
using Flux:onecold, onehotbatch
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
    for g in data_loader
        g = g |> device
        n = g.num_graphs
        y = g.gdata.y
        ŷ = model(g, g.ndata.x) |> vec
        loss += logitbinarycrossentropy(ŷ, y) * n 
        acc += mean((ŷ .> 0) .== y) * n
        ntot += n
    end 
    return (loss = round(loss/ntot, digits=4), acc = round(acc*100/ntot, digits=2))
end

function getdataset()
    tudata = TUDataset("MUTAG")
    
    x = Array{Float32}(onehotbatch(tudata.node_labels, 0:6))
    y = (1 .+ Array{Float32}(tudata.graph_labels)) ./ 2
    @assert all(∈([0,1]), y) # binary classification 
    
    ## The dataset also has edge features but we won't be using them
    # e = Array{Float32}(onehotbatch(data.edge_labels, sort(unique(data.edge_labels))))
    
    gall = GNNGraph(tudata.source, tudata.target, 
                num_nodes=tudata.num_nodes, 
                graph_indicator=tudata.graph_indicator,
                ndata=(; x), gdata=(; y))

    return [getgraph(gall, i) for i=1:gall.num_graphs]
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
    
    data = getdataset()
    shuffle!(data)
    
    train_loader = DataLoader(data[1:NUM_TRAIN], batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader(data[NUM_TRAIN+1:end], batchsize=args.batchsize, shuffle=false)
    
    # DEFINE MODEL

    nin = size(data[1].ndata.x, 1)
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
        for g in train_loader
            g = g |> device
            gs = Flux.gradient(ps) do
                ŷ = model(g, g.ndata.x) |> vec
                logitbinarycrossentropy(ŷ, g.gdata.y)
            end
            Flux.Optimise.update!(opt, ps, gs)
        end
        
        epoch % args.infotime == 0 && report(epoch)
    end
end

train()
