# An example of link prediction using negative and positive samples.
# Ported from https://docs.dgl.ai/tutorials/blitz/4_link_predict.html#sphx-glr-tutorials-blitz-4-link-predict-py

using Flux
using Flux: onecold, onehotbatch
using Flux.Losses: logitbinarycrossentropy
using GraphNeuralNetworks
using GraphNeuralNetworks: ones_like, zeros_like
using MLDatasets: Cora
using Statistics, Random, LinearAlgebra
using CUDA
CUDA.allowscalar(false)

"""
Transform vector of cartesian indexes into a tuple of vectors containing integers.
"""
ci2t(ci::AbstractVector{<:CartesianIndex}, dims) = ntuple(i -> map(x -> x[i], ci), dims)

# arguments for the `train` function 
Base.@kwdef mutable struct Args
    η = 1f-3             # learning rate
    epochs = 200          # number of epochs
    seed = 17             # set seed > 0 for reproducibility
    usecuda = false      # if true use cuda (if available)
    nhidden = 128        # dimension of hidden features
    infotime = 10 	     # report every `infotime` epochs
end

struct DotPredictor end

function (::DotPredictor)(g, x)
    z = apply_edges((xi, xj, e) -> sum(xi .* xj, dims=1), g, xi=x, xj=x)
    return vec(z)
end

function train(; kws...)
    # args = Args(; kws...)
    args = Args()

    args.seed > 0 && Random.seed!(args.seed)
    
    if args.usecuda && CUDA.functional()
        device = gpu
        args.seed > 0 && CUDA.seed!(args.seed)
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    ### LOAD DATA
    data = Cora.dataset()
    g = GNNGraph(data.adjacency_list) |> device
    X = data.node_features |> device
    
    #### SPLIT INTO NEGATIVE AND POSITIVE SAMPLES
    # Split edge set for training and testing
    s, t = edge_index(g)
    eids = randperm(g.num_edges)
    test_size = round(Int, g.num_edges * 0.1)
    train_size = g.num_edges - test_size
    test_pos_s, test_pos_t = s[eids[1:test_size]], t[eids[1:test_size]]
    train_pos_s, train_pos_t = s[eids[test_size+1:end]], t[eids[test_size+1:end]]

    # Find all negative edges and split them for training and testing
    adj = adjacency_matrix(g)
    adj_neg = 1 .- adj - I
    neg_s, neg_t = ci2t(findall(adj_neg .> 0), 2)

    neg_eids = randperm(length(neg_s))[1:g.num_edges]
    test_neg_s, test_neg_t = neg_s[neg_eids[1:test_size]], neg_t[neg_eids[1:test_size]]
    train_neg_s, train_neg_t = neg_s[neg_eids[test_size+1:end]], neg_t[neg_eids[test_size+1:end]]
    # train_neg_s, train_neg_t = neg_s[neg_eids[train_size+1:end]], neg_t[neg_eids[train_size+1:end]]
    
    train_pos_g = GNNGraph((train_pos_s, train_pos_t), num_nodes=g.num_nodes)
    train_neg_g = GNNGraph((train_neg_s, train_neg_t), num_nodes=g.num_nodes)

    test_pos_g = GNNGraph((test_pos_s, test_pos_t), num_nodes=g.num_nodes)
    test_neg_g = GNNGraph((test_neg_s, test_neg_t), num_nodes=g.num_nodes)
    
    @show train_pos_g test_pos_g train_neg_g test_neg_g

    ### DEFINE MODEL
    nin, nhidden = size(X,1), args.nhidden
    
    model = GNNChain(GCNConv(nin => nhidden, relu),
                     GCNConv(nhidden => nhidden)) |> device

    pred = DotPredictor()

    ps = Flux.params(model)
    opt = ADAM(args.η)

    ### LOSS FUNCTION

    function loss(pos_g, neg_g)
        h = model(train_pos_g, X)
        pos_score = pred(pos_g, h)
        neg_score = pred(neg_g, h)
        scores = [pos_score; neg_score]
        labels = [ones_like(pos_score); zeros_like(neg_score)]
        return logitbinarycrossentropy(scores, labels)
    end

    function accuracy(pos_g, neg_g)
        h = model(train_pos_g, X)
        pos_score = pred(pos_g, h)
        neg_score = pred(neg_g, h)
        scores = [pos_score; neg_score]
        labels = [ones_like(pos_score); zeros_like(neg_score)]
        return logitbinarycrossentropy(scores, labels)
    end
    
    ### LOGGING FUNCTION
    function report(epoch)
        train_loss = loss(train_pos_g, train_neg_g)
        test_loss = loss(test_pos_g, test_neg_g)
        println("Epoch: $epoch   Train: $(train_loss)   Test: $(test_loss)")
    end
    
    ### TRAINING
    report(0)
    for epoch in 1:args.epochs
        gs = Flux.gradient(() -> loss(train_pos_g, train_neg_g), ps)
        Flux.Optimise.update!(opt, ps, gs)
        epoch % args.infotime == 0 && report(epoch)
    end
end

# train()
