# An example of link prediction using negative and positive samples.
# Ported from https://docs.dgl.ai/tutorials/blitz/4_link_predict.html#sphx-glr-tutorials-blitz-4-link-predict-py

using Flux
# Link prediction task
# https://arxiv.org/pdf/2102.12557.pdf

using Flux: onecold, onehotbatch
using Flux.Losses: logitbinarycrossentropy
using GraphNeuralNetworks
using MLDatasets: PubMed, Cora
using Statistics, Random, LinearAlgebra
using CUDA
# using MLJBase: AreaUnderCurve
CUDA.allowscalar(false)

# arguments for the `train` function 
Base.@kwdef mutable struct Args
    η = 1f-3             # learning rate
    epochs = 200          # number of epochs
    seed = 17             # set seed > 0 for reproducibility
    usecuda = false      # if true use cuda (if available)
    nhidden = 64        # dimension of hidden features
    infotime = 10 	     # report every `infotime` epochs
end

struct DotPredictor end

function (::DotPredictor)(g, x)
    z = apply_edges((xi, xj, e) -> sum(xi .* xj, dims=1), g, xi=x, xj=x)
    return vec(z)
end

using ChainRulesCore

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
    s, t = edge_index(g)
    eids = randperm(g.num_edges)
    test_size = round(Int, g.num_edges * 0.1)
    
    test_pos_s, test_pos_t = s[eids[1:test_size]], t[eids[1:test_size]]
    test_pos_g = GNNGraph(test_pos_s, test_pos_t, num_nodes=g.num_nodes)
    
    train_pos_s, train_pos_t = s[eids[test_size+1:end]], t[eids[test_size+1:end]]
    train_pos_g = GNNGraph(train_pos_s, train_pos_t, num_nodes=g.num_nodes)

    test_neg_g = negative_sample(g, num_neg_edges=test_size)
    
    ### DEFINE MODEL #########
    nin, nhidden = size(X,1), args.nhidden
    
    model = WithGraph(GNNChain(GCNConv(nin => nhidden, relu),
                               GCNConv(nhidden => nhidden)),
                      train_pos_g) |> device

    pred = DotPredictor()

    ps = Flux.params(model)
    opt = ADAM(args.η)

    ### LOSS FUNCTION ############

    function loss(pos_g, neg_g = nothing)
        h = model(X)
        if neg_g === nothing
            # we sample a negative graph at each training step
            neg_g = negative_sample(pos_g)
        end
        pos_score = pred(pos_g, h)
        neg_score = pred(neg_g, h)
        scores = [pos_score; neg_score]
        labels = [fill!(similar(pos_score), 1); fill!(similar(neg_score), 0)]
        return logitbinarycrossentropy(scores, labels)
    end

    # function accuracy(pos_g, neg_g)
    #     h = model(train_pos_g, X)
    #     pos_score = pred(pos_g, h)
    #     neg_score = pred(neg_g, h)
    #     scores = [pos_score; neg_score]
    #     labels = [fill!(similar(pos_score), 1); fill!(similar(neg_score), 0)]
    #     return logitbinarycrossentropy(scores, labels)
    # end
    
    ### LOGGING FUNCTION
    function report(epoch)
        train_loss = loss(train_pos_g)
        test_loss = loss(test_pos_g, test_neg_g)
        println("Epoch: $epoch   Train: $(train_loss)   Test: $(test_loss)")
    end
    
    ### TRAINING
    report(0)
    for epoch in 1:args.epochs
        gs = Flux.gradient(() -> loss(train_pos_g), ps)
        Flux.Optimise.update!(opt, ps, gs)
        epoch % args.infotime == 0 && report(epoch)
    end
end

# train()