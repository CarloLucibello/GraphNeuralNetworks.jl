# An example of link prediction using negative and positive samples.
# Ported from https://docs.dgl.ai/tutorials/blitz/4_link_predict.html#sphx-glr-tutorials-blitz-4-link-predict-py
# See the comparison paper https://arxiv.org/pdf/2102.12557.pdf for more details

using Flux
using Flux: onecold, onehotbatch
using Flux.Losses: logitbinarycrossentropy
using GraphNeuralNetworks
using MLDatasets: PubMed
using Statistics, Random, LinearAlgebra
using CUDA
CUDA.allowscalar(false)

# arguments for the `train` function 
Base.@kwdef mutable struct Args
    η = 1.0f-3             # learning rate
    epochs = 200          # number of epochs
    seed = 17             # set seed > 0 for reproducibility
    usecuda = true      # if true use cuda (if available)
    nhidden = 64        # dimension of hidden features
    infotime = 10      # report every `infotime` epochs
end

# We define our own edge prediction layer but could also 
# use GraphNeuralNetworks.DotDecoder instead.
struct DotPredictor end

function (::DotPredictor)(g, x)
    z = apply_edges((xi, xj, e) -> sum(xi .* xj, dims = 1), g, xi = x, xj = x)
    # z = apply_edges(xi_dot_xj, g, xi=x, xj=x) # Same with built-in method
    return vec(z)
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

    ### LOAD DATA
    g = mldataset2gnngraph(PubMed())

    # Print some info
    display(g)
    @show is_bidirected(g)
    @show has_self_loops(g)
    @show has_multi_edges(g)
    @show mean(degree(g))
    isbidir = is_bidirected(g)

    # Move to device
    g = g |> device
    X = g.ndata.features

    #### TRAIN/TEST splits
    # With bidirected graph, we make sure that an edge and its reverse
    # are in the same split 
    train_pos_g, test_pos_g = rand_edge_split(g, 0.9, bidirected = isbidir)
    test_neg_g = negative_sample(g, num_neg_edges = test_pos_g.num_edges,
                                 bidirected = isbidir)

    ### DEFINE MODEL #########
    nin, nhidden = size(X, 1), args.nhidden

    # We embed the graph with positive training edges in the model 
    model = WithGraph(GNNChain(GCNConv(nin => nhidden, relu),
                               GCNConv(nhidden => nhidden)),
                      train_pos_g) |> device

    pred = DotPredictor()

    opt = Flux.setup(Adam(args.η), model)

    ### LOSS FUNCTION ############

    function loss(model, pos_g, neg_g = nothing; with_accuracy = false)
        h = model(X)
        if neg_g === nothing
            # We sample a negative graph at each training step
            neg_g = negative_sample(pos_g, bidirected = isbidir)
        end
        pos_score = pred(pos_g, h)
        neg_score = pred(neg_g, h)
        scores = [pos_score; neg_score]
        labels = [fill!(similar(pos_score), 1); fill!(similar(neg_score), 0)]
        l = logitbinarycrossentropy(scores, labels)
        if with_accuracy
            acc = 0.5 * mean(pos_score .>= 0) + 0.5 * mean(neg_score .< 0)
            return l, acc
        else
            return l
        end
    end

    ### LOGGING FUNCTION
    function report(epoch)
        train_loss, train_acc = loss(model, train_pos_g, with_accuracy = true)
        test_loss, test_acc = loss(model, test_pos_g, test_neg_g, with_accuracy = true)
        println("Epoch: $epoch  $((; train_loss, train_acc))  $((; test_loss, test_acc))")
    end

    ### TRAINING
    report(0)
    for epoch in 1:(args.epochs)
        grads = Flux.gradient(model -> loss(model, train_pos_g), model)
        Flux.update!(opt, model, grads[1])
        epoch % args.infotime == 0 && report(epoch)
    end
end

train()
