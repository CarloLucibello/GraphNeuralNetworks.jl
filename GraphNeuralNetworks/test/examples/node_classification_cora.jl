@testmodule TrainingExampleModule begin
    using Flux
    using Flux: onecold, onehotbatch
    using Flux.Losses: logitcrossentropy
    using GraphNeuralNetworks
    using MLDatasets: Cora
    using Statistics, Random
    using Test
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    
    function eval_loss_accuracy(X, y, ids, model, g)
        ŷ = model(g, X)
        l = logitcrossentropy(ŷ[:, ids], y[:, ids])
        acc = mean(onecold(ŷ[:, ids]) .== onecold(y[:, ids]))
        return (loss = round(l, digits = 4), acc = round(acc * 100, digits = 2))
    end

    # arguments for the `train` function 
    Base.@kwdef mutable struct Args
        η = 5.0f-3            # learning rate
        epochs = 10         # number of epochs
        seed = 17           # set seed > 0 for reproducibility
        use_gpu = false     # if true use gpu (if available)
        nhidden = 64        # dimension of hidden features
    end

    function train(Layer; verbose = false, kws...)
        args = Args(; kws...)
        args.seed > 0 && Random.seed!(args.seed)

        if args.use_gpu
            device = gpu_device(force=true)
            Random.seed!(default_device_rng(device))
        else
            device = cpu_device()
        end

        # LOAD DATA
        dataset = Cora()
        classes = dataset.metadata["classes"]
        g = mldataset2gnngraph(dataset) |> device
        X = g.ndata.features
        y = onehotbatch(g.ndata.targets |> cpu, classes) |> device # https://github.com/FluxML/OneHotArrays.jl/issues/16
        train_mask = g.ndata.train_mask
        test_mask = g.ndata.test_mask
        ytrain = y[:, train_mask]

        nin, nhidden, nout = size(X, 1), args.nhidden, length(classes)

        ## DEFINE MODEL
        model = GNNChain(Layer(nin, nhidden),
                        #  Dropout(0.5),
                        Layer(nhidden, nhidden),
                        Dense(nhidden, nout)) |> device

        opt = Flux.setup(Adam(args.η), model)

        ## TRAINING
        function report(epoch)
            train = eval_loss_accuracy(X, y, train_mask, model, g)
            test = eval_loss_accuracy(X, y, test_mask, model, g)
            println("Epoch: $epoch   Train: $(train)   Test: $(test)")
        end

        verbose && report(0)
        @time for epoch in 1:(args.epochs)
            grad = Flux.gradient(model) do model
                ŷ = model(g, X)
                logitcrossentropy(ŷ[:, train_mask], ytrain)
            end
            Flux.update!(opt, model, grad[1])
            verbose && report(epoch)
        end

        train_res = eval_loss_accuracy(X, y, train_mask, model, g)
        test_res = eval_loss_accuracy(X, y, test_mask, model, g)
        return train_res, test_res
    end

    function train_many(; use_gpu = false)
        for (layer, Layer) in [
            ("GCNConv", (nin, nout) -> GCNConv(nin => nout, relu)),
            ("ResGatedGraphConv", (nin, nout) -> ResGatedGraphConv(nin => nout, relu)),
            ("GraphConv", (nin, nout) -> GraphConv(nin => nout, relu, aggr = mean)),
            ("SAGEConv", (nin, nout) -> SAGEConv(nin => nout, relu)),
            ("GATConv", (nin, nout) -> GATConv(nin => nout, relu)),
            ("GINConv", (nin, nout) -> GINConv(Dense(nin, nout, relu), 0.01, aggr = mean)),
            ("TransformerConv",
            (nin, nout) -> TransformerConv(nin => nout, concat = false,
                                            add_self_loops = true, root_weight = false,
                                            heads = 2)),
            ## ("ChebConv", (nin, nout) -> ChebConv(nin => nout, 2)), # not working on gpu
            ## ("NNConv", (nin, nout) -> NNConv(nin => nout)),  # needs edge features
            ## ("GatedGraphConv", (nin, nout) -> GatedGraphConv(nout, 2)),  # needs nin = nout
            ## ("EdgeConv",(nin, nout) -> EdgeConv(Dense(2nin, nout, relu))), # Fits the training set but does not generalize well
        ]
            @show layer
            @time train_res, test_res = train(Layer; use_gpu, verbose = false)
            # @show train_res, test_res
            @test train_res.acc > 94
            @test test_res.acc > 69
        end
    end
end # module

@testitem "training example" setup=[TrainingExampleModule] begin
    using .TrainingExampleModule
    TrainingExampleModule.train_many()
end

@testitem "training example GPU" setup=[TestModule, TrainingExampleModule] tags=[:gpu] begin
    using .TestModule # for loading gpu packages
    using .TrainingExampleModule
    TrainingExampleModule.train_many(use_gpu = true)
end

