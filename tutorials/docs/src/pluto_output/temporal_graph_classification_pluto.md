```@raw html
<style>
    #documenter-page table {
        display: table !important;
        margin: 2rem auto !important;
        border-top: 2pt solid rgba(0,0,0,0.2);
        border-bottom: 2pt solid rgba(0,0,0,0.2);
    }

    #documenter-page pre, #documenter-page div {
        margin-top: 1.4rem !important;
        margin-bottom: 1.4rem !important;
    }

    .code-output {
        padding: 0.7rem 0.5rem !important;
    }

    .admonition-body {
        padding: 0em 1.25em !important;
    }
</style>

<!-- PlutoStaticHTML.Begin -->
<!--
    # This information is used for caching.
    [PlutoStaticHTML.State]
    input_sha = "33f6319bf52620ad6122ffc183f2167efc3646ae0018bd149248c49be7b3de4e"
    julia_version = "1.10.5"
-->

<div class="markdown"><h1>Temporal Graph classification with GraphNeuralNetworks.jl</h1><p>In this tutorial, we will learn how to extend the graph classification task to the case of temporal graphs, i.e., graphs whose topology and features are time-varying.</p><p>We will design and train a simple temporal graph neural network architecture to classify subjects' gender (female or male) using the temporal graphs extracted from their brain fMRI scan signals. Given the large amount of data, we will implement the training so that it can also run on the GPU.</p></div>


```
## Import
```@raw html
<div class="markdown">
<p>We start by importing the necessary libraries. We use <code>GraphNeuralNetworks.jl</code>, <code>Flux.jl</code> and <code>MLDatasets.jl</code>, among others.</p></div>

<pre class='language-julia'><code class='language-julia'>begin
    using Flux
    using GraphNeuralNetworks
    using Statistics, Random
    using LinearAlgebra
    using MLDatasets: TemporalBrains
    using CUDA
    using cuDNN
end</code></pre>



```
## Dataset: TemporalBrains
```@raw html
<div class="markdown">
<p>The TemporalBrains dataset contains a collection of functional brain connectivity networks from 1000 subjects obtained from resting-state functional MRI data from the <a href="https://www.humanconnectome.org/study/hcp-young-adult/document/extensively-processed-fmri-data-documentation">Human Connectome Project (HCP)</a>.  Functional connectivity is defined as the temporal dependence of neuronal activation patterns of anatomically separated brain regions.</p><p>The graph nodes represent brain regions and their number is fixed at 102 for each of the 27 snapshots, while the edges, representing functional connectivity, change over time. For each snapshot, the feature of a node represents the average activation of the node during that snapshot. Each temporal graph has a label representing gender ('M' for male and 'F' for female) and age group (22-25, 26-30, 31-35, and 36+). The network's edge weights are binarized, and the threshold is set to 0.6 by default.</p></div>

<pre class='language-julia'><code class='language-julia'>brain_dataset = TemporalBrains()</code></pre>
<pre class="code-output documenter-example-output" id="var-brain_dataset">dataset TemporalBrains:
  graphs  =&gt;    1000-element Vector{MLDatasets.TemporalSnapshotsGraph}</pre>


<div class="markdown"><p>After loading the dataset from the MLDatasets.jl package, we see that there are 1000 graphs and we need to convert them to the <code>TemporalSnapshotsGNNGraph</code> format. So we create a function called <code>data_loader</code> that implements the latter and splits the dataset into the training set that will be used to train the model and the test set that will be used to test the performance of the model.</p></div>

<pre class='language-julia'><code class='language-julia'>function data_loader(brain_dataset)
    graphs = brain_dataset.graphs
    dataset = Vector{TemporalSnapshotsGNNGraph}(undef, length(graphs))
    for i in 1:length(graphs)
        graph = graphs[i]
        dataset[i] = TemporalSnapshotsGNNGraph(GraphNeuralNetworks.mlgraph2gnngraph.(graph.snapshots))
        # Add graph and node features
        for t in 1:27
            s = dataset[i].snapshots[t]
            s.ndata.x = [I(102); s.ndata.x']
        end
        dataset[i].tgdata.g = Float32.(Flux.onehot(graph.graph_data.g, ["F", "M"]))
    end
    # Split the dataset into a 80% training set and a 20% test set
    train_loader = dataset[1:200]
    test_loader = dataset[201:250]
    return train_loader, test_loader
end;</code></pre>



<div class="markdown"><p>The first part of the <code>data_loader</code> function calls the <code>mlgraph2gnngraph</code> function for each snapshot, which takes the graph and converts it to a <code>GNNGraph</code>. The vector of <code>GNNGraph</code>s is then rewritten to a <code>TemporalSnapshotsGNNGraph</code>.</p><p>The second part adds the graph and node features to the temporal graphs, in particular it adds the one-hot encoding of the label of the graph (in this case we directly use the identity matrix) and appends the mean activation of the node of the snapshot (which is contained in the vector <code>dataset[i].snapshots[t].ndata.x</code>, where <code>i</code> is the index indicating the subject and <code>t</code> is the snapshot). For the graph feature, it adds the one-hot encoding of gender.</p><p>The last part splits the dataset.</p></div>


```
## Model
```@raw html
<div class="markdown">
<p>We now implement a simple model that takes a <code>TemporalSnapshotsGNNGraph</code> as input. It consists of a <code>GINConv</code> applied independently to each snapshot, a <code>GlobalPool</code> to get an embedding for each snapshot, a pooling on the time dimension to get an embedding for the whole temporal graph, and finally a <code>Dense</code> layer.</p><p>First, we start by adapting the <code>GlobalPool</code> to the <code>TemporalSnapshotsGNNGraphs</code>.</p></div>

<pre class='language-julia'><code class='language-julia'>function (l::GlobalPool)(g::TemporalSnapshotsGNNGraph, x::AbstractVector)
    h = [reduce_nodes(l.aggr, g[i], x[i]) for i in 1:(g.num_snapshots)]
    sze = size(h[1])
    reshape(reduce(hcat, h), sze[1], length(h))
end</code></pre>



<div class="markdown"><p>Then we implement the constructor of the model, which we call <code>GenderPredictionModel</code>, and the foward pass.</p></div>

<pre class='language-julia'><code class='language-julia'>begin
    struct GenderPredictionModel
        gin::GINConv
        mlp::Chain
        globalpool::GlobalPool
        f::Function
        dense::Dense
    end
    
    Flux.@layer GenderPredictionModel
    
    function GenderPredictionModel(; nfeatures = 103, nhidden = 128, activation = relu)
        mlp = Chain(Dense(nfeatures, nhidden, activation), Dense(nhidden, nhidden, activation))
        gin = GINConv(mlp, 0.5)
        globalpool = GlobalPool(mean)
        f = x -&gt; mean(x, dims = 2)
        dense = Dense(nhidden, 2)
        GenderPredictionModel(gin, mlp, globalpool, f, dense)
    end
    
    function (m::GenderPredictionModel)(g::TemporalSnapshotsGNNGraph)
        h = m.gin(g, g.ndata.x)
        h = m.globalpool(g, h)
        h = m.f(h)
        m.dense(h)
    end
    
end</code></pre>



```
## Training
```@raw html
<div class="markdown">
<p>We train the model for 100 epochs, using the Adam optimizer with a learning rate of 0.001. We use the <code>logitbinarycrossentropy</code> as the loss function, which is typically used as the loss in two-class classification, where the labels are given in a one-hot format. The accuracy expresses the number of correct classifications. </p></div>

<pre class='language-julia'><code class='language-julia'>lossfunction(ŷ, y) = Flux.logitbinarycrossentropy(ŷ, y);</code></pre>


<pre class='language-julia'><code class='language-julia'>function eval_loss_accuracy(model, data_loader)
    error = mean([lossfunction(model(g), g.tgdata.g) for g in data_loader])
    acc = mean([round(100 * mean(Flux.onecold(model(g)) .==     Flux.onecold(g.tgdata.g)); digits = 2) for g in data_loader])
    return (loss = error, acc = acc)
end;</code></pre>


<pre class='language-julia'><code class='language-julia'>function train(dataset; usecuda::Bool, kws...)

    if usecuda && CUDA.functional() #check if GPU is available 
        my_device = gpu
        @info "Training on GPU"
    else
        my_device = cpu
        @info "Training on CPU"
    end
    
    function report(epoch)
        train_loss, train_acc = eval_loss_accuracy(model, train_loader)
        test_loss, test_acc = eval_loss_accuracy(model, test_loader)
        println("Epoch: $epoch  $((; train_loss, train_acc))  $((; test_loss, test_acc))")
        return (train_loss, train_acc, test_loss, test_acc)
    end

    model = GenderPredictionModel() |&gt; my_device

    opt = Flux.setup(Adam(1.0f-3), model)

    train_loader, test_loader = data_loader(dataset)
    train_loader = train_loader |&gt; my_device
    test_loader = test_loader |&gt; my_device

    report(0)
    for epoch in 1:100
        for g in train_loader
            grads = Flux.gradient(model) do model
                ŷ = model(g)
                lossfunction(vec(ŷ), g.tgdata.g)
            end
            Flux.update!(opt, model, grads[1])
        end
        if  epoch % 10 == 0
            report(epoch)
        end
    end
    return model
end;
</code></pre>


<pre class='language-julia'><code class='language-julia'>train(brain_dataset; usecuda = true)</code></pre>
<pre class="code-output documenter-example-output" id="var-hash305203">GenderPredictionModel(GINConv(Chain(Dense(103 =&gt; 128, relu), Dense(128 =&gt; 128, relu)), 0.5), Chain(Dense(103 =&gt; 128, relu), Dense(128 =&gt; 128, relu)), GlobalPool{typeof(mean)}(Statistics.mean), var"#4#5"(), Dense(128 =&gt; 2))  # 30_082 parameters, plus 29_824 non-trainable</pre>


<div class="markdown"><p>We set up the training on the GPU because training takes a lot of time, especially when working on the CPU.</p></div>


```
## Conclusions
```@raw html
<div class="markdown">
<p>In this tutorial, we implemented a very simple architecture to classify temporal graphs in the context of gender classification using brain data. We then trained the model on the GPU for 100 epochs on the TemporalBrains dataset. The accuracy of the model is approximately 75-80%, but can be improved by fine-tuning the parameters and training on more data.</p></div>

<!-- PlutoStaticHTML.End -->
```

