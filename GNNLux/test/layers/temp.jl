

@testset "GINConv" begin
    nn = Chain(Dense(in_dims => out_dims, relu), Dense(out_dims => out_dims))
    l = GINConv(nn, 0.5)
    test_lux_layer(rng, l, g, x, sizey=(out_dims,g.num_nodes), container=true)
end

@testset "SGConv" begin
    l = SGConv(in_dims => out_dims, 2)
    test_lux_layer(rng, l, g, x, outputsize=(out_dims,))
end



function test_lux_layer(rng::AbstractRNG, l, g::GNNGraph, x; 
            outputsize=nothing, sizey=nothing, container=false,
            atol=1.0f-2, rtol=1.0f-2)


    ps = LuxCore.initialparameters(rng, l)
    st = LuxCore.initialstates(rng, l)
    @test LuxCore.parameterlength(l) == LuxCore.parameterlength(ps)
    @test LuxCore.statelength(l) == LuxCore.statelength(st)
    
    y, st′ = l(g, x, ps, st)
    @test eltype(y) == eltype(x)
    if outputsize !== nothing
        @test LuxCore.outputsize(l) == outputsize
    end
    if sizey !== nothing
        @test size(y) == sizey
    elseif outputsize !== nothing
        @test size(y) == (outputsize..., g.num_nodes)
    end
    
    loss = (x, ps) -> sum(first(l(g, x, ps, st)))
    test_gradients(loss, x, ps; atol, rtol, skip_backends=[AutoReverseDiff(), AutoTracker(), AutoForwardDiff(), AutoEnzyme()])
end

using LuxTestUtils: test_gradients, AutoReverseDiff, AutoTracker, AutoForwardDiff, AutoEnzyme
using StableRNGs

"""
MEGNetConv{Flux.Chain{Tuple{Flux.Dense{typeof(relu), Matrix{Float32}, Vector{Float32}}, Flux.Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}}}, Flux.Chain{Tuple{Flux.Dense{typeof(relu), Matrix{Float32}, Vector{Float32}}, Flux.Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}}}, typeof(mean)}(Chain(Dense(9 => 5, relu), Dense(5 => 5)), Chain(Dense(8 => 5, relu), Dense(5 => 5)), Statistics.mean)
"""

g = rand_graph(10, 40, seed=1234)
    in_dims = 3
    out_dims = 5
    x = randn(Float32, in_dims, 10)
    rng = StableRNG(1234)
    l = MEGNetConv(in_dims => out_dims)
    l
    l isa GNNContainerLayer
    test_lux_layer(rng, l, g, x, sizey=((out_dims, g.num_nodes), (out_dims, g.num_edges)), container=true)


        ps = LuxCore.initialparameters(rng, l)
        st = LuxCore.initialstates(rng, l)
        edata = rand(T, in_channel, g.num_edges)

        (x_new, e_new), st_new = l(g, x, ps, st)
        
        @test size(x_new) == (out_dims, g.num_nodes)
        @test size(e_new) == (out_dims, g.num_edges)
        

nn = Chain(Dense(in_dims => out_dims, relu), Dense(out_dims => out_dims))
l = GINConv(nn, 0.5)
test_lux_layer(rng, l, g, x, sizey=(out_dims,g.num_nodes), container=true)



    hin = 6
    hout = 7
    hidden = 8
    l = EGNNConv(hin => hout, hidden)
    ps = LuxCore.initialparameters(rng, l)
    st = LuxCore.initialstates(rng, l)
    h = randn(rng, Float32, hin, g.num_nodes)
    (hnew, xnew), stnew = l(g, h, x, ps, st)
    @test size(hnew) == (hout, g.num_nodes)
    @test size(xnew) == (in_dims, g.num_nodes)


    l = MEGNetConv(in_dims => out_dims)
    l
    l isa GNNContainerLayer
    test_lux_layer(rng, l, g, x, sizey=((out_dims, g.num_nodes), (out_dims, g.num_edges)), container=true)


        ps = LuxCore.initialparameters(rng, l)
        st = LuxCore.initialstates(rng, l)