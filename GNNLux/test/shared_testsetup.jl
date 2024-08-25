@testsetup module SharedTestSetup

import Reexport: @reexport

@reexport using Test
@reexport using GNNLux
@reexport using Lux
@reexport using StableRNGs
@reexport using Random, Statistics

using LuxTestUtils: test_gradients, AutoReverseDiff, AutoTracker, AutoForwardDiff, AutoEnzyme

export test_lux_layer

function test_lux_layer(rng::AbstractRNG, l, g::GNNGraph, x; 
            outputsize=nothing, sizey=nothing, container=false,
            atol=1.0f-2, rtol=1.0f-2, edge_weight=nothing) 

    if container
        @test l isa GNNContainerLayer
    else
        @test l isa GNNLayer
    end

    ps = LuxCore.initialparameters(rng, l)
    st = LuxCore.initialstates(rng, l)
    @test LuxCore.parameterlength(l) == LuxCore.parameterlength(ps)
    @test LuxCore.statelength(l) == LuxCore.statelength(st)

    if edge_weight !== nothing
        y, st′ = l(g, x, ps, st)
    else          
        y, st′ = l(g, x, edge_weight, ps, st)
    end
            
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

end
