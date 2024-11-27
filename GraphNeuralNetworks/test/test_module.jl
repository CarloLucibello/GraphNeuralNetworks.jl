@testmodule TestModule begin

using GraphNeuralNetworks
using Test
using Statistics, Random
using Flux
using Functors: fmapstructure_with_path
using Graphs
using ChainRulesTestUtils, FiniteDifferences
using Zygote
using SparseArrays
using Pkg

## Uncomment below to change the default test settings
# ENV["GNN_TEST_CPU"] = "false"
# ENV["GNN_TEST_CUDA"] = "true"
# ENV["GNN_TEST_AMDGPU"] = "true"
# ENV["GNN_TEST_Metal"] = "true"

if get(ENV, "GNN_TEST_CUDA", "false") == "true"
    # Pkg.add(["CUDA", "cuDNN"])
    using CUDA
    CUDA.allowscalar(false)
end
if get(ENV, "GNN_TEST_AMDGPU", "false") == "true"
    # Pkg.add("AMDGPU")
    using AMDGPU
    AMDGPU.allowscalar(false)
end
if get(ENV, "GNN_TEST_Metal", "false") == "true"
    # Pkg.add("Metal")
    using Metal
    Metal.allowscalar(false)
end


# from Base
export mean, randn, SparseArrays, AbstractSparseMatrix

# from other packages
export Flux, gradient, Dense, Chain, relu, random_regular_graph, erdos_renyi,
       BatchNorm, LayerNorm, Dropout, Parallel

# from this module
export D_IN, D_OUT, GRAPH_TYPES, TEST_GRAPHS,
       test_gradients, finitediff_withgradient, 
       check_equal_leaves


const D_IN = 3
const D_OUT = 5

function finitediff_withgradient(f, x...)
    y = f(x...)
    # We set a range to avoid domain errors
    fdm = FiniteDifferences.central_fdm(5, 1, max_range=1e-2)
    return y, FiniteDifferences.grad(fdm, f, x...)
end

function check_equal_leaves(a, b; rtol=1e-4, atol=1e-4)
    fmapstructure_with_path(a, b) do kp, x, y
        if x isa AbstractArray
            # @show kp
            @test x ≈ y rtol=rtol atol=atol
        # elseif x isa Number
        #     @show kp
        #     @test x ≈ y rtol=rtol atol=atol
        end
    end
end

function test_gradients(
            f,
            graph::GNNGraph, 
            xs...;
            rtol=1e-5, atol=1e-5,
            test_gpu = false,
            test_grad_f = true,
            test_grad_x = true,
            compare_finite_diff = true,
            loss = (f, g, xs...) -> mean(f(g, xs...)),
            )

    if !test_gpu && !compare_finite_diff
        error("You should either compare finite diff vs CPU AD \
               or CPU AD vs GPU AD.")
    end

    ## Let's make sure first that the forward pass works.
    l = loss(f, graph, xs...)
    @test l isa Number
    if test_gpu
        gpu_dev = gpu_device(force=true)
        cpu_dev = cpu_device()
        graph_gpu = graph |> gpu_dev
        xs_gpu = xs |> gpu_dev
        f_gpu = f |> gpu_dev
        l_gpu = loss(f_gpu, graph_gpu, xs_gpu...)
        @test l_gpu isa Number
    end

    if test_grad_x
        # Zygote gradient with respect to input.
        y, g = Zygote.withgradient((xs...) -> loss(f, graph, xs...), xs...)
        
        if compare_finite_diff
            # Cast to Float64 to avoid precision issues.
            f64 = f |> Flux.f64
            xs64 = xs .|> Flux.f64
            y_fd, g_fd = finitediff_withgradient((xs...) -> loss(f64, graph, xs...), xs64...)
            @test y ≈ y_fd rtol=rtol atol=atol
            check_equal_leaves(g, g_fd; rtol, atol)
        end

        if test_gpu
            # Zygote gradient with respect to input on GPU.
            y_gpu, g_gpu = Zygote.withgradient((xs...) -> loss(f_gpu, graph_gpu, xs...), xs_gpu...)
            @test get_device(g_gpu) == get_device(xs_gpu)
            @test y_gpu ≈ y rtol=rtol atol=atol
            check_equal_leaves(g_gpu |> cpu_dev, g; rtol, atol)
        end
    end

    if test_grad_f
        # Zygote gradient with respect to f.
        y, g = Zygote.withgradient(f -> loss(f, graph, xs...), f)

        if compare_finite_diff
            # Cast to Float64 to avoid precision issues.
            f64 = f |> Flux.f64
            ps, re = Flux.destructure(f64)
            y_fd, g_fd = finitediff_withgradient(ps -> loss(re(ps),graph, xs...), ps)
            g_fd = (re(g_fd[1]),)
            @test y ≈ y_fd rtol=rtol atol=atol
            check_equal_leaves(g, g_fd; rtol, atol)
        end

        if test_gpu
            # Zygote gradient with respect to f on GPU.
            y_gpu, g_gpu = Zygote.withgradient(f -> loss(f,graph_gpu, xs_gpu...), f_gpu)
            # @test get_device(g_gpu) == get_device(xs_gpu)
            @test y_gpu ≈ y rtol=rtol atol=atol
            check_equal_leaves(g_gpu |> cpu_dev, g; rtol, atol)
        end
    end
    return true
end


function generate_test_graphs(graph_type)
    adj1 = [0 1 0 1
            1 0 1 0
            0 1 0 1
            1 0 1 0]

    g1 = GNNGraph(adj1,
                    ndata = rand(Float32, D_IN, 4);
                    graph_type)

    adj_single_vertex = [0 0 0 1
                            0 0 0 0
                            0 0 0 1
                            1 0 1 0]

    g_single_vertex = GNNGraph(adj_single_vertex,
                                ndata = rand(Float32, D_IN, 4);
                                graph_type)

    return (g1, g_single_vertex)
end

GRAPH_TYPES = [:coo, :dense, :sparse]
TEST_GRAPHS = [generate_test_graphs(:coo)...,
               generate_test_graphs(:dense)...,
               generate_test_graphs(:sparse)...]

end # testmodule

