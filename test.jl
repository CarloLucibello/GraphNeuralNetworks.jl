using GraphNeuralNetworks, Random, Flux, Test, CUDA, SparseArrays, CUDA.CUSPARSE

Random.seed!(17)
g = rand_graph(6, 14)
@test !has_self_loops(g)
x = rand(2, g.num_nodes)
l = GCNConv(2 => 2)
y = l(g, x)
s, t = edge_index(g)
A = adjacency_matrix(g)

g_gpu = g |> gpu
x_gpu = x |> gpu
l_gpu = l |> gpu
s_gpu, t_gpu = edge_index(g_gpu)
y_gpu = l_gpu(g_gpu, x_gpu)
A_gpu = adjacency_matrix(g_gpu)

@test Array(s_gpu) ≈ s
@test Array(t_gpu) ≈ t

@test Array(A_gpu) ≈ Array(A)
@test Array(degree(g_gpu)) ≈ Array(degree(g))

@test Array(y_gpu) ≈ y



# @testset "Conv Layers" begin
#     in_channel = 3
#     out_channel = 5
#     N = 4
#     T = Float32

#     adj1 =  [0 1 0 1
#              1 0 1 0
#              0 1 0 1
#              1 0 1 0]
    
#     g1 = GNNGraph(adj1, 
#             ndata=rand(T, in_channel, N), 
#             graph_type=GRAPH_T)
        
#     adj_single_vertex =  [0 0 0 1
#                           0 0 0 0
#                           0 0 0 1
#                           1 0 1 0]
    
#     g_single_vertex = GNNGraph(adj_single_vertex, 
#                                 ndata=rand(T, in_channel, N), 
#                                 graph_type=GRAPH_T)    

#     test_graphs = [g1, g_single_vertex]

#     @testset "GCNConv" begin
#         l = GCNConv(in_channel => out_channel)
#         for g in test_graphs
#             test_layer(l, g, rtol=1e-5, outsize=(out_channel, g.num_nodes))
#         end

#         l = GCNConv(in_channel => out_channel, tanh, bias=false)
#         for g in test_graphs
#             test_layer(l, g, rtol=1e-5, outsize=(out_channel, g.num_nodes))
#         end

#         l = GCNConv(in_channel => out_channel, add_self_loops=false)
#         test_layer(l, g1, rtol=1e-5, outsize=(out_channel, g1.num_nodes))
#     end
# end