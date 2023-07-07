RTOL_LOW = 1e-2
RTOL_HIGH = 1e-5
ATOL_LOW = 1e-3

in_channel = 3
out_channel = 5
N = 4
T = Float32

g1 = GNNGraph(rand_graph(N,8),
                ndata = rand(T, in_channel, N),
                graph_type = :sparse)

@testset "TGCNCell" begin
    l = TGCNCell(in_channel => out_channel)
    test_layer(l, g1, rtol = RTOL_HIGH, outsize = (out_channel, g1.num_nodes))
end