
g = rand_bipartite_heterograph((2, 3), 6)
import NNlib.relu

x = (A = rand(Float32, 4, 2), B = rand(Float32, 4, 3));
# nn = Dense(4,2)

#l = GINConv(nn, 0.01f0, aggr = mean)
layers = HeteroGraphConv( (:A, :to, :B) => GATConv(4 => 2, relu),
                               (:B, :to, :A) => GATConv(4 => 2, relu));


y = layer(g, x); # output is a named tuple



size(y.A) == (2, 2) && size(y.B) == (2, 3)


gs = rand_graph(3, 2)
edge_index(gs)
x = rand(Float32, 4, 3)
# create layer
l = GCNConv(4 => 2) 

# forward pass
y = l(gz, x)       # size:  5 × num_nodes

########
gs = rand_bipartite_graph((2, 3), 6)

################################################
import NNlib.relu
g = rand_bipartite_heterograph((2,3), 6)

x = (A = rand(Float32, 4,2), B = rand(Float32, 4, 3));

layers = HeteroGraphConv( (:A, :to, :B) => GATConv(4 => 2, relu),
                               (:B, :to, :A) => GATConv(4 => 2, relu));

y = layers(g, x); # output is a named tuple

size(y.A) == (2,2) && size(y.B) == (2,3)


edim = 10
RTOL_HIGH = 1e-5
nn = Dense(edim, 4 * 2)

l = NNConv(4 => 2, nn, tanh, bias = true, aggr = +)

layers = HeteroGraphConv( (:A, :to, :B) => l);

layers(l, g, rtol = RTOL_HIGH, outsize = (2, g.num_nodes[:B]))


RTOL_LOW = 1e-2

################
# create data
s = [1, 1, 2, 1, 2]
t = [3, 2, 3, 1, 2]
gz = GNNGraph(s, t)
x = randn(3, gz.num_nodes)

# create layer
l = GCNConv(3 => 6) 

# forward pass
y = l(gz, x)       # size:  5 × num_nodes

# convolution with edge weights and custom normalization function
w = [1.1, 0.1, 2.3, 0.5]
custom_norm_fn(d) = 1 ./ sqrt.(d + 1)  # Custom normalization function
y = l(g, x, w, custom_norm_fn)

# Edge weights can also be embedded in the graph.
g = GNNGraph(s, t, w)
l = GCNConv(3 => 5, use_edge_weight=true) 
y = l(g, x) # same as l(g, x, w) 







######################3
struct GraphConv{W <: AbstractMatrix, B, F, A} <: GNNLayer
    weight1::W
    weight2::W
    bias::B
    σ::F
    aggr::A
end

@functor GraphConv

function GraphConv(ch::Pair{Int, Int}, σ = identity; aggr = +,
                   init = glorot_uniform, bias::Bool = true)
    in, out = ch
    W1 = init(out, in)
    W2 = init(out, in)
    b = bias ? Flux.create_bias(W1, true, out) : false
    GraphConv(W1, W2, b, σ, aggr)
end

function (l::GraphConv)(g::AbstractGNNGraph, x)
    check_num_nodes(g, x)
    xj, xi = expand_srcdst(g, x)
    m = propagate(copy_xj, g, l.aggr, xj = xj)
    x = l.σ.(l.weight1 * xi .+ l.weight2 * m .+ l.bias)
    return x
end



############
gs = rand_graph(6, 2)
edge_index(gs)
x = rand(Float32, 3, 6)
# create layer
l = GCNConv(3 => 6) 

# forward pass
y = l(gz, x)       # size:  5 × num_nodes

################
# function (l::GCNConv)(g::AbstractGNNGraph, 
#                       x, #x::AbstractMatrix{T},
#                       edge_weight::EW = nothing,
#                       norm_fn::Function = d -> 1 ./ sqrt.(d)  
#                       ) where {T, EW <: Union{Nothing, AbstractVector}}

#     check_gcnconv_input(g, edge_weight)

#     xj, xi = expand_srcdst(g, x)
#     println("sizexi", size(xi))
#     println("sizexj", size(xj))

#     println("dupa: ", l.weight)
#     #typeof("x")
#     #println("T: ", T)
#     #println("koniec T")

    # if l.add_self_loops
    #     if g isa GNNHeteroGraph
    #         for edge_t in g.etypes
    #             src_t, _, tgt_t = edge_t
    #             src_t === tgt_t && add_self_loops(g, edge_t) 
    #         end
    #     else
    #         g = add_self_loops(g)
    #     end
    #     if edge_weight !== nothing
#             # Pad weights with ones
#             # TODO for ADJMAT_T the new edges are not generally at the end
#             edge_weight = [edge_weight; fill!(similar(edge_weight, g.num_nodes), 1)]
#             @assert length(edge_weight) == g.num_edges
#         end
#     end
#     Dout, Din = size(l.weight)
#     if Dout < Din
#         # multiply before convolution if it is more convenient, otherwise multiply after
#         #println("l weight: ", typeof(l.weight))
#         println("xi: ", typeof(x))
#         println("size l weight: ", size(l.weight))
#         println("l weight: ", l.weight)
#         x = map( d -> d .* l.weight, x)
#         println("xXXXXXXXXXXx: ", x)
#         size(x)
#         size(x[1])
#         size(x[2])
#     end
#     # if edge_weight !== nothing
#     #     d = degree(g, nothing; dir = :in, edge_weight)
#     # else
#     #     d = degree(g, nothing; dir = :in, edge_weight = l.use_edge_weight)
#     #     #d = degree(g) # debug
#     # end
#     d = [[3, 2], [3, 2, 2]]'
#     println("degree: ", d)
#     c = norm_fn.(d)
#     println("sizex1: ", size(x[1]))
#     println("sizex2: ", size(x[2]))
#     println("x: ", size(x), " ", x)
#     println("c: ", c)
#     println("c1: ", size(c[1]), " ", c[1])
#     println("c2: ", size(c[2]), " ", c[2])
    
#     println(size(c))
#     println("Sssssssss:",size(x[2]))
#     #x = [[x[1] .* c[1]], [x[2] .* x[2]]]
#     x = x .* c'

#    println("X: ", x)
#    println("X: ", size(x[1]))
#    println("X: ", size(x[2]))
#     if edge_weight !== nothing
#         println("one")
#         x = propagate(e_mul_xj, g, +, xj = xj, e = edge_weight)
#     elseif l.use_edge_weight
#         println("two")
#         x = propagate(w_mul_xj, g, +, xj = xj)
#     else
#         println("three")
#         x = propagate(copy_xj, g, +, xj = xj) ### this is outputting one mtrx
#     end
#     println("x: ", x)
#     println("dimx1: ", size(x[1]))
#     println("dimx2: ", size(x[2]))
#     x = xi .* c[1]' ### this is experimental, should be x .* c'
#     if Dout >= Din
#         x = l.weight * xi
#     end
#     return l.σ.(x .+ l.bias)
# end
