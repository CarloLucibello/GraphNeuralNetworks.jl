using Flux, GraphNeuralNetworks



struct EdgeConv2{NN, A} <: GNNLayer
    nn::NN
    aggr::A
end

Flux.@functor EdgeConv2

EdgeConv2(nn; aggr = max) = EdgeConv2(nn, aggr)

function (l::EdgeConv2)(g::AbstractGNNGraph, x)
    check_num_nodes(g, x)
    xj, xi = expand_srcdst(g, x)

    message(l, xi, xj, e) = l.nn(vcat(xi, xj .- xi))

    x = propagate(message, g, l.aggr, l, xi = xi, xj = xj, e = nothing)
    return x
end

function Base.show(io::IO, l::EdgeConv2)
    print(io, "EdgeConv2(", l.nn)
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end

function check_num_nodes(g::GNNGraph, x::AbstractArray)
    @assert g.num_nodes==size(x, ndims(x)) "Got $(size(x, ndims(x))) as last dimension size instead of num_nodes=$(g.num_nodes)"
    return true
end
function check_num_nodes(g::GNNGraph, x::Union{Tuple, NamedTuple})
    map(x -> check_num_nodes(g, x), x)
    return true
end

check_num_nodes(::GNNGraph, ::Nothing) = true

function check_num_nodes(g::GNNGraph, x::Tuple)
    @assert length(x) == 2
    check_num_nodes(g, x[1])
    check_num_nodes(g, x[2])
    return true
end

# x = (Xsrc, Xdst) = (Xj, Xi)
function check_num_nodes(g::GNNHeteroGraph, x::Tuple)
    @assert length(x) == 2
    @assert length(g.etypes) == 1
    nt1, _, nt2 = only(g.etypes)
    if x[1] isa AbstractArray
        @assert size(x[1], ndims(x[1])) == g.num_nodes[nt1]
    end
    if x[2] isa AbstractArray
        @assert size(x[2], ndims(x[2])) == g.num_nodes[nt2] 
    end
    return true
end


expand_srcdst(g::AbstractGNNGraph, x) = throw(ArgumentError("Invalid input type, expected matrix or tuple of matrices."))
expand_srcdst(g::AbstractGNNGraph, x::AbstractMatrix) = (x, x)
expand_srcdst(g::AbstractGNNGraph, x::Tuple{<:AbstractMatrix, <:AbstractMatrix}) = x


struct pap{W <: AbstractMatrix, B, F, A} <: GNNLayer
    weight::W
    bias::B
    σ::F
    aggr::A
end

Flux.@functor pap

using Flux: glorot_uniform, leakyrelu, GRUCell, @functor, batch




function check_num_nodes(g::GNNGraph, x::AbstractArray)
    @assert g.num_nodes==size(x, ndims(x)) "Got $(size(x, ndims(x))) as last dimension size instead of num_nodes=$(g.num_nodes)"
    return true
end
function check_num_nodes(g::GNNGraph, x::Union{Tuple, NamedTuple})
    map(x -> check_num_nodes(g, x), x)
    return true
end

check_num_nodes(::GNNGraph, ::Nothing) = true

function check_num_nodes(g::GNNGraph, x::Tuple)
    @assert length(x) == 2
    check_num_nodes(g, x[1])
    check_num_nodes(g, x[2])
    return true
end

# x = (Xsrc, Xdst) = (Xj, Xi)
function check_num_nodes2(g::GNNHeteroGraph, x::Tuple)
    print("po")
    @assert length(x) == 2
    @assert length(g.etypes) == 1
    nt1, _, nt2 = only(g.etypes)
    if x[1] isa AbstractArray
        @assert size(x[1], ndims(x[1])) == g.num_nodes[nt1]
    end
    if x[2] isa AbstractArray
        @assert size(x[2], ndims(x[2])) == g.num_nodes[nt2] 
    end
    return true
end


function Base.show(io::IO, l::pap)
    out_channel, in_channel = size(l.weight)
    print(io, "pap(", in_channel ÷ 2, " => ", out_channel)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end

"""
julia> g = rand_bipartite_heterograph((10, 15), 20)
GNNHeteroGraph:
  num_nodes: Dict(:A => 10, :B => 15)
  num_edges: Dict((:A, :to, :B) => 20, (:B, :to, :A) => 20)

julia> x = (A = rand(Float32, 64, 10), B = rand(Float32, 64, 15));

julia> layer = HeteroGraphConv((:A, :to, :B) => GraphConv(64 => 32, relu),
                               (:B, :to, :A) => GraphConv(64 => 32, relu));

julia> y = layer(g, x); # output is a named tuple

julia> size(y.A) == (32, 10) && size(y.B) == (32, 15)
true
"""


function (l::pap)(g::AbstractGNNGraph, x)
    
    check_num_nodes(g, x)
    xj, xi = expand_srcdst(g, x)
    m = propagate(copy_xj, g, l.aggr, xj = xj)
    print("AAASASAS", x, "\n\n\n\n")
    print("PAAA", size(xj), size(xi), size(m), size(vcat(x,m)), size(l.weight))
    #cant vcat m and xj both shud be 2,3 
    x = l.σ.(l.weight * vcat(xi, m) .+ l.bias)
    return x
end

function pap(ch::Pair{Int, Int}, σ = identity; aggr = mean,
                  init = glorot_uniform, bias::Bool = true)
    in, out = ch
    W = init(out, 2 * in)
    b = bias ? Flux.create_bias(W, true, out) : false
    pap(W, b, σ, aggr)
end


g = rand_bipartite_heterograph((2, 3), 6)
x = (A = rand(Float32, 4, 2), B = rand(Float32, 4, 3))

layers = HeteroGraphConv((:A, :to, :B) => pap(4 => 2, relu, bias = false, aggr = +),
                         (:B, :to, :A) => pap(4 => 2, relu, bias = false, aggr = +),
                         );

                       
y = layers(g, x); 
print(size(y.B), size(y.A))
size(y.A) == (2,2) && size(y.B) == (2,3)





iin = 2
oout = 3
W = glorot_uniform(oout, 2*iin)
g = g::AbstractGNNGraph




print(layers,"\n\n\n\n")  




hg = rand_bipartite_heterograph((2,3), 6)
x = (A = rand(Float32, 4,2), B = rand(Float32, 4, 3))
layers = HeteroGraphConv( (:A, :to, :B) => CGConv(4 => 2, relu),
                                    (:B, :to, :A) => CGConv(4 => 2, relu));
y = layers(hg, x); 
print(size(y.A))
print(size(y.B))
size(y.A) == (2,2) && size(y.B) == (2,3)




d, n = 3, 5  # Dimension of node features, number of nodes

# Initialize EGNNConv layers for different types of edges in the heterogeneous graph
layers = HeteroGraphConv([
    (:A, :to, :B) => EGNNConv((d, 0) => d; hidden_size = 2*d, residual = false),
    (:B, :to, :A) => EGNNConv((d, 0) => d; hidden_size = 2*d, residual = false)
])

# Generate random node features for each node type in the graph
x = (A = rand(Float32, d, 2), B = rand(Float32, d, 3))

y = layers(hg, x)


function (hgc::HeteroGraphConv)(g::GNNHeteroGraph, x::Union{NamedTuple,Dict}, h::Union{NamedTuple,Dict})
    function forw(l, et)
        sg = edge_type_subgraph(g, et)
        node1_t, _, node2_t = et
        # Ensure both x and h have the necessary node types before calling l
        if haskey(x, node1_t) && haskey(h, node1_t) && haskey(x, node2_t) && haskey(h, node2_t)
            return l(sg, (x[node1_t], x[node2_t]), (h[node1_t], h[node2_t]))
        else
            # Handle the case where either x or h does not have the necessary node types
            # This might involve returning some default value or throwing an error
            throw(ErrorException("Missing node types in x or h for edge type $et"))
        end
    end
    outs = [forw(l, et) for (l, et) in zip(hgc.layers, hgc.etypes)]
    dst_ntypes = [et[3] for et in hgc.etypes]
    return _reduceby_node_t(hgc.aggr, outs, dst_ntypes)
end



in_channel = 3
out_channel = 5
hin = 5
hout = 5
hidden = 5
h = (A = rand(Float32, 4, 5), # 5 nodes of type A with 4 features each
     B = rand(Float32, 4, 3)) # 3 nodes of type B with 4 features each

x = (A = rand(Float32, 3, 5), # 5 nodes of type A with 3D coordinates
     B = rand(Float32, 3, 3)) # 3 nodes of type B with 3D coordinates

hg = rand_bipartite_heterograph((2,3), 6)  # Create a heterogeneous graph with specified node types and edge count

# Adjusted layers to match feature sizes for A and B
layers = HeteroGraphConv(
    (:A, :to, :B) => EGNNConv((4, 0) => 4; hidden_size = 10, residual = false),
    (:B, :to, :A) => EGNNConv((4, 0) => 4; hidden_size = 10, residual = false)
)

# Assuming `hg` is your heterogeneous graph
# The call now correctly matches the expected input sizes
y = layers(hg, x, h)









hin_A, hout_A, hidden_A = 5, 5, 10
        hin_B, hout_B, hidden_B = 3, 3, 6
        num_nodes_A, num_nodes_B = 5, 3  # Number of nodes for types A and B
    
        # Create a random bipartite heterogeneous graph
        hg = rand_bipartite_heterograph((num_nodes_A, num_nodes_B), 15)  # Adjust the edge count as needed
    
        # Initialize EGNNConv layers for edges between node types A and B, and B and A
        layers = HeteroGraphConv([
            (:A, :to, :B) => EGNNConv((hin_A, 0) => hout_B; hidden_size = hidden_A, residual = false),
            (:B, :to, :A) => EGNNConv((hin_B, 0) => hout_A; hidden_size = hidden_B, residual = false)
        ])
    
        # Generate random node features `h` and coordinates `x` for each node type
        T = Float32  # Assuming the data type for features and coordinates is Float32
        h = randn(T, hin_A, num_nodes_A)
        x = (A = rand(T, 3, num_nodes_A), B = rand(T, 3, num_nodes_B))  # Assuming 3D coordinates
    
        # Test EGNNConv within the HeteroGraphConv framework
        y = layers(hg, x, h)
    
        # Assert the dimensions of the updated node features and coordinates for each node type
        @test size(y[:A].h) == (hout_A, num_nodes_A)
        @test size(y[:B].h) == (hout_B, num_nodes_B)
        @test size(y[:A].x) == (3, num_nodes_A)  # Assuming coordinates remain unchanged in dimension
        @test size(y[:B].x) == (3, num_nodes_B)









        hin = 5
        hout = 5
        hidden = 5
        l = EGNNConv(hin => hout, hidden)
        g = rand_graph(10, 20)
        x = rand(T, in_channel, g.num_nodes)
        h = randn(T, hin, g.num_nodes)
        hnew, xnew = l(g, h, x)
        size(hnew) == (hout, g.num_nodes)
        size(xnew) == (in_channel, g.num_nodes)
