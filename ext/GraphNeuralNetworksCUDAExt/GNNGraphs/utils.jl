
GNNGraphs.iscuarray(x::AnyCuArray) = true


function sort_edge_index(u::AnyCuArray, v::AnyCuArray)
    #TODO proper cuda friendly implementation
    sort_edge_index(u |> Flux.cpu, v |> Flux.cpu) |> Flux.gpu
end