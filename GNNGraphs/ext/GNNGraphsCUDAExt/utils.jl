
GNNGraphs.iscuarray(x::AnyCuArray) = true


function sort_edge_index(u::AnyCuArray, v::AnyCuArray)
    dev = get_device(u)
    cdev = cpu_device()
    u, v = u |> cdev, v |> cdev
    #TODO proper cuda friendly implementation
    sort_edge_index(u, v) |> dev
end
