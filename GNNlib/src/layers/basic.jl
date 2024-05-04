function dot_decoder(g, x)
    return apply_edges(xi_dot_xj, g, xi = x, xj = x)
end
