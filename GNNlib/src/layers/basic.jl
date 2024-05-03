function dot_encoder(g, x)
    return apply_edges(xi_dot_xj, g, xi = x, xj = x)
end
