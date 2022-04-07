    function u = heat1(init, tspan, c)
    dom=[0 1];
    L = chebop(@(u) c*diff(u, 2), dom);
    L.bc = 'periodic';
    u0 = init;
    u = expm(L, tspan, u0);