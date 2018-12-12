function normalize(v::Vec3)::Vec3
    return v / norm(v)
end

function random_in_unit_sphere()
    p::Vec3 = [0.0, 0.0, 0.0]
    while true
        p = 2.0 * [rand(), rand(), rand()] - [1.0, 1.0, 1.0]
        norm(p)^2 < 1.0 && break;
    end
    return p
end

function random_in_unit_disk()
    p::Vec3 = [0.0, 0.0, 0.0]
    while true
        p = 2.0 * [rand(), rand(), 0.0] - [1.0, 1.0, 0.0]
        norm(p)^2 < 1.0 && break;
    end
    return p
end
