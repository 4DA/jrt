# function normalize(v::Vec3)::Vec3
#     return v / norm(v)
# end

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


# Let’s review now because that was most of the concepts that underlie MC ray tracers.
# 1. You have an integral of f(x) over some domain​ [a,b]
# 2. You pick a pdf p that is non-zero over [a,b]
# 3. You average a whole ton of f(r)/p(r) where r is a random number ​ r with pdf ​ p.

# This will always converge to the right answer. The more ​ p ​ follows f, the faster it converges.

# p(dir) = cos(theta) / pi
# r2 = Integral[0, theta, 2 * pi * (cos(t) / pi) * sin(t)] = 1 - cos^2(theta)
# so cos(theta) = sqrt(1-r2)
function random_cosine_direction()::Vec3
    r1 = rand()
    r2 = rand()
    z = sqrt(1.0 - r2)
    phi = 2 * pi * r1
    x = cos(phi) * 2.0 * sqrt(r2)
    y = sin(phi) * 2.0 * sqrt(r2)
    return [x, y, z]
end
