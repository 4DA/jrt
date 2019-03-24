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

struct onb
    u::Vec3
    v::Vec3
    w::Vec3

    function onb(n::Vec3)
        u::Vec3 = zeros(3)
        v::Vec3 = zeros(3)
        w::Vec3 = n
        a::Vec3 = zeros(3)

        if (abs(n[1]) > 0.9)
            a = [0.0, 1.0, 0.0]
        else
            a = [1.0, 0.0, 0.0]
        end

        v = normalize(cross(w, a))
        u = cross(w, v)

        return new(u, v, w)
    end
end

function to_local(basis::onb, a::Vec3)::Vec3
    return a[1] * basis.u + a[2] * basis.v + a[3] * basis.w
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
