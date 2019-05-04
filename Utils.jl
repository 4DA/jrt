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

# some theory on this:
# https://www.scratchapixel.com/lessons/3d-basic-rendering/global-illumination-path-tracing/global-illumination-path-tracing-practical-implementation

# sample space = 2PI steradians
# Int[p(ω) dω, {ω, 0, 2PI}] = 1
# p(ω) = C = const, bc all samples have the same chance
# p(ω) = C = 1 / 2PI | by evaluating the integral
# dω = sinθ * dθ * dφ | differential solid angle
# p(φ, θ) * dω * dφ = p(ω) * dω
# p(φ, θ) = sinθ / 2PI | join probability distribution.

# A marginal distribution is one in which the probability of a given variable
# (for example θ) can be expressed without any reference to the other variables
# (for example ϕ). You can do this by integrating the given PDF with respect to
# one of the variables. In our case, we get (will integrate p(θ,ϕ) with respect
# to ϕ):

# P(θ) = Int[p(φ, θ) dφ, {φ, 0, 2PI}] = sinθ

# The conditional probability function gives the probably function of one of the
# variables (for example ϕ) when the other variable (θ) is known. For continuous
# function (which the case of our PDFs in this example) this can simply be
# computed as the joint probability distribution p(θ,ϕ) over the marginal
# distribution p(θ):
# p(φ) = p(θ, φ) / p(θ) = 1 / 2PI

# Now that we have the PDFs for θ and ϕ, we need to compute their respective
# CDFs and inverse them.

function random_cosine_direction()::Vec3
    r1 = rand()
    r2 = rand()
    z = sqrt(1.0 - r2)
    phi = 2 * pi * r1
    x = cos(phi) * 2.0 * sqrt(r2)
    y = sin(phi) * 2.0 * sqrt(r2)
    return [x, y, z]
end
