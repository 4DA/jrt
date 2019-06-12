const Vec3 = Array{Float64, 1}

abstract type Hitable end
abstract type Material end
abstract type Texture end

# There are two functions a ​ pdf ​ needs to support:
# 1. What is your value at this location?
# 2. Return a random vector that is distributed appropriately.
abstract type PDF
end

struct AABB
    min::Vec3
    max::Vec3
end

struct Ray
    origin::Vec3
    direction::Vec3
    time::Float64
    Ray(origin, direction, time) = new(origin, direction, time)
    Ray(origin, direction) = new(origin, direction, 0.0)
end

struct HitRecord
    t::Float64
    p::Vec3
    normal::Vec3
    material::Material
    u::Float64
    v::Float64
end

struct ScatterRecord
    specular_ray::Ray
    is_specular::Bool
    attenuation::Vec3
    pdf::PDF
end

struct onb
    u::Vec3
    v::Vec3
    w::Vec3

    # build ortho-normal basis from normal
    function onb(n::Vec3)
        u::Vec3 = zeros(3)
        v::Vec3 = zeros(3)
        w::Vec3 = normalize(n)
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

