const Vec3 = Array{Float64, 1}

abstract type Hitable end
abstract type Material end
abstract type Texture end

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
    ray::Ray
    albedo::Vec3
    pdf::Float64
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

abstract type PDF
end

struct CosinePDF <: PDF
    uvw::onb

    function CosinePDF(w::Vec3)
        return new(onb(w))
    end
end

function value(pdf::CosinePDF, direction::Vec3)::Float64
    cosine = dot(normalize(direction), pdf.uvw.w)

    if (cosine > 0.0)
        return cosine / pi
    else
        return 0.0
    end
end

function generate(pdf::CosinePDF)::Vec3
    return to_local(pdf.uvw, random_cosine_direction())
end

