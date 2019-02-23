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

