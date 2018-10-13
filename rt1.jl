using Printf
using LinearAlgebra

abstract type Hitable end

struct Ray
    origin::Array{Float64, 1}
    direction::Array{Float64, 1}
end

struct HitRecord
    t::Float64
    p::Array{Float64}
    normal::Array{Float64}
    HitRecord() = new(0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    HitRecord(t,p,normal) = new(t, p, normal)
end

struct Camera
    lower_left_corner::Array{Float64}
    horizontal::Array{Float64}
    vertical::Array{Float64}
    origin::Array{Float64}
end

function getRay(c::Camera, u::Float64, v::Float64)::Ray
    return Ray(c.origin, c.lower_left_corner + u * c.horizontal + v * c.vertical)
end


function point_at_parameter(r::Ray, t::Float64)
    return r.origin + t * r.direction
end

struct Sphere <: Hitable
    center::Array{Float64}
    radius::Float64
end

function hit(sphere::Sphere, r::Ray, t_min::Float64, t_max::Float64)::Union{HitRecord, Nothing}
    a = dot(r.direction, r.direction)
    s2o = r.origin - sphere.center
    b = 2.0 * dot(r.direction, s2o)
    c = dot(s2o, s2o) - sphere.radius^2
    D = b^2 - 4*a*c
    if (D > 0)
        t1 = (-b - sqrt(D)) / (2.0 * a)
        if (t1 < t_max && t1 > t_min)
            p = point_at_parameter(r, t1)
            return HitRecord(t1,
                              p,
                              (p - sphere.center) / sphere.radius)
        end
        
        t2 = (-b + sqrt(D)) / (2.0 * a)
        if (t2 < t_max && t2 > t_min)
            p = point_at_parameter(r, t1)
            return HitRecord(t2,
                              p, 
                              (p - sphere.center) / sphere.radius)
        end
        return nothing
    end
end

function hit(hitables::Array{Hitable}, r::Ray, t_min::Float64, t_max::Float64)::Union{HitRecord, Nothing}
    result::Union{HitRecord, Nothing} = nothing
    closest_t = t_max

    for i = 1:length(hitables)
        hit_res = hit(hitables[i], r, t_min, closest_t)
        if (isa(hit_res, HitRecord))
            closest_t = hit_res.t
            result = hit_res
        end
    end

    return result
end

function random_in_unit_sphere()
    p::Array{Float64} = [0.0, 0.0, 0.0]
    while true
        p = 2.0 * [rand(), rand(), rand()] - [1.0, 1.0, 1.0]
        norm(p)^2 >= 1.0 && break;
    end
    return p
end

function color(r::Ray, world::Array{Hitable})
    hitres = hit(world, r, 0.0, typemax(Float64))
    if (isa(hitres, HitRecord))
        target = hitres.p + hitres.normal + random_in_unit_sphere()
        return 0.5 * color( Ray(hitres.p, target - hitres.p), world)
    else
        u = r.direction / norm(r.direction)
        t = 0.5 * (u[2] + 1.0)
        return (1.0 - t) * [1.0, 1.0, 1.0] + t * [0.5, 0.7, 1.0]
    end
end

function main()
    nx::Int = 200;
    ny::Int = 100;
    ns::Int = 100;
    @printf("P3\n%d %d\n255\n", nx, ny);

    camera = Camera([-2.0, -1.0, -1.0], [4.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0])

    world::Array{Hitable} = [Sphere([0.0, 0.0, -1.0], 0.5), Sphere([0.0, -100.5, -1], 100)]

    for j::Int = ny - 1 : -1 : 0
        for i::Int = 0 : nx - 1

            col::Array{Float64} = [0.0, 0.0, 0.0]
            for s = 1:ns
                u::Float64 = (convert(Float64, i) + rand()) / nx
                v::Float64 = (convert(Float64, j) + rand()) / ny
                r = getRay(camera, u, v)
                col += color(r, world)
            end

            col /= convert(Float64, ns)
                    
            ir::Int = trunc(255.99 * col[1])
            ig::Int = trunc(255.99 * col[2])
            ib::Int = trunc(255.99 * col[3])
            @printf("%d %d %d\n", ir, ig, ib)
        end
    end
end

main()
