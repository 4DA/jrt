using Printf
using LinearAlgebra
using Base

abstract type Hitable end
abstract type Material end

const Vec3 = Array{Float64, 1}

struct Sphere <: Hitable
    center::Vec3
    radius::Float64
    material::Material
end

struct Ray
    origin::Vec3
    direction::Vec3
end

struct HitRecord
    t::Float64
    p::Vec3
    normal::Vec3
    material::Material
end

struct ScatterRecord
    ray::Ray
    attenuation::Vec3
end

struct Camera
    lower_left_corner::Vec3
    horizontal::Vec3
    vertical::Vec3
    origin::Vec3
end

struct Lambertian <: Material
    albedo::Vec3
end

struct Metal <: Material
    albedo::Vec3
    fuzz::Float64
    Metal(a, f) = new(a, clamp(f, 0.0, 1.0))
    Metal(a) = new(a, 0.0)
end

struct Dielectric <: Material
    ref_idx::Float64
end

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

function schlick(cosine::Float64, ref_idx::Float64)
    r0 = ((1.0 - ref_idx) / (1.0 + ref_idx)) ^ 2
    return r0 + (1 - r0) * (1.0 - cosine) ^ 5
end

function refract(v::Vec3, n::Vec3, ni_over_nt::Float64)::Union{Vec3, Nothing}
    uv = normalize(v)

    dt = dot(uv, n)
    discriminant = 1.0 - ni_over_nt^2 * (1.0 - dt^2)

    if (discriminant > 0.0)
        return ni_over_nt * (uv - n * dt) - n * sqrt(discriminant)
    else
        return nothing
    end
end

function reflect(v::Array{Float64}, n::Array{Float64})::Array{Float64}
    return v - 2.0 * dot(v,n) * n
end

function scatter(m::Lambertian, r_in::Ray, hit::HitRecord)::Union{ScatterRecord, Nothing}
    target = hit.p + hit.normal + random_in_unit_sphere()
    scattered = Ray(hit.p, target - hit.p)
    return ScatterRecord(scattered, m.albedo)
end

function scatter(m::Metal, r_in::Ray, hit::HitRecord)::Union{ScatterRecord, Nothing}
    reflected = reflect(r_in.direction / norm(r_in.direction), hit.normal)
    scattered = Ray(hit.p, reflected + m.fuzz * random_in_unit_sphere())
    return ScatterRecord(scattered, m.albedo)
end

function scatter(d::Dielectric, r_in::Ray, hit::HitRecord)::Union{ScatterRecord, Nothing}
    outward_normal::Vec3 = hit.normal
    ni_over_nt::Float64 = 1.0
    attenuation = [1.0, 1.0, 1.0]
    reflected = reflect(r_in.direction, hit.normal)
    refracted::Vec3 = [1.0, 1.0, 1.0]
    cosine::Float64 = 0.0
    reflect_prob::Float64 = 0.0

    if (dot(r_in.direction, hit.normal) > 0.0)
        outward_normal = -hit.normal
        ni_over_nt = d.ref_idx
        cosine = d.ref_idx * dot(r_in.direction, hit.normal) / norm(r_in.direction)
    else
        outward_normal = hit.normal
        ni_over_nt = 1.0 / d.ref_idx
        cosine = -dot(r_in.direction, hit.normal) / norm(r_in.direction)
    end

    refract_res = refract(r_in.direction, outward_normal, ni_over_nt)

    if (isa(refract_res, Vec3))
        reflect_prob = schlick(cosine, d.ref_idx)
        scattered = Ray(hit.p, refract_res)
    else
        reflect_prob = 1.0
        scattered = Ray(hit.p, reflected)
    end

    if (rand() < reflect_prob)
        return ScatterRecord(Ray(hit.p, reflected), attenuation)
    else
        return ScatterRecord(Ray(hit.p, refract_res), attenuation)
    end
end


function getRay(c::Camera, u::Float64, v::Float64)::Ray
    return Ray(c.origin, c.lower_left_corner + u * c.horizontal + v * c.vertical)
end


function point_at_parameter(r::Ray, t::Float64)
    return r.origin + t * r.direction
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

        # @printf(Base.fdio(2), "hit1 = <%f,%f,%f> | t = %f\n",
        #         p[1],
        #         p[2],
        #         p[3],
        #         t1
        #         )
            return HitRecord(t1,
                              p,
                             (p - sphere.center) / sphere.radius,
                             sphere.material)
        end
        
        t2 = (-b + sqrt(D)) / (2.0 * a)
        if (t2 < t_max && t2 > t_min)
            p = point_at_parameter(r, t2)

        # @printf(Base.fdio(2), "hit2 = <%f,%f,%f> | t = %f\n",
        #         p[1],
        #         p[2],
        #         p[3],
        #         t2,
        #         )

            return HitRecord(t2,
                              p, 
                             (p - sphere.center) / sphere.radius,
                             sphere.material)
        end
    end
    return nothing
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

function color(r::Ray, world::Array{Hitable}, depth::Int64)::Array{Float64}
    hitres = hit(world, r, 0.001, typemax(Float64))
    if (isa(hitres, HitRecord))
        if (depth < 50)
            scatterRes = scatter(hitres.material, r, hitres)
            if (isa(scatterRes, ScatterRecord))
                return scatterRes.attenuation .* color(scatterRes.ray, world, depth + 1)
            end
        end
        return [0.0, 0.0, 0.0]
    else
        u = r.direction / norm(r.direction)
        t = 0.5 * (u[2] + 1.0)
        return (1.0 - t) * [1.0, 1.0, 1.0] + t * [0.5, 0.7, 1.0]
    end
end

function main()
    nx::Int = 200;
    ny::Int = 100;
    ns::Int = 50;
    @printf("P3\n%d %d\n255\n", nx, ny);

    camera = Camera([-2.0, -1.0, -1.0], [4.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0])

    world::Array{Hitable} = [
        Sphere([0.0, 0.0, -1.0], 0.5, Lambertian([0.1, 0.2, 0.5])),
        Sphere([0.0, -100.5, -1], 100, Lambertian([0.8, 0.8, 0.0])),
        Sphere([1.0, 0.0, -1.0], 0.5, Metal([0.8, 0.6, 0.2])),
        Sphere([-1.0, 0.0, -1], 0.5, Dielectric(1.5)),
        Sphere([-1.0, 0.0, -1], -0.45, Dielectric(1.5))
    ]

    for j::Int = ny - 1 : -1 : 0
        for i::Int = 0 : nx - 1

            col::Array{Float64} = [0.0, 0.0, 0.0]
            for s = 1:ns
                u::Float64 = (convert(Float64, i) + rand()) / nx
                v::Float64 = (convert(Float64, j) + rand()) / ny
                r = getRay(camera, u, v)
                col += color(r, world, 0)
            end

            col /= convert(Float64, ns)
            col = sqrt.(col)
                    
            ir::Int = trunc(255.99 * col[1])
            ig::Int = trunc(255.99 * col[2])
            ib::Int = trunc(255.99 * col[3])
            @printf("%d %d %d\n", ir, ig, ib)
        end
    end
end

main()
