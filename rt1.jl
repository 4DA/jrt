using Printf
using LinearAlgebra
using Base

const Vec3 = Array{Float64, 1}

abstract type Hitable end
abstract type Material end
abstract type Texture end

abstract type AbstractSphere <: Hitable end

struct Sphere <: AbstractSphere
    center::Vec3
    radius::Float64
    material::Material
end

struct AABB
    min::Vec3
    max::Vec3
end

struct HitableList <: Hitable
    array::Array{Hitable}
end

struct ConstantTexture <: Texture
    color::Vec3
end

function value(texture::ConstantTexture, u::Float64, v::Float64, p::Vec3)
    return texture.color
end

struct CheckerTexture <: Texture
    odd::Texture
    even::Texture
end

function value(texture::CheckerTexture, u::Float64, v::Float64, p::Vec3)
    sines = sin(10.0 * p[1]) * sin(10.0 * p[2]) * sin(10.0 * p[3])

    if (sines < 0.0)
        return value(texture.odd, u, v, p)
    else
        return value(texture.even, u, v, p)
    end
end


struct BVHNode <: Hitable
    left::Hitable
    right::Hitable
    box::AABB

    function BVHNode(lst::Array{Hitable}, time0::Float64, time1::Float64)
        sz = length(lst)
        axis = convert(Int64, floor(3.0 * rand()))

        if axis == 0
            sort!(lst; lt = boxXCompare)
        elseif axis == 1
            sort!(lst; lt = boxYCompare)
        elseif axis == 2
            sort!(lst; lt = boxZCompare)
        end

        if sz == 1
            left = lst[1]
            right = lst[1]
        elseif sz == 2
            left = lst[1]
            right = lst[2]
        else
            left = BVHNode(lst[1: div(sz, 2)], time0, time1)
            right = BVHNode(lst[div(sz, 2) + 1: sz], time0, time1)
        end

        box_left = boundingBox(left, time0, time1)
        box_right = boundingBox(right, time0, time1)
        box = surroundingBox(box_left, box_right)

        return new(left, right, box)
    end
end

struct MovingSphere <: AbstractSphere
    center0::Vec3
    center1::Vec3
    time0::Float64
    time1::Float64
    radius::Float64
    material::Material
end

function boxXCompare(x::Hitable, y::Hitable)::Bool
    box_left = boundingBox(x, 0.0, 0.0)
    box_right = boundingBox(y, 0.0, 0.0)

    if !(isa(box_left, AABB)) || !(isa(box_right, AABB))
        @printf(Base.fdio(2), "no bounding box in boxXCompare")
    end

    return box_left.min[1] < box_right.min[1]
end

function boxYCompare(x::Hitable, y::Hitable)::Bool
    box_left = boundingBox(x, 0.0, 0.0)
    box_right = boundingBox(y, 0.0, 0.0)

    if !(isa(box_left, AABB)) || !(isa(box_right, AABB))
        @printf(Base.fdio(2), "no bounding box in boxYCompare")
    end

    return box_left.min[2] < box_right.min[2]
end

function boxZCompare(x::Hitable, y::Hitable)::Bool
    box_left = boundingBox(x, 0.0, 0.0)
    box_right = boundingBox(y, 0.0, 0.0)

    if !(isa(box_left, AABB)) || !(isa(box_right, AABB))
        @printf(Base.fdio(2), "no bounding box in boxZCompare")
    end

    return box_left.min[3] < box_right.min[3]
end


function center(s::Sphere, ::Float64)::Vec3
    return s.center
end

function center(s::MovingSphere, time::Float64)::Vec3
    # if (time > 0.5)
    #     @printf(Base.fdio(2), "time = %f, t1 = %f, t2 = %f\n", time, s.time0, s.time1)
    # end
    c = s.center0 + ((time - s.time0) / (s.time1 - s.time0)) * (s.center1 - s.center0)
    # @printf(Base.fdio(2), "c = <%f,%f,%f>\n", c[1],c[2],c[3])
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
end

struct ScatterRecord
    ray::Ray
    attenuation::Vec3
end

struct Camera
    origin::Vec3
    lower_left_corner::Vec3
    horizontal::Vec3
    vertical::Vec3
    u::Vec3
    v::Vec3
    w::Vec3
    lens_radius::Float64
    time0::Float64
    time1::Float64

    function Camera(lookFrom::Vec3, lookAt::Vec3, vUp::Vec3, vfov::Float64, aspect::Float64,
                    aperture::Float64, focus_dist::Float64,
                    t0::Float64, t1::Float64)
        lens_radius = aperture / 2.0
        theta = vfov * pi / 180.0
        half_height = tan(theta / 2.0)
        half_width = aspect * half_height
        origin = lookFrom
        w = normalize(lookFrom - lookAt)
        u = normalize(cross(vUp, w))
        v = cross(w, u)
        lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w
        horizontal = 2 * half_width * focus_dist * u
        vertical = 2 * half_height * focus_dist * v
        return new(origin, lower_left_corner, horizontal, vertical, u, v, w, lens_radius, t0, t1)
    end
end

struct Lambertian <: Material
    albedo::Texture
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

function random_in_unit_disk()
    p::Vec3 = [0.0, 0.0, 0.0]
    while true
        p = 2.0 * [rand(), rand(), 0.0] - [1.0, 1.0, 0.0]
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
    return ScatterRecord(scattered, value(m.albedo, 0.0, 0.0, hit.p))
end

function scatter(m::Metal, r_in::Ray, hit::HitRecord)::Union{ScatterRecord, Nothing}
    reflected = reflect(r_in.direction / norm(r_in.direction), hit.normal)
    scattered = Ray(hit.p, reflected + m.fuzz * random_in_unit_sphere(), r_in.time)
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


function getRay(c::Camera, s::Float64, t::Float64)::Ray
    rd = c.lens_radius * random_in_unit_disk()
    offset = c.u * rd[1] + c.v * rd[2]
    return Ray(c.origin + offset,
               c.lower_left_corner + s * c.horizontal + t * c.vertical - c.origin - offset,
               c.time0 + rand() * (c.time1 - c.time0))
end


function point_at_parameter(r::Ray, t::Float64)
    return r.origin + t * r.direction
end

function boundingBox(sphere::Sphere, t0::Float64, t1::Float64)::Union{AABB, Nothing}
    return AABB(sphere.center - [radius, radius, radius], sphere.center + [radius, radius, radius])
end

function surroundingBox(box0::AABB, box1::AABB)::AABB
    small = min.(box0.min, box1.min)
    big = max.(box0.max, box1.max)
    return AABB(small, big)
end

function boundingBox(s::Sphere, t0::Float64, t1::Float64)::Union{AABB, Nothing}
    return AABB(s.center - [s.radius, s.radius, s.radius], s.center + [s.radius, s.radius, s.radius])
end

function boundingBox(s::MovingSphere, t0::Float64, t1::Float64)::Union{AABB, Nothing}
    box0 = AABB(center(s, t0) - [s.radius, s.radius, s.radius],
                center(s, t0) + [s.radius, s.radius, s.radius])

    box1 = AABB(center(s, t1) - [s.radius, s.radius, s.radius],
                center(s, t1) + [s.radius, s.radius, s.radius])

    return surroundingBox(box0, box1)
end

function boundingBox(bvh::BVHNode, t0::Float64, t1::Float64)::Union{AABB, Nothing}
    return bvh.box
end

function boundingBox(lst::HitableList, t0::Float64, t1::Float64)::Union{AABB, Nothing}
    if (size(lst) < 1)
        return nothing
    end

    box::Union{AABB, Nothing} = nothing

    box = boundingBox(list[1], t0, t1)

    if !isa(temp_box, AABB)
        return nothing
    end

    for i = 2:size(lst)
        temp_box = boundingBox(lst[i], t0, t1)
        if (isa(temp_box, AABB))
            box = surroundingBox(box, temp_box)
        else
            return nothing
        end
    end

    return box
end

function hit(aabb::AABB, r::Ray, tmin::Float64, tmax::Float64)::Bool
    for a = 1:3
        invD = 1.0 / r.direction[a]
        t0 = (aabb.min[a] - r.origin[a]) * invD
        t1 = (aabb.max[a] - r.origin[a]) * invD

        if (invD < 0.0)
            t0, t1 = t1, t0
        end
        
        tmin = t0 > tmin ? t0 : tmin
        tmax = t1 < tmax ? t1 : tmax

        if (tmax <= tmin)
            return false
        end
    end
    return true
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


function hit(sphere::MovingSphere, r::Ray, t_min::Float64, t_max::Float64)::Union{HitRecord, Nothing}
    a = dot(r.direction, r.direction)
    s2o = r.origin - center(sphere, r.time)
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
                             (p - center(sphere, r.time)) / sphere.radius,
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
                             (p - center(sphere, r.time)) / sphere.radius,
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

function print(s::Sphere, str::String)
    @printf(Base.fdio(2),
            "%s: sphere c = <%.1f, %.1f, %.1f>, r = %.1f\n",
            str,
            s.center[1], s.center[2], s.center[3],
            s.radius)
end

function print(box::AABB, str::String)
    @printf(Base.fdio(2),
            "%s: <%.1f, %.1f, %.1f>-<%.1f, %.1f, %.1f>\n",
            str,
            box.min[1], box.min[2], box.min[3],
            box.max[1], box.max[2], box.max[3])
end

function print(mvs::MovingSphere, str::String)
    c1 = center(mvs, 0.0)
    c2 = center(mvs, 1.0)

    @printf(Base.fdio(2),
            "%s: msphere c1 = <%.1f, %.1f, %.1f>, c2 = <%.1f, %.1f, %.1f>, r = %.1f\n",
            str,
            c1[1], c1[2], c1[3],
            c2[1], c2[2], c2[3],
            mvs.radius)
end


function hit(node::BVHNode, r::Ray, t_min::Float64, t_max::Float64)::Union{HitRecord, Nothing}
    if hit(node.box, r, t_min, t_max)
        left_rec = hit(node.left, r, t_min, t_max)
        right_rec = hit(node.right, r, t_min, t_max)

        if isa(left_rec, HitRecord) && isa(right_rec, HitRecord)
            # @printf(Base.fdio(2), "hit both\n")
            if left_rec.t < right_rec.t
                return left_rec
            else
                return right_rec
            end
        elseif isa(left_rec, HitRecord)
            return left_rec
        elseif isa(right_rec, HitRecord)
            return right_rec
        else
            return nothing
        end
    else
        return nothing
    end
end

function color(r::Ray, world::BVHNode, depth::Int64)::Array{Float64}
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

function two_spheres()::BVHNode
    checker = CheckerTexture(ConstantTexture([0.2, 0.3, 0.1]), ConstantTexture([0.9, 0.9, 0.9]))
    list::Array{Hitable} = []
    push!(list, Sphere([0.0, -10.0, 0.0], 10.0, Lambertian(checker)))
    push!(list, Sphere([0.0, 10.0, 0.0], 10.0, Lambertian(checker)))

    return BVHNode(list, 0.0, 1.0)
end

function random_scene()::BVHNode
    list::Array{Hitable} = []
    checker = CheckerTexture(ConstantTexture([0.2, 0.3, 0.1]),
                             ConstantTexture([0.9, 0.9, 0.9]))

    # push!(list, Sphere([0.0, -1000.0, 0.0], 1000.0, Lambertian(ConstantTexture([0.5, 0.5, 0.5]))))

    push!(list, Sphere([0.0, -1000.0, 0.0], 1000.0, Lambertian(checker)))

    for a = -11 : 10
        for b = -11 : 10
            choose_mat = rand()
            center = [a + 0.9 * rand(), 0.2, b + 0.9 * rand()]

            if (norm(center - [4.0, 0.2, 0.0]) > 0.9)
                if (choose_mat < 0.8) # diffuse
                    push!(list, MovingSphere(center, center + [0.0, 0.5 * rand(), 0.0], 
                                             0.0, 1.0,
                                             0.2,
                                       Lambertian(ConstantTexture([rand()^2, rand()^2, rand()^2]))))
                elseif (choose_mat < 0.95) #metal
                    push!(list, Sphere(center, 0.2,
                                       Metal([0.5 * (1 + rand()),0.5 * (1 + rand()),0.5 * (1 + rand())])))
                else #glass
                    push!(list, Sphere(center, 0.2, Dielectric(1.5)))
                end
            end
        end
    end

    push!(list, Sphere([0.0, 1.0, 0.0], 1.0, Dielectric(1.5)))
    push!(list, Sphere([-4.0, 1.0, 0.0], 1.0,  Lambertian(ConstantTexture([0.4, 0.2, 0.1]))))
    push!(list, Sphere([4.0, 1.0, 0.0], 1.0,  Metal([0.7, 0.6, 0.5])))

    return BVHNode(list, 0.0, 1.0)
end

function main()
    nx::Int = 600;
    ny::Int = 400;
    ns::Int = 40;
    @printf("P3\n%d %d\n255\n", nx, ny);

    lookFrom = [13.0, 2.0, 3.0]
    lookAt = [0.0, 0.0, 0.0]
    aperture = 0.0
    dist_to_focus = 10.0
    camera = Camera(lookFrom, lookAt , [0.0, 1.0, 0.0], 15.0,
                    convert(Float64, nx) / convert(Float64, ny), aperture, dist_to_focus,
                    0.0, 1.0)

    R = cos(pi / 4)
    
    hitables::Array{Hitable} = [
        Sphere([0.0, 0.0, -1.0], 0.5, Lambertian(ConstantTexture([0.1, 0.2, 0.5]))),
        Sphere([0.0, -100.5, -1], 100, Lambertian(ConstantTexture([0.8, 0.8, 0.0]))),
        Sphere([1.0, 0.0, -1.0], 0.5, Metal([0.8, 0.6, 0.2])),
        Sphere([-1.0, 0.0, -1], 0.5, Dielectric(1.5)),
        Sphere([-1.0, 0.0, -1], -0.45, Dielectric(1.5)),

        # Sphere([-R, 0.0, -1], R, Lambertian(ConstantTexture([0.0, 0.0, 1.0]))),
        # Sphere([R, 0.0, -1], R, Lambertian(ConstantTexture([1.0, 0.0, 0.0]))),
    ]

    world::BVHNode = random_scene()

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

        if (j % (ny / 10) == 0)
        @printf(Base.fdio(2), "progress: %f\n", 1.0 - convert(Float64, j) / ny)
        end
    end
end

main()
