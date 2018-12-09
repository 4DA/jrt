using Printf
using LinearAlgebra
using Base
using Random
using Images
using ColorTypes
using FileIO
using Colors

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

struct DiffuseLight <: Material
    emit::Texture
end

function emitted(light::DiffuseLight, u::Float64, v::Float64, p::Vec3)::Vec3
    return value(light.emit, u, v, p)
end

struct Translate <: Hitable
    ptr::Hitable
    offset::Vec3
end

struct RotateY <: Hitable
    sin_t::Float64
    cos_t::Float64
    box::AABB
    ptr::Hitable

    function RotateY(p::Hitable, angle::Float64)
        radians = pi / 180.0 * angle
        sin_t = sin(radians)
        cos_t = cos(radians)
        box = boundingBox(p, 0.0, 1.0)
        tmax = typemax(Float64)
        tmin = typemin(Float64)
        min = [tmax, tmax, tmax]
        max = [tmin, tmin, tmin]

        for i = 0.0:1.0:1.0
            for j = 0.0:1.0:1.0
                for k = 0.0:1.0:1.0
                    x = i * box.max[1] + (1.0 - i) * box.min[1]
                    y = j * box.max[2] + (1.0 - j) * box.min[2]
                    z = k * box.max[3] + (1.0 - k) * box.min[3]
                    newx = cos_t * x + sin_t * z
                    newz = -sin_t * x + cos_t * z

                    tester = [newx, y, newz]
                    for c = 1:3
                        if (tester[c] > max[c])
                            max[c] = tester[c]
                        end

                        if (tester[c] < min[c])
                            min[c] = tester[c]
                        end
                    end

                end
            end
        end

        aabb = AABB(min, max)

        return new(sin_t, cos_t, AABB(min, max), p)
    end
end

struct Isotropic <: Material
    albedo::Texture
end

struct ConstantMedium <: Hitable
    boundary::Hitable
    density::Float64
    phaseFunction::Material

    function ConstantMedium(boundary::Hitable, density::Float64, a::Texture)
        return new(boundary, density, Isotropic(a))
    end
end

function normalize(v::Vec3)::Vec3
    return v / norm(v)
end

function generatePermutation()::Array{Int64}
    p = Array{Float64}(undef, 256)
    for i = 1:256
        p[i] = i
    end

    shuffle!(p)
    return p
end

struct Perlin
    ranvec::Array{Vec3}
    perm_x::Array{Int64}
    perm_y::Array{Int64}
    perm_z::Array{Int64}

    function Perlin()
        ranvec = Array{Vec3, 1}(undef, 256)
        perm_x = generatePermutation()
        perm_y = generatePermutation()
        perm_z = generatePermutation()

        for i = 1:256
            ranvec[i] = normalize([-1.0, -1.0, -1.0] + 2 * [rand(), rand(), rand()])
        end

        return new(ranvec, perm_x, perm_y, perm_z)
    end
end

g_Perlin = Perlin()

struct ConstantTexture <: Texture
    color::Vec3
end


struct CheckerTexture <: Texture
    odd::Texture
    even::Texture
end

struct NoiseTexture <: Texture
    noise::Perlin
    scale::Float64

    function NoiseTexture()
        return new(g_Perlin, 5)
    end

    function NoiseTexture(scale)
        return new(g_Perlin, scale)
    end
end

struct ImageTexture <: Texture
    data::Array{RGB{N0f8},2}
    nx::Int64
    ny::Int64

    function ImageTexture(data::Array{RGB{N0f8},2})
        return new(data, size(data)[1], size(data)[2])
    end

    function ImageTexture(data::Array{RGBA{N0f8},2})
        return new(convert.(RGB, data), size(data)[1], size(data)[2])
    end
end

function value(texture::ConstantTexture, u::Float64, v::Float64, p::Vec3)
    return texture.color
end

function value(texture::CheckerTexture, u::Float64, v::Float64, p::Vec3)::Vec3
    sines = sin(10.0 * p[1]) * sin(10.0 * p[2]) * sin(10.0 * p[3])

    if (sines < 0.0)
        return value(texture.odd, u, v, p)
    else
        return value(texture.even, u, v, p)
    end
end

function value(texture::NoiseTexture, u::Float64, v::Float64, p::Vec3)::Vec3
    return [1.0, 1.0, 1.0] * 0.5 *
        (1.0 + sin(texture.scale * p[3] + 10.0 * turb(texture.noise, p , 11)))
end

function value(texture::ImageTexture, u::Float64, v::Float64, p::Vec3)::Vec3
    i = convert(Int64, floor(u * texture.nx)) + 1
    j = convert(Int64, floor((1.0 - v) * texture.ny - 0.001)) + 1

    i = clamp(i, 1, texture.nx)
    j = clamp(j, 1, texture.ny)

    r = red(texture.data[i, j])
    g = green(texture.data[i, j])
    b = blue(texture.data[i, j])

    return convert.(Float64, [r, g, b])
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

struct XYRect <: Hitable
    x0::Float64
    x1::Float64
    y0::Float64
    y1::Float64
    k::Float64
    material::Material
end

struct XZRect <: Hitable
    x0::Float64
    x1::Float64
    z0::Float64
    z1::Float64
    k::Float64
    material::Material
end

struct YZRect <: Hitable
    y0::Float64
    y1::Float64
    z0::Float64
    z1::Float64
    k::Float64
    material::Material
end

struct FlipNormals <: Hitable
    ptr::Hitable
end

struct Box <: Hitable
    pmin::Vec3
    pmax::Vec3
    list::HitableList

    function Box(p0::Vec3, p1::Vec3, material::Material)
        list::Array{Hitable} = []

        push!(list,             XYRect(p0[1], p1[1], p0[2], p1[2], p1[3], material))
        push!(list, FlipNormals(XYRect(p0[1], p1[1], p0[2], p1[2], p0[3], material)))

        push!(list,             XZRect(p0[1], p1[1], p0[3], p1[3], p1[2], material))
        push!(list, FlipNormals(XZRect(p0[1], p1[1], p0[3], p1[3], p0[2], material)))

        push!(list,             YZRect(p0[2], p1[2], p0[3], p1[3], p1[1], material))
        push!(list, FlipNormals(YZRect(p0[2], p1[2], p0[3], p1[3], p0[1], material)))

        return new(p0, p1, HitableList(list))
    end
end

function noise_f(t::Float64)::Float64
    t = abs(t)

    if t < 1.0
        return 1.0 - ( 3.0 - 2.0 * t ) * t * t;
    else
        return 0.0
    end
end

function surflet(p::Vec3, grad::Vec3)
    return noise_f(p[1]) * noise_f(p[2]) * noise_f(p[3]) * dot(p, grad)
end

function turb(n::Perlin, p::Vec3, depth::Int64)::Float64
    accum = 0.0
    weight = 1.0
    temp_p = p

    for i = 1:depth
        accum += weight * noise(n, temp_p)
        weight *= 0.5
        temp_p *= 2
    end

    return abs(accum)
end

function noise(n::Perlin, p::Vec3)::Float64
    result = 0.0
    cell = convert.(Int64, floor.(p))

    c = Array{Vec3, 3}(undef, 2, 2, 2)

    for grid_z = cell[3]:(cell[3]+1)
        for grid_y = cell[2]:(cell[2]+1)
            for grid_x = cell[1]:(cell[1]+1)
                hash = (n.perm_x[mod1(grid_x,256)] - 1) ⊻ (n.perm_y[mod1(grid_y,256)] - 1) ⊻
                    (n.perm_z[mod1(grid_z, 256)] - 1) + 1

                result += surflet(p - [grid_x, grid_y, grid_z], n.ranvec[hash])
            end
        end
    end

    return (result + 1.0) / 2.0
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
    u::Float64
    v::Float64
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

function emitted(m::Lambertian, u::Float64, v::Float64, p::Vec3)::Vec3
    return [0.0, 0.0, 0.0]
end

struct Metal <: Material
    albedo::Vec3
    fuzz::Float64
    Metal(a, f) = new(a, clamp(f, 0.0, 1.0))
    Metal(a) = new(a, 0.0)
end

function emitted(m::Metal, u::Float64, v::Float64, p::Vec3)::Vec3
    return [0.0, 0.0, 0.0]
end

struct Dielectric <: Material
    ref_idx::Float64
end

function emitted(m::Dielectric, u::Float64, v::Float64, p::Vec3)::Vec3
    return [0.0, 0.0, 0.0]
end

function emitted(iso::Isotropic, u::Float64, v::Float64, p::Vec3)::Vec3
    return [0.0, 0.0, 0.0]
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
    return ScatterRecord(scattered, value(m.albedo, hit.u, hit.v, hit.p))
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

function scatter(l::DiffuseLight, r_in::Ray, hit::HitRecord)::Union{ScatterRecord, Nothing}
    return nothing
end

function scatter(iso::Isotropic, r_in::Ray, hit::HitRecord)::Union{ScatterRecord, Nothing}
    res = ScatterRecord(Ray(hit.p, random_in_unit_sphere()), value(iso.albedo, hit.u, hit.v, hit.p))
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

function boundingBox(rect::XYRect, t0::Float64, t1::Float64)::Union{AABB, Nothing}
    return AABB([rect.x0, rect.y0, rect.k - 0.0001], [rect.x1, rect.y1, rect.k + 0.0001])
end

function boundingBox(rect::XZRect, t0::Float64, t1::Float64)::Union{AABB, Nothing}
    return AABB([rect.x0, rect.k - 0.0001, rect.z0], [rect.x1, rect.k + 0.0001, rect.z1])
end

function boundingBox(rect::YZRect, t0::Float64, t1::Float64)::Union{AABB, Nothing}
    return AABB([rect.k - 0.0001, rect.y0, rect.z0], [rect.k + 0.0001, rect.y1, rect.z1])
end

function boundingBox(fn::FlipNormals, t0::Float64, t1::Float64)::Union{AABB, Nothing}
    return boundingBox(fn.ptr, t0, t1)
end

function boundingBox(box::Box, t0::Float64, t1::Float64)::Union{AABB, Nothing}
    return AABB(box.pmin, box.pmax)
end

function boundingBox(t::Translate, t0::Float64, t1::Float64)::Union{AABB, Nothing}
    box = boundingBox(t.ptr, t0, t1)
    if (isa(box, AABB))
        return AABB(box.min + t.offset, box.max + t.offset)
    else
        return nothing
    end
end
    
function boundingBox(ty::RotateY, t0::Float64, t1::Float64)::Union{AABB, Nothing}
    return ty.box
end

function boundingBox(medium::ConstantMedium, t0::Float64, t1::Float64)::Union{AABB, Nothing}
    return boundingBox(medium.boundary, t0, t1)
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

function get_sphere_uv(p::Vec3)::Tuple{Float64, Float64}
    phi = atan(p[3], p[1])
    theta = asin(p[2])
    u = 1.0 - (phi + pi) / (2.0 * pi)
    v = (theta + pi / 2.0) / pi
    return (u,v)
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
            u, v = get_sphere_uv((p - sphere.center) / sphere.radius)
            return HitRecord(t1,
                              p,
                             (p - sphere.center) / sphere.radius,
                             sphere.material, u, v)
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
            u, v = get_sphere_uv((p - sphere.center) / sphere.radius)
            return HitRecord(t2,
                              p, 
                             (p - sphere.center) / sphere.radius,
                             sphere.material, u, v)
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
            u, v = get_sphere_uv((p - center(sphere, r.time)) / sphere.radius)
            return HitRecord(t1,
                              p,
                             (p - center(sphere, r.time)) / sphere.radius,
                             sphere.material, u, v)
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
            u, v = get_sphere_uv((p - center(sphere, r.time)) / sphere.radius)
            return HitRecord(t2,
                              p, 
                             (p - center(sphere, r.time)) / sphere.radius,
                             sphere.material, u, v)
        end
    end
    return nothing
end


function hit(rect::XYRect, r::Ray, t0::Float64, t1::Float64)::Union{HitRecord, Nothing}
    t = (rect.k - r.origin[3]) / r.direction[3]
    
    if (t < t0 || t > t1)
        return nothing
    end

    x = r.origin[1] + t * r.direction[1]
    y = r.origin[2] + t * r.direction[2]

    if (x < rect.x0 || x > rect.x1 || y < rect.y0 || y > rect.y1)
        return nothing
    end

    return HitRecord(t, point_at_parameter(r, t), [0.0, 0.0, 1.0], rect.material,
                     (x - rect.x0) / (rect.x1 - rect.x0), # u
                     (y - rect.y0) / (rect.y1 - rect.y0), # v
                     )
    
end

function hit(rect::XZRect, r::Ray, t0::Float64, t1::Float64)::Union{HitRecord, Nothing}
    t = (rect.k - r.origin[2]) / r.direction[2]
    
    if (t < t0 || t > t1)
        return nothing
    end

    x = r.origin[1] + t * r.direction[1]
    z = r.origin[3] + t * r.direction[3]

    if (x < rect.x0 || x > rect.x1 || z < rect.z0 || z > rect.z1)
        return nothing
    end

    return HitRecord(t, point_at_parameter(r, t), [0.0, 1.0, 0.0], rect.material,
                     (x - rect.x0) / (rect.x1 - rect.x0), # u
                     (z - rect.z0) / (rect.z1 - rect.z0), # v
                     )
    
end

function hit(rect::YZRect, r::Ray, t0::Float64, t1::Float64)::Union{HitRecord, Nothing}
    t = (rect.k - r.origin[1]) / r.direction[1]
    
    if (t < t0 || t > t1)
        return nothing
    end

    y = r.origin[2] + t * r.direction[2]
    z = r.origin[3] + t * r.direction[3]

    if (y < rect.y0 || y > rect.y1 || z < rect.z0 || z > rect.z1)
        return nothing
    end

    return HitRecord(t, point_at_parameter(r, t), [1.0, 0.0, 0.0], rect.material,
                     (y - rect.y0) / (rect.y1 - rect.y0), # u
                     (z - rect.z0) / (rect.z1 - rect.z0), # v
                     )
    
end

function hit(fn::FlipNormals, r::Ray, t0::Float64, t1::Float64)::Union{HitRecord, Nothing}
    rec = hit(fn.ptr, r, t0, t1)
    if (isa(rec, HitRecord))
        return HitRecord(rec.t, rec.p, -rec.normal, rec.material, rec.u, rec.v)
    end

    return nothing
end

function hit(box::Box, r::Ray, t0::Float64, t1::Float64)::Union{HitRecord, Nothing}
    return hit(box.list, r, t0, t1)
end

function hit(hitables::HitableList, r::Ray, t_min::Float64, t_max::Float64)::Union{HitRecord, Nothing}
    result::Union{HitRecord, Nothing} = nothing
    closest_t = t_max

    for i = 1:length(hitables.array)
        hit_res = hit(hitables.array[i], r, t_min, closest_t)
        if (isa(hit_res, HitRecord))
            closest_t = hit_res.t
            result = hit_res
        end
    end

    return result
end

function hit(t::Translate, r::Ray, t_min::Float64, t_max::Float64)::Union{HitRecord, Nothing}
    moved_r = Ray(r.origin - t.offset, r.direction, r.time)
    rec = hit(t.ptr, moved_r, t_min, t_max)
    if (isa(rec, HitRecord))
        return HitRecord(rec.t, rec.p + t.offset, rec.normal, rec.material, rec.u, rec.v)
    else
        return nothing
    end
end

function hit(ry::RotateY, r::Ray, t_min::Float64, t_max::Float64)::Union{HitRecord, Nothing}
    origin = copy(r.origin)
    direction = copy(r.direction)
    origin[1] = ry.cos_t * r.origin[1] - ry.sin_t * r.origin[3]
    origin[3] = ry.sin_t * r.origin[1] + ry.cos_t * r.origin[3]
    direction[1] = ry.cos_t * r.direction[1] - ry.sin_t * r.direction[3]
    direction[3] = ry.sin_t * r.direction[1] + ry.cos_t * r.direction[3]
    rotated_r = Ray(origin, direction, r.time)

    rec = hit(ry.ptr, rotated_r, t_min, t_max)

    if (isa(rec, HitRecord))
        p = rec.p
        normal = rec.normal
        p[1] = ry.cos_t * rec.p[1] + ry.sin_t * rec.p[3]
        p[3] = -ry.sin_t * rec.p[1] + ry.cos_t * rec.p[3]
        normal[1] = ry.cos_t * rec.normal[1] + ry.sin_t * rec.normal[3]
        normal[3] = -ry.sin_t * rec.normal[1] + ry.cos_t * rec.normal[3]
        return HitRecord(rec.t, p, normal, rec.material, rec.u, rec.v)
    end

    return nothing
end

function hit(medium::ConstantMedium, r::Ray, t_min::Float64, t_max::Float64)::Union{HitRecord, Nothing}
    res = HitRecord
    rec1 = hit(medium.boundary, r, typemin(Float64), typemax(Float64))

    db = rand() < 0.00001 ? true : false
    
    if (isa(rec1, HitRecord))
        rec2 = hit(medium.boundary, r, rec1.t + 0.0001, typemax(Float64))

        if (!isa(rec2, HitRecord))
            return nothing
        end

        t1 = clamp(rec1.t, t_min, typemax(Float64))
        t2 = clamp(rec2.t, typemin(Float64), t_max)

        if (t1  >= t2)
            return nothing
        end

        # if (db)
        #     @printf(Base.fdio(2), "t0: %f, t1: %f\n", rec1.t, rec2.t)
        # end

        t1 = clamp(t1, 0, typemax(Float64))

        distance_inside_boundary = (t2 - t1) * norm(r.direction)

        hit_distance = -(1.0 / medium.density) * log(rand())
        res_t = t1 + hit_distance / norm(r.direction)

        if (hit_distance < distance_inside_boundary)
            return HitRecord(res_t,
                             point_at_parameter(r, res_t),
                             [1.0, 0.0, 0.0], # arbitrary
                             medium.phaseFunction,
                             0.0,
                             0.0)
        end
    end
    return nothing
end

function printObj(s::Sphere, str::String)
    @printf(Base.fdio(2),
            "%s: sphere c = <%.1f, %.1f, %.1f>, r = %.1f\n",
            str,
            s.center[1], s.center[2], s.center[3],
            s.radius)
end

function printObj(box::AABB, str::String)
    @printf(Base.fdio(2),
            "%s: <%.1f, %.1f, %.1f>-<%.1f, %.1f, %.1f>\n",
            str,
            box.min[1], box.min[2], box.min[3],
            box.max[1], box.max[2], box.max[3])
end

function printObj(mvs::MovingSphere, str::String)
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
        emission = emitted(hitres.material, hitres.u, hitres.v, hitres.p)
        if (depth < 50)
            scatterRes = scatter(hitres.material, r, hitres)
            if (isa(scatterRes, ScatterRecord))
                return emission + scatterRes.attenuation .* color(scatterRes.ray, world, depth + 1)
            end
        end
        return emission
    else
        return [0.0, 0.0, 0.0]
    end
end

function two_spheres()::BVHNode
    checker = CheckerTexture(ConstantTexture([0.2, 0.3, 0.1]), ConstantTexture([0.9, 0.9, 0.9]))
    list::Array{Hitable} = []
    push!(list, Sphere([0.0, -10.0, 0.0], 10.0, Lambertian(checker)))
    push!(list, Sphere([0.0, 10.0, 0.0], 10.0, Lambertian(checker)))

    return BVHNode(list, 0.0, 1.0)
end

function sphere_textured()::BVHNode
    imgT = ImageTexture(load("earth.png"))
    list::Array{Hitable} = []
    push!(list, Sphere([0.0, 0.0, 0.0], 2.0, Lambertian(imgT)))

    return BVHNode(list, 0.0, 1.0)
end

function two_perlin_spheres()::BVHNode
    list::Array{Hitable} = []

    pertext = NoiseTexture()
    push!(list, Sphere([0.0, -1000.0, 0.0], 1000.0, Lambertian(pertext)))
    push!(list, Sphere([0.0, 2.0, 0.0], 2.0, Lambertian(pertext)))

    return BVHNode(list, 0.0, 1.0)
end

function simple_light()::BVHNode
    list::Array{Hitable} = []

    pertext = NoiseTexture()
    push!(list, Sphere([0.0, -1000.0, 0.0], 1000.0, Lambertian(pertext)))
    push!(list, Sphere([0.0, 2.0, 0.0], 2.0, Lambertian(pertext)))
    push!(list, Sphere([0.0, 7.0, 0.0], 2.0, DiffuseLight(ConstantTexture([4.0, 4.0, 4.0]))))
    push!(list, XYRect(3.0, 5.0, 1.0, 3.0, -2.0, DiffuseLight(ConstantTexture([4.0, 4.0, 4.0]))))

    return BVHNode(list, 0.0, 1.0)    
end

function cornell_smoke()::BVHNode
    list::Array{Hitable} = []

    red = Lambertian(ConstantTexture([0.65, 0.05, 0.05]))
    white = Lambertian(ConstantTexture([0.73, 0.73, 0.73]))
    green = Lambertian(ConstantTexture([0.12, 0.45, 0.15]))
    light = DiffuseLight(ConstantTexture([7.0, 7.0, 7.0]))
    
    push!(list, FlipNormals(YZRect(0.0, 555.0, 0.0, 555.0, 555.0, green)))
    push!(list, YZRect(0.0, 555.0, 0.0, 555.0, 0.0, red))
    push!(list, XZRect(113.0, 443.0, 127.0, 432.0, 554.0, light))
    push!(list, FlipNormals(XZRect(0.0, 555.0, 0.0, 555.0, 555.0, white)))
    push!(list, XZRect(0.0, 555.0, 0.0, 555.0, 0.0,  white))
    push!(list, FlipNormals(XYRect(0.0, 555.0, 0.0, 555.0, 555.0,  white)))

    b1 = Translate(RotateY(Box([0.0, 0.0, 0.0], [165.0, 165.0, 165.0], white), -18.0),
                   [130.0, 0.0, 65.0])

    b2 = Translate(RotateY(Box([0.0, 0.0, 0.0], [165.0, 330.0, 165.0], white), 15.0),
                   [265.0, 0.0, 295.0])

    push!(list, ConstantMedium(b1, 0.01, ConstantTexture([1.0, 1.0, 1.0])))
    push!(list, ConstantMedium(b2, 0.01, ConstantTexture([0.0, 0.0, 0.0])))

    return BVHNode(list, 0.0, 1.0)    
end

function final_scene()::BVHNode
    list::Array{Hitable} = []
    boxlist::Array{Hitable} = []
    boxlist2::Array{Hitable} = []

    white = Lambertian(ConstantTexture([0.73, 0.73, 0.73]))
    ground = Lambertian(ConstantTexture([0.48, 0.83, 0.53]))

    nb = 20
    
    for i = 0:nb-1
        for j = 0:nb-1
            w = 100
            x0 = convert(Float64, -1000 + i * w)
            z0 = convert(Float64, -1000 + j * w)
            y0 = 0.0
            x1 = x0 + w
            y1 = 100.0 * (rand() + 0.01)
            z1 = z0 + w
            push!(boxlist, Box([x0, y0, z0], [x1, y1, z1], ground))
        end
    end

    push!(list, BVHNode(boxlist, 0.0, 1.0))

    light = DiffuseLight(ConstantTexture([7.0, 7.0, 7.0]))
    push!(list, XZRect(123.0, 423.0, 147.0, 412.0, 554.0, light))
    center = [400.0, 400.0, 200.0]
    push!(list, MovingSphere(center, center + [30.0, 0.0, 0.0], 0.0, 1.0, 50.0,
                             Lambertian(ConstantTexture([0.7, 0.3, 0.1]))))

    push!(list, Sphere([260.0, 150.0, 45.0], 50.0, Dielectric(1.5)))
    push!(list, Sphere([0.0, 150.0, 145.0], 50.0, Metal([0.8, 0.8, 0.9], 10.0)))

    boundary = Sphere([360.0, 150.0, 145.0], 70.0, Dielectric(1.5))
    push!(list, boundary)
    push!(list, ConstantMedium(boundary, 0.2, ConstantTexture([0.2, 0.4, 0.9])))

    boundary = Sphere([0.0, 0.0, 0.0], 5000.0, Dielectric(1.5))
    push!(list, ConstantMedium(boundary, 0.0001, ConstantTexture([1.0, 1.0, 1.0])))

    imgT = ImageTexture(load("earth.png"))
    push!(list, Sphere([400.0, 200.0, 400.0], 100.0, Lambertian(imgT)))
    
    pertext = NoiseTexture(0.1)
    push!(list, Sphere([220.0, 280.0, 300.0], 80.0, Lambertian(pertext)))

    for i = 0:1000
        push!(boxlist2, Sphere([165.0 * rand(), 165.0 * rand(), 165.0 * rand()], 10.0, white))
    end

    push!(list, Translate(RotateY(BVHNode(boxlist2, 0.0, 1.0), 15.0), [-100.0, 270.0, 395]))

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
    nx::Int = 400;
    ny::Int = 400;
    ns::Int = 200;
    @printf("P3\n%d %d\n255\n", nx, ny);

    lookFrom = [478.0, 278.0, -600.0]
    lookAt = [278.0, 278.0, 0.0]
    aperture = 0.0
    dist_to_focus = 10.0
    vfov = 40.0
    camera = Camera(lookFrom, lookAt , [0.0, 1.0, 0.0], vfov,
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

    # world::BVHNode = random_scene()
    # world::BVHNode = two_spheres()
    # world::BVHNode = sphere_textured()
    # world::BVHNode = simple_light()
    # world::BVHNode = cornell_smoke()
    world::BVHNode = final_scene()

    for j::Int = ny - 1 : -1 : 0
        for i::Int = 0 : nx - 1

            col::Array{Float64} = [0.0, 0.0, 0.0]

            # if (!(j < ny / 4 && i > nx / 4))
            #     @printf("%d %d %d\n", 0, 0, 0)
            #     continue

            # end
            
            
            for s = 1:ns
                u::Float64 = (convert(Float64, i) + rand()) / nx
                v::Float64 = (convert(Float64, j) + rand()) / ny
                r = getRay(camera, u, v)
                col += color(r, world, 0)
            end

            col /= convert(Float64, ns)
            if (col[1] < 0.0 || col[2] < 0.0 || col[3] < 0.0)
                @printf(Base.fdio(2), "negative color: <%f, %f, %f>, exiting \n",
                        col[1], col[2], col[3])
                return
            end
            col = sqrt.(col)
            col = clamp.(col, 0.0, 1.0)
                    
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

# using Profile
# using ProfileView
# ProfileView.view()

