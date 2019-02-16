using Printf

abstract type AbstractSphere <: Hitable end

struct Sphere <: AbstractSphere
    center::Vec3
    radius::Float64
    material::Material
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

struct ConstantMedium <: Hitable
    boundary::Hitable
    density::Float64
    phaseFunction::Material

    function ConstantMedium(boundary::Hitable, density::Float64, a::Texture)
        return new(boundary, density, Isotropic(a))
    end
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



struct HitableList <: Hitable
    array::Array{Hitable}
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

function boundingBox(box::Box, t0::Float64, t1::Float64)::Union{AABB, Nothing}
    return AABB(box.pmin, box.pmax)
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

function hit(ry::RotateY, ray::Ray, t_min::Float64, t_max::Float64)::Union{HitRecord, Nothing}
    origin = copy(ray.origin)
    direction = copy(ray.direction)
    origin[1] = ry.cos_t * ray.origin[1] - ry.sin_t * ray.origin[3]
    origin[3] = ry.sin_t * ray.origin[1] + ry.cos_t * ray.origin[3]
    direction[1] = ry.cos_t * ray.direction[1] - ry.sin_t * ray.direction[3]
    direction[3] = ry.sin_t * ray.direction[1] + ry.cos_t * ray.direction[3]
    rotated_r = Ray(origin, direction, ray.time)

    rec = hit(ry.ptr, rotated_r, t_min, t_max)

    if (isa(rec, HitRecord))
        p = copy(rec.p)
        normal = copy(rec.normal)
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

