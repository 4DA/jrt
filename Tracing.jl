using LinearAlgebra

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




