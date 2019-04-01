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

function ⊗(a::Vec3, b::Vec3)::Vec3
    return a .* b
end

# send rays only to light source
# on_light = [213.0 + rand() * (343.0 - 213.0), 554.0, 227.0 + rand() * (332.0 - 227)]
# to_light = on_light - hitres.p
# d2 = (norm(to_light)) ^ 2
# to_light = normalize(to_light)
# if (dot(to_light, hitres.normal) < 0.0)
#     return emission
# end
# light_area = (343.0 - 213.0) * (332 - 227.0)
# lcos = abs(to_light[2])
# if (lcos < 0.000001)
#     return emission
# end

# pdf = d2 / (lcos * light_area)
# sray = Ray(hitres.p, to_light, r.time)

function color(r::Ray, world::Hitable, depth::Int64)::Array{Float64}
    hitres = hit(world, r, 0.001, typemax(Float64))
    if (isa(hitres, HitRecord))
        emission = emitted(hitres.material, r, hitres, hitres.u, hitres.v, hitres.p)
        if (depth < 50)
            scatterRes = scatter(hitres.material, r, hitres)
            if (isa(scatterRes, ScatterRecord))
                light_shape = XZRect(213.0, 343.0, 227.0, 332.0, 554.0,
                                     DiffuseLight(ConstantTexture([7.0, 7.0, 7.0])))

                p1 = HitablePDF(light_shape, hitres.p)
                p2 = CosinePDF(hitres.normal)
                pdf = MixturePDF(p1, p2)

                scattered = Ray(hitres.p, generate(pdf), r.time)
                pdf_val = value(pdf, scattered.direction)
                return emission + scatterRes.albedo ⊗
                    color(scattered, world, depth + 1) *
                    scatteringPDF(hitres.material, r, hitres, scattered) / pdf_val
            end
        end
        return emission
    else
        return [0.0, 0.0, 0.0]
    end
end




