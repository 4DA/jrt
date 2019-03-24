struct DiffuseLight <: Material
    emit::Texture
end

function emitted(light::DiffuseLight, u::Float64, v::Float64, p::Vec3)::Vec3
    return value(light.emit, u, v, p)
end

struct Isotropic <: Material
    albedo::Texture
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

function emitted(m::Lambertian, u::Float64, v::Float64, p::Vec3)::Vec3
    return [0.0, 0.0, 0.0]
end


function emitted(m::Metal, u::Float64, v::Float64, p::Vec3)::Vec3
    return [0.0, 0.0, 0.0]
end


function emitted(m::Dielectric, u::Float64, v::Float64, p::Vec3)::Vec3
    return [0.0, 0.0, 0.0]
end

function emitted(iso::Isotropic, u::Float64, v::Float64, p::Vec3)::Vec3
    return [0.0, 0.0, 0.0]
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

function scatteringPDF(m::Lambertian, r_in::Ray, hit::HitRecord, scattered::Ray)::Float64
    cosine = dot(hit.normal, normalize(scattered.direction))
    cosine = clamp(cosine, 0.0, 1.0)
    return cosine / pi
end

function scatter(m::Lambertian, r_in::Ray, hit::HitRecord)::Union{ScatterRecord, Nothing}
    uvw = onb(hit.normal)
    direction = to_local(uvw, random_cosine_direction())

    scattered = Ray(hit.p, normalize(direction), r_in.time)
    pdf = dot(hit.normal, normalize(scattered.direction)) / pi

    pdf = clamp(pdf, 0.0, 1.0)
    return ScatterRecord(scattered, value(m.albedo, hit.u, hit.v, hit.p), pdf)
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
