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

# ---------

struct HitablePDF <: PDF
    ptr::Hitable
    o::Vec3
end

function value(pdf::HitablePDF, direction::Vec3)::Float64
    return pdf_value(pdf.ptr, pdf.o, direction)
end

function generate(pdf::HitablePDF)::Vec3
    return random(pdf.ptr, pdf.o)
end

# hitable pdf support code

function pdf_value(h::XZRect, o::Vec3, v::Vec3)::Float64
    rec = hit(h, Ray(o, v), 0.001, typemax(Float64))
    if (isa(rec, HitRecord))
        area = (h.x1 - h.x0) * (h.z1 - h.z0)
        distance_squared = rec.t^2 * norm(v)^2
        cosine = abs(dot(v, rec.normal)) / norm(v)
        return distance_squared / (cosine * area)
    else
        return 0.0
    end
end

function random(h::XZRect, o::Vec3)::Vec3
    random_point = [h.x0 + rand() * (h.x1 - h.x0), h.k, h.z0 + rand() * (h.z1 - h.z0)]
    return random_point - o
end

# default implementations for Hitable
function pdf_value(h::Hitable, o::Vec3, v::Vec3)::Float64
    return 0.0
end

function random(h::Hitable, o::Vec3)::Vec3
    return [1.0, 0.0, 0.0]
end


# -- mixture pdf
struct MixturePDF
    p1::PDF
    p2::PDF
end

function value(pdf::MixturePDF, direction::Vec3)::Float64
    return 0.5 * value(pdf.p1, direction) + 0.5 * value(pdf.p2, direction)
end

function generate(pdf::MixturePDF)::Vec3
    if (rand() < 0.5)
        return generate(pdf.p1)
    else
        return generate(pdf.p2)
    end
end
