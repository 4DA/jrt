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

struct FlipNormals <: Hitable
    ptr::Hitable
end

function boundingBox(fn::FlipNormals, t0::Float64, t1::Float64)::Union{AABB, Nothing}
    return boundingBox(fn.ptr, t0, t1)
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
