using Printf

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

function boundingBox(bvh::BVHNode, t0::Float64, t1::Float64)::Union{AABB, Nothing}
    return bvh.box
end
