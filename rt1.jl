using Printf
using LinearAlgebra

struct Ray
    origin::Array{Float64,1}
    direction::Array{Float64,1}
end

function point_at_parameter(r::Ray, t::Float64)
    return r.origin + t * r.direction
end

function color(r::Ray)
    u = r.direction / norm(r.direction)
    t = 0.5 * (u[2] + 1.0)
    return (1.0 - t) * [1.0, 1.0, 1.0] + t * [0.5, 0.7, 1.0]
end

function main()
    nx::Int = 200;
    ny::Int = 100;
    @printf("P3\n%d %d\n255\n", nx, ny);

    lower_left_corner = [-2.0, -1.0, -1.0]
    horizontal = [4.0, 0.0, 0.0]
    vertical = [0.0, 2.0, 0.0]
    origin = [0.0, 0.0, 0.0]

    for j::Int = ny - 1 : -1 : 0
        for i::Int = 0 : nx - 1
            u::Float64 = convert(Float64, i) / nx
            v::Float64 = convert(Float64, j) / ny

            r = Ray(origin, lower_left_corner + u * horizontal + v * vertical)
            col = color(r)
                    
            ir::Int = trunc(255.99 * col[1])
            ig::Int = trunc(255.99 * col[2])
            ib::Int = trunc(255.99 * col[3])
            @printf("%d %d %d\n", ir, ig, ib)
        end
    end
end

main()

