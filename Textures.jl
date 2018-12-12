using Random
using LinearAlgebra
using Images
using Colors
using FileIO

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
