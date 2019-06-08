using Printf
using Base
using Random
using FileIO
using ImageView

include("CoreTypes.jl")
include("Utils.jl")
include("BVH.jl")
include("Tracing.jl")
include("Transforms.jl")
include("Hitables.jl")
include("pdf.jl")
include("Materials.jl")
include("Textures.jl")

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

function cornell_box()::Hitable
    list::Array{Hitable} = []

    red = Lambertian(ConstantTexture([0.65, 0.05, 0.05]))
    white = Lambertian(ConstantTexture([0.73, 0.73, 0.73]))
    green = Lambertian(ConstantTexture([0.12, 0.45, 0.15]))
    light = DiffuseLight(ConstantTexture([15.0, 15.0, 15.0]))
    
    push!(list, FlipNormals(YZRect(0.0, 555.0, 0.0, 555.0, 555.0, green)))
    push!(list, YZRect(0.0, 555.0, 0.0, 555.0, 0.0, red))
    push!(list, FlipNormals(XZRect(213.0, 343.0, 227.0, 332.0, 554.0, light)))
    push!(list, FlipNormals(XZRect(0.0, 555.0, 0.0, 555.0, 555.0, white)))
    push!(list, XZRect(0.0, 555.0, 0.0, 555.0, 0.0,  white))
    push!(list, FlipNormals(XYRect(0.0, 555.0, 0.0, 555.0, 555.0,  white)))

    glass_sphere = Sphere([190.0, 90.0, 190.0], 90.0, Dielectric(1.5))

    push!(list, glass_sphere)

    # b1 = Translate(RotateY(Box([0.0, 0.0, 0.0], [165.0, 165.0, 165.0], white), -18.0),
    #                [130.0, 0.0, 65.0])
    # push!(list, b1)
    # aluminum = Metal([0.8, 0.85, 0.88], 0.0)

    b2 = Translate(RotateY(Box([0.0, 0.0, 0.0], [165.0, 330.0, 165.0], white), 15.0),
                   [265.0, 0.0, 295.0])
    push!(list, b2)

    return HitableList(list)
end

function test_cosine()
    N = 1000000
    sum = 0.0

    for i = 1:N
        v = random_cosine_direction()
        sum += v[3]^3 / (v[3] / pi)
    end

    @printf("estimate: %f", sum / N)
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

function remove_nan(v::Vec3)::Vec3
    res = [0.0, 0.0, 0.0]
    if (!isnan(v[1]))
        res[1] = v[1]
    end
    if (!isnan(v[2]))
        res[2] = v[2]
    end
    if (!isnan(v[3]))
        res[3] = v[3]
    end

    return res
end

function main_ppm(nx::Int, ny::Int, ns::Int)
    @printf("P3\n%d %d\n255\n", nx, ny);

    lookFrom = [278.0, 278.0, -800.0]
    lookAt = [278.0, 278.0, 0.0]
    aperture = 0.0
    dist_to_focus = 10.0
    vfov = 40.0
    camera = Camera(lookFrom, lookAt , [0.0, 1.0, 0.0], vfov,
                    convert(Float64, nx) / convert(Float64, ny), aperture, dist_to_focus,
                    0.0, 1.0)

    R = cos(pi / 4)

    world = cornell_box()

    light_shape = XZRect(213.0, 343.0, 227.0, 332.0, 554.0,
                         DiffuseLight(ConstantTexture([7.0, 7.0, 7.0])))

    glass_sphere = Sphere([190.0, 90.0, 190.0], 90.0,
                          Dielectric(1.5))

    light_list = HitableList([light_shape, glass_sphere])

    for j::Int = ny - 1 : -1 : 0
        for i::Int = 0 : nx - 1

            col::Array{Float64} = [0.0, 0.0, 0.0]

            for s = 1:ns
                u::Float64 = (convert(Float64, i) + rand()) / nx
                v::Float64 = (convert(Float64, j) + rand()) / ny
                r = getRay(camera, u, v)
                sample = remove_nan(color(r, world, light_list, 0))
                col += sample
            end

            col /= convert(Float64, ns)
            if (col[1] < 0.0 || col[2] < 0.0 || col[3] < 0.0)
                @printf(Base.fdio(2), "negative color: <%f, %f, %f>, exiting \n",
                        col[1], col[2], col[3])
                return
            end

            col = sqrt.(col)

            ir::Int = trunc(255.99 * clamp(col[1], 0.0, 1.0))
            ig::Int = trunc(255.99 * clamp(col[2], 0.0, 1.0))
            ib::Int = trunc(255.99 * clamp(col[3], 0.0, 1.0))
            @printf("%d %d %d\n", ir, ig, ib)
        end

        if (j % (ny / 10) == 0)
        @printf(Base.fdio(2), "progress: %f\n", 1.0 - convert(Float64, j) / ny)
        end
    end
end


function main(nx::Int, ny::Int, ns::Int, out::Array{RGB, 2})
    lookFrom = [278.0, 278.0, -800.0]
    lookAt = [278.0, 278.0, 0.0]
    aperture = 0.0
    dist_to_focus = 10.0
    vfov = 40.0
    camera = Camera(lookFrom, lookAt , [0.0, 1.0, 0.0], vfov,
                    convert(Float64, nx) / convert(Float64, ny), aperture, dist_to_focus,
                    0.0, 1.0)

    R = cos(pi / 4)

    world = cornell_box()

    for j::Int = ny - 1 : -1 : 0
        for i::Int = 0 : nx - 1

            col::Array{Float64} = [0.0, 0.0, 0.0]

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
            out[ny - j, i+1] = RGB(col[1], col[2], col[3])
        end

        if (j % (ny / 10) == 0)
        @printf(Base.fdio(2), "progress: %f\n", 1.0 - convert(Float64, j) / ny)
        end
    end
end

function driver()
    nx::Int = 400;
    ny::Int = 400;
    ns::Int = 50;
    out = Array{RGB, 2}(undef, ny, nx)

    main_ppm(nx, ny, ns)

    # main(nx, ny, ns, out)
    # imshow(out)
    # readline(stdin)
end

driver()

# using Profile
# using ProfileView
# ProfileView.view()

