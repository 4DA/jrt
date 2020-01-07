include("CoreTypes.jl")

using Printf
using LinearAlgebra

# reflection math --------------------------------------------------------------
function CosTheta(w::Vec3)::Float64
        return w[3]
end

function AbsCosTheta(w::Vec3)::Float64
 return abs(w[3])
end

function Cos2Theta(w::Vec3)::Float64 
 return w[3] * w[3]
end

function AbsCosTheta(w::Vec3)::Float64 
 return abs(w[3]) 
end

function Sin2Theta(w::Vec3)::Float64 
    return max(0, 1 - Cos2Theta(w))
end

function SinTheta(w::Vec3)::Float64  
 return sqrt(Sin2Theta(w)) 
end

function SinPhi(w::Vec3)::Float64
    sinTheta = SinTheta(w);
    return (sinTheta == 0.0) ? 0.0 : clamp(w[1] / sinTheta, -1.0, 1.0);
end

function CosPhi(w::Vec3)::Float64
    sinTheta = SinTheta(w)
    return sinTheta == 0.0 ? 1.0 : clamp(w[1] / sinTheta, 1.0, 1.0)
end

function Cos2Phi(w::Vec3)::Float64
    return CosPhi(w) * CosPhi(w)
end

function Sin2Phi(w::Vec3)::Float64
    return SinPhi(w) * SinPhi(w)
end

function TanTheta(w::Vec3)::Float64  
    return SinTheta(w) / CosTheta(w)
end

function Tan2Theta(w::Vec3)::Float64  
    return Sin2Theta(w) / Cos2Theta(w);
end

function Faceforward(n::Vec3, v::Vec3)::Vec3
    return (dot(n, v) < 0.0) ? -n : n
end

# ------------------------------------------------------------------------------
abstract type MicrofacetDistribution end
abstract type BxDF <: Material end

abstract type Fresnel end

struct FresnelDielectric <: Fresnel
    etaI::Float64
    etaT::Float64
    k::Vec3
end

function fresnelDielectric(cosThetaI::Float64, etaI::Float64, etaT::Float64)::Float64
    cosThetaI = clamp(cosThetaI, -1.0, 1.0);

    # @printf("cosThetaI = %f\n", cosThetaI)

    # Potentially swap indices of refraction
    entering = cosThetaI > 0.0;
    if (!entering) 
        # swap(etaI, etaT);
        # todo lib function to swap
        tmp = etaT
        etaT = etaI
        etaI = tmp
        cosThetaI = abs(cosThetaI);
    end

    # Compute _cosThetaT_ using Snell's law
    sinThetaI = sqrt(max(0.0, 1.0 - cosThetaI * cosThetaI));
    sinThetaT = etaI / etaT * sinThetaI;

    # @printf("cosThetaI = %f, sinThetaI = %f, sinThetaT = %f\n", cosThetaI, sinThetaI, sinThetaT)

    # Handle total internal reflection
    if sinThetaT >= 1
        return 1;
    end

    cosThetaT = sqrt(max(0.0, 1.0 - sinThetaT * sinThetaT));
    Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
                  ((etaT * cosThetaI) + (etaI * cosThetaT));
    Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
                  ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) / 2.0;
end

function evaluate(f::FresnelDielectric, cosThetaI::Float64)::Float64
    return fresnelDielectric(cosThetaI, f.etaI, f.etaT)
end

struct MicrofacetReflection <: BxDF
    R::Vec3
    distribution::MicrofacetDistribution
    fresnel::Fresnel
end


struct BeckmannDistribution <: MicrofacetDistribution
    alphax::Float64
    alphay::Float64
end

function lambda(dist::BeckmannDistribution, w::Vec3)::Float64
    absTanTheta = abs(TanTheta(w))
    if (isinf(absTanTheta))
        return 0.0
    end

    # Compute _alpha_ for direction _w_
    alpha = sqrt(Cos2Phi(w) * dist.alphax * dist.alphax + Sin2Phi(w) * dist.alphay * dist.alphay);
    a = 1.0 / (alpha * absTanTheta);

    if (a >= 1.6)
        return 0
    end

    return (1.0 - 1.259 * a + 0.396 * a * a) / (3.535 * a + 2.181 * a * a);
end

function G(dist::MicrofacetDistribution, wo::Vec3, wi::Vec3)::Float64
    return 1.0 / (1.0 + lambda(dist, wo) + lambda(dist, wi))
end

# function G1(w::Vec3)::Float64
#     #    if (Dot(w, wh) * CosTheta(w) < 0.) return 0.;
#     return 1 / (1 + Lambda(w));
# end

function D(dist::BeckmannDistribution, wh::Vec3)::Float64
    tan2Theta::Float64 = Tan2Theta(wh);

    if (isinf(tan2Theta))
        return 0.;

    end
    cos4Theta::Float64 = Cos2Theta(wh) * Cos2Theta(wh);

    return exp(-tan2Theta * (Cos2Phi(wh) / (dist.alphax * dist.alphax) +
                             Sin2Phi(wh) / (dist.alphay * dist.alphay))) /
                             (pi * dist.alphax * dist.alphay * cos4Theta);
end

# Torrance–Sparrow BRDF
function f(bxdf::MicrofacetReflection, wo::Vec3, wi::Vec3)::Vec3
    cosThetaO = AbsCosTheta(wo)
    cosThetaI = AbsCosTheta(wi)

    wh = wi + wo;
    # Handle degenerate cases for microfacet reflection

    if cosThetaI == 0 || cosThetaO == 0
        return Vec3(0.);
    end

    if (wh[1] == 0 && wh[2] == 0 && wh[3] == 0)
        return Vec3(0.);
    end

    wh = normalize(wh)
    # For the Fresnel call, make sure that wh is in the same hemisphere
    # as the surface normal, so that TIR is handled correctly.
    F::Float64 = evaluate(bxdf.fresnel, dot(wi, Faceforward(wh, [0.0, 0.0, 1.0])))

    # @printf("fresnel = %f\n", F)
    return bxdf.R * D(bxdf.distribution, wh) * G(bxdf.distribution, wo, wi) * F /
           (4 * cosThetaI * cosThetaO);
end

function sanityTest()
    f(MicrofacetReflection(
        [1.0, 1.0, 1.0],
        BeckmannDistribution(1.0, 1.0), 
                           FresnelDielectric(1.0,
                                             1.5,
                                             [1.0, 1.0, 1.0])), 
      normalize([1.0, 1.0, 1.0]),
      normalize([-1.0, 1.0, 1.0]))
end

function brdfTest(N::Int64)
    S::Vec3 = [0.0, 0.0, 0.0]

    for i = 1:N
        wo = [rand(), rand(), rand()]
        wi = [rand(), rand(), rand()]
        
        S += f(MicrofacetReflection(
            [1.0, 1.0, 1.0],
            BeckmannDistribution(1.0, 1.0), 
            FresnelDielectric(1.0,
                              1.5,
                              [1.0, 1.0, 1.0])), 
               normalize(wo),
               normalize(wi))
    end

    return S / N
end

function emitted(m::MicrofacetReflection, r_in::Ray, rec::HitRecord, u::Float64, v::Float64, p::Vec3)::Vec3
    return [0.0, 0.0, 0.0]
end




# --- Beckmann distribution
# function Pdf(D::MicrofacetDistribution, wo::Vec3, wh::Vec3)::Float64
#     # if (sampleVisibleArea)
#     return D(wh) * G1(wo) * abs(dot(wo, wh)) / AbsCosTheta(wo);
#     # else
#     #     return D(wh) * AbsCosTheta(wh);
# end

# function sampleBeckmann()
# #<<Compute and for Beckmann distribution sample>>= 
#     tan2Theta = 0.0;
#     phi = 0.0;
# if (alphax == alphay) {
#     logSample::Float64 = log(1 - u[0]);
#     if (isinf(logSample)) logSample = 0;
#     tan2Theta = -alphax * alphax * logSample;
#     phi = u[1] * 2 * Pi;
# } else {
# #    <<Compute tan2Theta and phi for anisotropic Beckmann distribution>> 
#        logSample::Float64 = log(u[0]);
#        phi = atan(alphay / alphax *
#                        tan(2 * Pi * u[1] + 0.5f * Pi));
#        if (u[1] > 0.5f)
#            phi += Pi;
#        sinPhi::Float64 = sin(phi), cosPhi = cos(phi);
#        alphax2::Float64 = alphax * alphax, alphay2 = alphay * alphay;
#        tan2Theta = -logSample /
#            (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
# }
# end
# # 

#     function pdf(dist::MicrofacetDistribution,
#         wo::Vec3,
#         wh::Vec3)::Float64
#     # if (sampleVisibleArea)
#     #     return D(wh) * G1(wo) * AbsDot(wo, wh) / AbsCosTheta(wo);
#     # else
#             return D(wh) * AbsCosTheta(wh);
# end

# Vector3f BeckmannDistribution::Sample_wh(const Vector3f &wo,
#                                          const Point2f &u) const {
#     if (!sampleVisibleArea) {
#         // Sample full distribution of normals for Beckmann distribution

#         // Compute $\tan^2 \theta$ and $\phi$ for Beckmann distribution sample
#         Float tan2Theta, phi;
#         if (alphax == alphay) {
#             Float logSample = std::log(1 - u[0]);
#             DCHECK(!std::isinf(logSample));
#             tan2Theta = -alphax * alphax * logSample;
#             phi = u[1] * 2 * Pi;
#         } else {
#             // Compute _tan2Theta_ and _phi_ for anisotropic Beckmann
#             // distribution
#             Float logSample = std::log(1 - u[0]);
#             DCHECK(!std::isinf(logSample));
#             phi = std::atan(alphay / alphax *
#                             std::tan(2 * Pi * u[1] + 0.5f * Pi));
#             if (u[1] > 0.5f) phi += Pi;
#             Float sinPhi = std::sin(phi), cosPhi = std::cos(phi);
#             Float alphax2 = alphax * alphax, alphay2 = alphay * alphay;
#             tan2Theta = -logSample /
#                         (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
#         }

#         // Map sampled Beckmann angles to normal direction _wh_
#         Float cosTheta = 1 / std::sqrt(1 + tan2Theta);
#         Float sinTheta = std::sqrt(std::max((Float)0, 1 - cosTheta * cosTheta));
#         Vector3f wh = SphericalDirection(sinTheta, cosTheta, phi);
#         if (!SameHemisphere(wo, wh)) wh = -wh;
#         return wh;
#     } else {
#         // Sample visible area of normals for Beckmann distribution
#         Vector3f wh;
#         bool flip = wo.z < 0;
#         wh = BeckmannSample(flip ? -wo : wo, alphax, alphay, u[0], u[1]);
#         if (flip) wh = -wh;
#         return wh;
#     }
# }


# Vec3 MicrofacetReflection::Sample_f(wo::Vec3, Vector3f *wi,
#         const Point2f &u, Float *pdf, BxDFType *sampledType) const {
# #    Sample microfacet orientation and reflected direction 

#        Vector3f wh = distribution->Sample_wh(wo, u);
#        *wi = Reflect(wo, wh);
#        if (!SameHemisphere(wo, *wi)) return Vec3(0.f);

# # Compute PDF of wi for microfacet reflection
#             *pdf = distribution->Pdf(wo, wh) / (4 * Dot(wo, wh));

#     return f(wo, *wi);
# }
