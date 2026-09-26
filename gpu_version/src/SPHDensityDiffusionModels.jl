module SPHDensityDiffusionModels

using StaticArrays, LinearAlgebra
using ..SimulationEquations
using ..SimulationGeometry: MotionLimiterValue
#---------------------------------------------------------------
# Exported
#---------------------------------------------------------------
export  SPHDensityDiffusion, 
        ZeroDensityDiffusion, 
        ZeroGravityLinearDensityDiffusion,
        LinearDensityDiffusion,
        ZeroGravityComplexDensityDiffusion,
        ComplexDensityDiffusion,
        compute_density_diffusion


#---------------------------------------------------------------
# Abstract supertype
#---------------------------------------------------------------
abstract type SPHDensityDiffusion end

"""
    compute_density_diffusion(model, SimKernel, SimConstants, SimParticles,
                              xᵢⱼ, ∇ᵢWᵢⱼ, d², ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹, i, j, ParticleType)

Density diffusion contribution of the pair `(i, j)` for the selected `model`.
Returns `(Dᵢ, Dⱼ)`.

The interaction kernel supplies the densities of the state being evaluated
(the predictor density during the corrector loop) and their precomputed
reciprocals, so models neither reload nor divide by density. Same signature
as the CPU package. `SimParticles` is a NamedTuple of the evaluated state's
device arrays for custom models; the built in models only read
`ParticleType`.
"""
function compute_density_diffusion end

#---------------------------------------------------------------
# 1) ZeroDensityDiffusion(): ignore all diffusion
#---------------------------------------------------------------
"""
        ZeroDensityDiffusion()
A model that always returns zero(). No extra density diffusion
and ignores all other parameters.
"""
struct ZeroDensityDiffusion <: SPHDensityDiffusion end

@inline function compute_density_diffusion(
        ::ZeroDensityDiffusion,
        SimKernel,
        SimConstants,
        SimParticles,
        xᵢⱼ,
        ∇ᵢWᵢⱼ,
        d²,
        ρᵢ,
        ρⱼ,
        ρᵢ⁻¹,
        ρⱼ⁻¹,
        i,
        j,
        ParticleType
)
        return zero(d²), zero(d²)
end

#---------------------------------------------------------------
# 2) ZeroGravityLinearDensityDiffusion(): 
#---------------------------------------------------------------
"""
A linear density diffusion approach, but there is no hydrostatic 
term, and we skip ρᵢⱼᴴ enitrely.
"""
struct ZeroGravityLinearDensityDiffusion <: SPHDensityDiffusion end

@inline function compute_density_diffusion(
        ::ZeroGravityLinearDensityDiffusion,
        SimKernel,
        SimConstants,
        SimParticles,
        xᵢⱼ,
        ∇ᵢWᵢⱼ,
        d²,
        ρᵢ,
        ρⱼ,
        ρᵢ⁻¹,
        ρⱼ⁻¹,
        i,
        j,
        ParticleType
)

        (; ρ₀, m₀, c₀, δᵩ, Cb, Cb⁻¹, γ) = SimConstants
        (; h, η²) = SimKernel

        # g == 0 => skip any hydrostatic parts
        
        invdᵢⱼ²η² = one(eltype(ρᵢ)) / (d² + η²)

        ρⱼᵢ = ρⱼ - ρᵢ
        ψᵢⱼ = 2 * ρⱼᵢ * (-xᵢⱼ) * invdᵢⱼ²η²

        Dᵢ  = δᵩ * h * c₀ * (m₀ * ρⱼ⁻¹) * dot(ψᵢⱼ, ∇ᵢWᵢⱼ)
        Dⱼ  = -Dᵢ


        return Dᵢ, Dⱼ
end

#---------------------------------------------------------------
# 3) LinearDensityDiffusion(): Linear approach, uses gravity from
# SimConstants.g
#---------------------------------------------------------------
"""
        LinearDensityDiffusion()

Uses a linear relationship for the hydrostatic correction.
"""
struct LinearDensityDiffusion <: SPHDensityDiffusion end

@inline function compute_density_diffusion(
        ::LinearDensityDiffusion,
        SimKernel,
        SimConstants,
        SimParticles,
        xᵢⱼ,
        ∇ᵢWᵢⱼ,
        d²,
        ρᵢ,
        ρⱼ,
        ρᵢ⁻¹,
        ρⱼ⁻¹,
        i,
        j,
        ParticleType
)

        (; ρ₀, m₀, c₀, δᵩ, Cb, Cb⁻¹, γ, g) = SimConstants
        (; h, η²) = SimKernel

        Linear_ρ_factor = (1/(Cb*γ))*ρ₀

        Pᵢⱼᴴ  = ρ₀ * (-g) * -xᵢⱼ[end]
        ρᵢⱼᴴ  = Pᵢⱼᴴ * Linear_ρ_factor

        
        invdᵢⱼ²η² = one(eltype(ρᵢ)) / (d² + η²)

        ρⱼᵢ = ρⱼ - ρᵢ
        ψᵢⱼ = 2 * (ρⱼᵢ - ρᵢⱼᴴ)  * (-xᵢⱼ) * invdᵢⱼ²η²

        MLcond = MotionLimiterValue(typeof(ρᵢ), ParticleType[i]) *
                 MotionLimiterValue(typeof(ρᵢ), ParticleType[j])

        Dᵢ  = δᵩ * h * c₀ * (m₀ * ρⱼ⁻¹) * dot(ψᵢⱼ, ∇ᵢWᵢⱼ) * MLcond
        Dⱼ  = -Dᵢ

        return Dᵢ, Dⱼ
end

#---------------------------------------------------------------
# 3) LinearDensityDiffusion(): Linear approach, uses gravity from
# SimConstants.g
#---------------------------------------------------------------
"""
        ComplexDensityDiffusion()

Uses a 'complex' relationship for the hydrostatic correction. In essence the inverse
hydrostatic equation of state.
"""
struct ComplexDensityDiffusion <: SPHDensityDiffusion end

@inline function compute_density_diffusion(
        ::ComplexDensityDiffusion,
        SimKernel,
        SimConstants,
        SimParticles,
        xᵢⱼ,
        ∇ᵢWᵢⱼ,
        d²,
        ρᵢ,
        ρⱼ,
        ρᵢ⁻¹,
        ρⱼ⁻¹,
        i,
        j,
        ParticleType
)

        (; ρ₀, m₀, c₀, δᵩ, Cb, Cb⁻¹, γ, g) = SimConstants
        (; h, η²) = SimKernel

        # In theory these two equations are not completely symmetric.
        # In practice it is 'good' enough and saves a lot of time to
        # not do it the mathematically correct way.
        Pᵢⱼᴴ  = ρ₀ * (-g) * -xᵢⱼ[end]
        ρᵢⱼᴴ  = InverseHydrostaticEquationOfState(ρ₀, Pᵢⱼᴴ, Cb⁻¹)
        # ρᵢⱼᴴ = ρ₀ * ( Estimate7thRoot( 1 + (Pᵢⱼᴴ * Cb⁻¹)) - 1)
        # ρⱼᵢᴴ  = InverseHydrostaticEquationOfState(ρ₀, Pⱼᵢᴴ, Cb⁻¹)
        
        invdᵢⱼ²η² = one(eltype(ρᵢ)) / (d² + η²)

        ρⱼᵢ = ρⱼ - ρᵢ
        ψᵢⱼ = 2 * (ρⱼᵢ - ρᵢⱼᴴ)  * (-xᵢⱼ) * invdᵢⱼ²η²

        MLcond = MotionLimiterValue(typeof(ρᵢ), ParticleType[i]) *
                 MotionLimiterValue(typeof(ρᵢ), ParticleType[j])

        Dᵢ  = δᵩ * h * c₀ * (m₀ * ρⱼ⁻¹) * dot(ψᵢⱼ, ∇ᵢWᵢⱼ) * MLcond
        Dⱼ  = -Dᵢ

        return Dᵢ, Dⱼ
end

end #module SPHDensityDiffusionModels
