module SPHDensityDiffusionModels

using StaticArrays, LinearAlgebra, Parameters
using ..SimulationEquations
using ..SimulationGeometry
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

# Propagate the neighbor loop's @inbounds context to particle-array reads.
# Calls made outside that context retain their ordinary bounds checks.
#
# Every model receives the pair densities of the evaluated state (the
# predictor density during midpoint evaluations) and their precomputed
# reciprocals from the pair loop, so no model reloads or divides by density.

#---------------------------------------------------------------
# 1) ZeroDensityDiffusion(): ignore all diffusion
#---------------------------------------------------------------
"""
        ZeroDensityDiffusion()
A model that always returns zero(). No extra density diffusion
and ignores all other parameters.
"""
struct ZeroDensityDiffusion <: SPHDensityDiffusion end

Base.@propagate_inbounds function compute_density_diffusion(
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

Base.@propagate_inbounds function compute_density_diffusion(
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

        @unpack ρ₀, m₀, c₀, δᵩ, Cb, Cb⁻¹, γ    = SimConstants
        @unpack h, η²                          = SimKernel

        # g == 0 => skip any hydrostatic parts

        invdᵢⱼ²η² = one(eltype(ρᵢ)) / (d² + η²)

        ρⱼᵢ = ρⱼ - ρᵢ
        ψᵢⱼ = 2 * ρⱼᵢ * (-xᵢⱼ) * invdᵢⱼ²η²

        # ψⱼᵢ equals ψᵢⱼ exactly and ∇ⱼWᵢⱼ = -∇ᵢWᵢⱼ, so Dⱼ is what particle j
        # computes as the center, not merely -Dᵢ.
        ψ∇W = dot(ψᵢⱼ, ∇ᵢWᵢⱼ)
        Dᵢ  = δᵩ * h * c₀ * (m₀ * ρⱼ⁻¹) * ψ∇W
        Dⱼ  = δᵩ * h * c₀ * (m₀ * ρᵢ⁻¹) * -ψ∇W


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

Base.@propagate_inbounds function compute_density_diffusion(
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

        # Only fluid-fluid pairs diffuse in this model. Reject boundary pairs
        # before evaluating the hydrostatic correction and distance reciprocal.
        Typeᵢ = ParticleType[i]
        Typeⱼ = ParticleType[j]
        if Typeᵢ != Fluid || Typeⱼ != Fluid
            return zero(d²), -zero(d²)
        end

        @unpack ρ₀, m₀, c₀, δᵩ, Cb, Cb⁻¹, γ, g = SimConstants
        @unpack h, η²                          = SimKernel

        Linear_ρ_factor = (1/(Cb*γ))*ρ₀

        Pᵢⱼᴴ  = ρ₀ * (-g) * -xᵢⱼ[end]
        ρᵢⱼᴴ  = Pᵢⱼᴴ * Linear_ρ_factor


        invdᵢⱼ²η² = one(eltype(ρᵢ)) / (d² + η²)

        ρⱼᵢ = ρⱼ - ρᵢ
        ψᵢⱼ = 2 * (ρⱼᵢ - ρᵢⱼᴴ)  * (-xᵢⱼ) * invdᵢⱼ²η²

        # The linear hydrostatic term is exactly antisymmetric, so ψⱼᵢ equals
        # ψᵢⱼ and Dⱼ is what particle j computes as the center.
        ψ∇W = dot(ψᵢⱼ, ∇ᵢWᵢⱼ)
        Dᵢ  = δᵩ * h * c₀ * (m₀ * ρⱼ⁻¹) * ψ∇W
        Dⱼ  = δᵩ * h * c₀ * (m₀ * ρᵢ⁻¹) * -ψ∇W

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
Use Float64: the current inverse-hydrostatic estimator relies on its bit layout.
"""
struct ComplexDensityDiffusion <: SPHDensityDiffusion end

Base.@propagate_inbounds function compute_density_diffusion(
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

        # Boundary pairs contribute zero in this model. Skip their expensive
        # inverse hydrostatic equation of state and divisions altogether.
        Typeᵢ = ParticleType[i]
        Typeⱼ = ParticleType[j]
        if Typeᵢ != Fluid || Typeⱼ != Fluid
            return zero(d²), -zero(d²)
        end

        @unpack ρ₀, m₀, c₀, δᵩ, Cb, Cb⁻¹, γ, g = SimConstants
        @unpack h, η²                          = SimKernel

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

        MotionLimiterCondition = MotionLimiterValue(eltype(ρᵢ), ParticleType[i]) * MotionLimiterValue(eltype(ρᵢ), ParticleType[j])

        Dᵢ  = δᵩ * h * c₀ * (m₀ * ρⱼ⁻¹) * dot(ψᵢⱼ, ∇ᵢWᵢⱼ) * MotionLimiterCondition

        # The inverse equation of state is not antisymmetric in the pressure
        # sign, so evaluate what particle j computes as the center exactly
        # (with -xᵢⱼ and -∇ᵢWᵢⱼ) rather than negating Dᵢ.
        Pⱼᵢᴴ  = ρ₀ * (-g) * xᵢⱼ[end]
        ρⱼᵢᴴ  = InverseHydrostaticEquationOfState(ρ₀, Pⱼᵢᴴ, Cb⁻¹)
        ρᵢⱼ = ρᵢ - ρⱼ
        ψⱼᵢ = 2 * (ρᵢⱼ - ρⱼᵢᴴ) * xᵢⱼ * invdᵢⱼ²η²
        Dⱼ  = δᵩ * h * c₀ * (m₀ * ρᵢ⁻¹) * dot(ψⱼᵢ, -∇ᵢWᵢⱼ) * MotionLimiterCondition

        return Dᵢ, Dⱼ
end

end #module SPHDensityDiffusionModels
