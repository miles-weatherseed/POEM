using YAML 
using DifferentialEquations, LinearAlgebra
using DelimitedFiles
using NLsolve
using Sundials
using LSODA
using Statistics
using NPZ
using ProgressMeter

input_dir = ARGS[1]

# Load the configuration from YAML file
config = YAML.load_file("lib/mechanistic_model/data/mechanistic_model_settings.yml")

ed_config = config["ed_parameters"]
ed_order = [
    "gM", "gT", "dM", "dT", "e", "c", "bT", "uT", "dMe", "q", "v", 
    "dP", "u_binder", "u_nbinder", "g_self_er", "rho_b"
]

# Extract parameters in order
ed_parameters = [ed_config[param] for param in ed_order]

hearn_config = config["hearn_fitted_parameters"]
hearn_order = ["d_cyt", "u_TAP", "k_TAP", "E0", "T0", "g_self_cyt"]

hearn_parameters = [hearn_config[param] for param in hearn_order]

# Assigning parameters to variables
# Unpacking ed_parameters into typed variables
const (gM::Float64, gT::Float64, dM::Float64, dT::Float64, e::Float64, 
       c::Float64, bT::Float64, uT::Float64, dMe::Float64, q::Float64, 
       v::Float64, dP::Float64, u_binder::Float64, u_nbinder::Float64, 
       g_self_er::Float64, rho_b::Float64) = ed_parameters

# Unpacking hearn_parameters into typed variables
const (d_cyt::Float64, u_TAP::Float64, k_TAP::Float64, 
       E0::Float64, T0::Float64, g_self_cyt::Float64) = hearn_parameters

# Example calculation based on parameters
const b_TAP_self::Float64 = ((u_TAP + k_TAP) * d_cyt) / (g_self_cyt * (d_cyt + k_TAP))
const v_ER::Float64 = hearn_config["V_ER"]
const v_TAP::Float64 = hearn_config["V_TAP"]
const Km::Float64 = 100 * v_ER


function endog_derivative!(du::AbstractVector, u::AbstractVector, p::Real, t::Real)

    b = p

    P_selfC = u[1]
    TAP = u[2]
    TPself = u[3]
    P_nbinder = u[4]
    P_binder = u[5]
    M = u[6]
    T = u[7]
    TM = u[8]
    MP_nbinder = u[9]
    MP_binder = u[10]
    TMP_nbinder = u[11]
    TMP_binder = u[12]
    MeP_nbinder = u[13]
    MeP_binder = u[14]
    Me = u[15]

    du[1] = (
        g_self_cyt
        - b_TAP_self * TAP * P_selfC
        - d_cyt * P_selfC
        + u_TAP * TPself
    )
    du[2] = (k_TAP + u_TAP) * (TPself) - (b_TAP_self * P_selfC) * TAP

    du[3] = b_TAP_self * P_selfC * TAP - (k_TAP + u_TAP) * TPself
    du[4] = (
        (1 - rho_b) * k_TAP * TPself
        - b * M * P_nbinder
        + u_nbinder * MP_nbinder
        + q * u_nbinder * TMP_nbinder
        - c * TM * P_nbinder
        - dP * P_nbinder
    )
    du[5] = (
        rho_b * k_TAP * TPself
        - b * M * P_binder
        + u_binder * MP_binder
        + q * u_binder * TMP_binder
        - c * TM * P_binder
        - dP * P_binder
    )
    du[6] = (
        gM
        + u_nbinder * MP_nbinder
        + u_binder * MP_binder
        + uT * TM
        - b * (P_binder + P_nbinder) * M
        - dM * M
        - bT * T * M
    )
    du[7] = (
        uT * TM + gT + uT * v * (TMP_binder + TMP_nbinder) - (bT * M + dT) * T
    )
    du[8] = (
        bT * T * M
        + q * (u_nbinder * TMP_nbinder + u_binder * TMP_binder)
        - (uT + c * (P_binder + P_nbinder)) * TM
    )
    du[9] = (
        b * M * P_nbinder - (u_nbinder + e) * MP_nbinder + uT * v * TMP_nbinder
    )
    du[10] = b * M * P_binder - (u_binder + e) * MP_binder + uT * v * TMP_binder
    du[11] = c * TM * P_nbinder - (q * u_nbinder + uT * v) * TMP_nbinder
    du[12] = c * TM * P_binder - (q * u_binder + uT * v) * TMP_binder
    du[13] = e * MP_nbinder - u_nbinder * MeP_nbinder
    du[14] = e * MP_binder - u_binder * MeP_binder
    du[15] = u_nbinder * MeP_nbinder + u_binder * MeP_binder - dMe * Me

end


function derivative!(du::AbstractVector, u::AbstractVector, p::Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64}, t::Real)
    gPs, b_TAPs, kcs_in, kcs_out, us, kxs_in, kxs_out, b = p
    P_Cs = u[1:n_peptides]
    P_selfC = u[n_peptides + 1]
    TAP = u[n_peptides + 2]
    TPs = u[n_peptides + 3:2 * n_peptides + 2]
    TPself = u[2 * n_peptides + 3]
    P_nbinder = u[2 * n_peptides + 4]
    P_binder = u[2 * n_peptides + 5]
    Ps = u[2 * n_peptides + 6:3 * n_peptides + 5]
    M = u[3 * n_peptides + 6]
    T = u[3 * n_peptides + 7]
    TM = u[3 * n_peptides + 8]
    MP_nbinder = u[3 * n_peptides + 9]
    MP_binder = u[3 * n_peptides + 10]
    MPs = u[3 * n_peptides + 11:4 * n_peptides + 10]
    TMP_nbinder = u[4 * n_peptides + 11]
    TMP_binder = u[4 * n_peptides + 12]
    TMPs = u[4 * n_peptides + 13:5 * n_peptides + 12]
    MeP_nbinder = u[5 * n_peptides + 13]
    MeP_binder = u[5 * n_peptides + 14]
    MePs = u[5 * n_peptides + 15:6 * n_peptides + 14]
    Me = u[6 * n_peptides + 15]

    du[1:n_peptides] .= gPs .- b_TAPs .* TAP .* P_Cs .- d_cyt .* P_Cs .+ u_TAP .* TPs .- kcs_out .* P_Cs .+ kcs_in .* vcat([0.0], P_Cs[1:end-1])
    du[n_peptides + 1] = g_self_cyt - b_TAP_self * TAP * P_selfC - d_cyt * P_selfC + u_TAP * TPself
    du[n_peptides + 2] = (k_TAP + u_TAP) * (sum(TPs) + TPself) - (b_TAP_self * P_selfC + dot(b_TAPs, P_Cs)) * TAP

    du[n_peptides + 3:2 * n_peptides + 2] .= b_TAPs .* TAP .* P_Cs .- (k_TAP + u_TAP) .* TPs
    du[2 * n_peptides + 3] = b_TAP_self * P_selfC * TAP - (k_TAP + u_TAP) * TPself
    du[2 * n_peptides + 4] = (1 - rho_b) * k_TAP * TPself - b * M * P_nbinder + u_nbinder * MP_nbinder + q * u_nbinder * TMP_nbinder - c * TM * P_nbinder - dP * P_nbinder
    du[2 * n_peptides + 5] = rho_b * k_TAP * TPself - b * M * P_binder + u_binder * MP_binder + q * u_binder * TMP_binder - c * TM * P_binder - dP * P_binder

    du[2 * n_peptides + 6:3 * n_peptides + 5] .= k_TAP .* TPs .- b .* M .* Ps .+ us .* MPs .+ q .* us .* TMPs .- c .* TM .* Ps .- dP .* Ps .- E0 * (kxs_out .* Ps / Km) ./ (1 .+ sum(Ps) / Km .+ (P_nbinder + P_binder) / Km) .+ E0 * (kxs_in .* vcat([0.0], Ps[1:end-1]) / Km) ./ (1 .+ sum(Ps) / Km .+ (P_nbinder + P_binder) / Km)
    du[3 * n_peptides + 6] = gM + u_nbinder * MP_nbinder + u_binder * MP_binder + dot(us, MPs) + uT * TM - b * (P_binder + P_nbinder + sum(Ps)) * M - dM * M - bT * T * M
    du[3 * n_peptides + 7] = uT * TM + gT + uT * v * (TMP_binder + TMP_nbinder + sum(TMPs)) - (bT * M + dT) * T
    du[3 * n_peptides + 8] = bT * T * M + q * (u_nbinder * TMP_nbinder + u_binder * TMP_binder + dot(us, TMPs)) - (uT + c * (P_binder + P_nbinder + sum(Ps))) * TM
    du[3 * n_peptides + 9] = b * M * P_nbinder - (u_nbinder + e) * MP_nbinder + uT * v * TMP_nbinder
    du[3 * n_peptides + 10] = b * M * P_binder - (u_binder + e) * MP_binder + uT * v * TMP_binder
    du[3 * n_peptides + 11:4 * n_peptides + 10] .= b * M .* Ps .- (us .+ e) .* MPs .+ uT .* v .* TMPs

    du[4 * n_peptides + 11] = c * TM * P_nbinder - (q * u_nbinder + uT * v) * TMP_nbinder
    du[4 * n_peptides + 12] = c * TM * P_binder - (q * u_binder + uT * v) * TMP_binder
    du[4 * n_peptides + 13:5 * n_peptides + 12] .= c .* TM .* Ps .- (q .* us .+ uT .* v) .* TMPs
    du[5 * n_peptides + 13] = e * MP_nbinder - u_nbinder * MeP_nbinder
    du[5 * n_peptides + 14] = e * MP_binder - u_binder * MeP_binder
    du[5 * n_peptides + 15:6 * n_peptides + 14] .= e .* MPs .- us .* MePs
    du[6 * n_peptides + 15] = u_nbinder * MeP_nbinder + u_binder * MeP_binder + dot(us, MePs) - dMe * Me
end

# Define the residual function that represents the equilibrium condition
function equilibrium_residual!(residual, u, p, t)
    du = zeros(length(u))  # du will store the derivatives
    endog_derivative!(du, u, p, t)  # Compute derivatives
    residual .= du  # Residual is zero at equilibrium, so we set residual to du
end

function endog_to_exog(eqm::Vector{Float64}, n_peptides::Int)
    """Takes in eqm solution of endog only model and returns startpoint for full model"""
    return vcat(
        zeros(n_peptides),
        eqm[1],
        eqm[2],
        zeros(n_peptides),
        eqm[3],
        eqm[4],
        eqm[5],
        zeros(n_peptides),
        eqm[6],
        eqm[7],
        eqm[8],
        eqm[9],
        eqm[10],
        zeros(n_peptides),
        eqm[11],
        eqm[12],
        zeros(n_peptides),
        eqm[13],
        eqm[14],
        zeros(n_peptides),
        eqm[15]
    )
end

gPs = 100.0 .* readdlm("/Users/weatherseed/Documents/POEM_Training/cache/test_data/covid/gPs_pepsickle-epitope.txt", ' ')
b_TAPs = readdlm("/Users/weatherseed/Documents/POEM_Training/cache/test_data/covid/taps.txt", ' ')
kcxs_in = readdlm("/Users/weatherseed/Documents/POEM_Training/cache/test_data/covid/kcxs_in.txt", ' ')
kcxs_out = readdlm("/Users/weatherseed/Documents/POEM_Training/cache/test_data/covid/kcxs_out.txt", ' ')

kcats_out = readdlm("/Users/weatherseed/Documents/POEM_Training/cache/test_data/covid/kcats_out.txt", ' ')
kcats_in = readdlm("/Users/weatherseed/Documents/POEM_Training/cache/test_data/covid/kcats_in.txt", ' ')

bs = 10 .^ readdlm("/Users/weatherseed/Documents/POEM_Training/cache/test_data/covid/bERs.txt", ' ')
us = bs .* 1e-3 .* v_ER .* readdlm("/Users/weatherseed/Documents/POEM_Training/cache/test_data/covid/NetMHCpan_Affinities.txt", ' ')

n_peptides::Int = 9 # 16 --> 8

startp = zeros(15)
startp[2] = T0  # Remember Julia uses 1-based indexing

# Solving the endogenous model
tspan_endog = (0.0, 1_000_000.0)
t_endog = range(0.0, stop=200_000.0, length=1000)
outputs = zeros(size(gPs))

@showprogress for i in 1:size(gPs, 1)


    prob_endog = ODEProblem(endog_derivative!, startp, tspan_endog, bs[i, 1])
    sol_endog = solve(prob_endog, Rodas4(), saveat=t_endog)

    # Solve for the equilibrium point
    x0 = sol_endog.u[end]
    exog_x0 = endog_to_exog(x0, n_peptides)

    # Define parameters
    p = (
        gPs[i, :],
        (u_TAP ./ (b_TAPs[i, :] .* v_TAP)),
        kcxs_in[i, :],
        kcxs_out[i, :],
        us[i, :],
        kcats_in[i, :],
        kcats_out[i, :],
        bs[i, 1],
    )

    # Solving the exogenous model
    prob_exog = ODEProblem(derivative!, exog_x0, (0.0, 1_000_000.0), p)
    sol_exog = solve(prob_exog, TRBDF2(), abstol=1e-6, reltol=1e-3, saveat=1_000_000.0)
    outputs[i, :] = sol_exog.u[end][end-n_peptides:end-1]

end

writedlm("mechanistic_outputs/test/covid/NetMHCpan_pepsickle-epitope_Julia.csv", outputs, ",")