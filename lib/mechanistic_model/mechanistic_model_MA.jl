using YAML 
using DifferentialEquations
using LinearAlgebra
using DelimitedFiles
using NPZ
using ProgressMeter

# directory containing input files
input_dir = ARGS[1]
yaml_dir = ARGS[2]
#current_dir = @__DIR__

# Load the configuration from YAML file
config = YAML.load_file(yaml_dir)

ed_config = config["ed_parameters"]
ed_order = [
    "gM_A1", "gM_A2", "gM_B1", "gM_B2", "gM_C1", "gM_C2", "gT", "dM", "dT", "e", "c", "bT", "uT", "dMe", "q", "v", 
    "dP", "u_binder", "u_nbinder", "g_self_er", "rho_b"
]

# Extract parameters in order
ed_parameters = [ed_config[param] for param in ed_order]

hearn_config = config["hearn_fitted_parameters"]
hearn_order = ["d_cyt", "u_TAP", "k_TAP", "E0", "T0", "g_self_cyt"]

hearn_parameters = [hearn_config[param] for param in hearn_order]

# Unpacking ed_parameters into typed variables
const (gM_A1::Float64, gM_A2::Float64, gM_B1::Float64, gM_B2::Float64, gM_C1::Float64, gM_C2::Float64, gT::Float64, dM::Float64, dT::Float64, e::Float64, 
       c::Float64, bT::Float64, uT::Float64, dMe::Float64, q::Float64, 
       v::Float64, dP::Float64, u_binder::Float64, u_nbinder::Float64, 
       g_self_er::Float64, rho_b::Float64) = ed_parameters

# Unpacking hearn_parameters into typed variables
const (d_cyt::Float64, u_TAP::Float64, k_TAP::Float64, 
       E0::Float64, T0::Float64, g_self_cyt::Float64) = hearn_parameters

# Calculation of parameters
const b_TAP_self::Float64 = ((u_TAP + k_TAP) * d_cyt) / (g_self_cyt * (d_cyt + k_TAP))
const v_ER::Float64 = hearn_config["V_ER"]
const v_TAP::Float64 = hearn_config["V_TAP"]
const Km::Float64 = 100 * v_ER


function endogenous_derivative!(du::AbstractVector, u::AbstractVector, p::Tuple{Float64, Float64, Float64, Float64, Float64, Float64}, t::Real)

    b_A1, b_A2, b_B1, b_B2, b_C1, b_C2 = p

    # Unpack states
    P_selfC = u[1]
    TAP = u[2]
    TPself = u[3]
    P_nbinder = u[4]
    P_binder = u[5]

    # States for alleles
    MA1 = u[6]
    MA2 = u[7]
    MB1 = u[8]
    MB2 = u[9]
    MC1 = u[10]
    MC2 = u[11]

    T = u[12]
    
    TMA1 = u[13]
    TMA2 = u[14]
    TMB1 = u[15]
    TMB2 = u[16]
    TMC1 = u[17]
    TMC2 = u[18]

    MA1P_nbinder = u[19]
    MA1P_binder = u[20]
    MA2P_nbinder = u[21]
    MA2P_binder = u[22]
    MB1P_nbinder = u[23]
    MB1P_binder = u[24]
    MB2P_nbinder = u[25]
    MB2P_binder = u[26]
    MC1P_nbinder = u[27]
    MC1P_binder = u[28]
    MC2P_nbinder = u[29]
    MC2P_binder = u[30]

    TMA1P_nbinder = u[31]
    TMA1P_binder = u[32]
    TMA2P_nbinder = u[33]
    TMA2P_binder = u[34]
    TMB1P_nbinder = u[35]
    TMB1P_binder = u[36]
    TMB2P_nbinder = u[37]
    TMB2P_binder = u[38]
    TMC1P_nbinder = u[39]
    TMC1P_binder = u[40]
    TMC2P_nbinder = u[41]
    TMC2P_binder = u[42]

    MeA1P_nbinder = u[43]
    MeA2P_nbinder = u[44]
    MeB1P_nbinder = u[45]
    MeB2P_nbinder = u[46]
    MeC1P_nbinder = u[47]
    MeC2P_nbinder = u[48]

    MeA1P_binder = u[49]
    MeA2P_binder = u[50]
    MeB1P_binder = u[51]
    MeB2P_binder = u[52]
    MeC1P_binder = u[53]
    MeC2P_binder = u[54]

    MeA1 = u[55]
    MeA2 = u[56]
    MeB1 = u[57]
    MeB2 = u[58]
    MeC1 = u[59]
    MeC2 = u[60]

    # Differential equations for P_selfC
    du[1] = g_self_cyt - b_TAP_self * TAP * P_selfC - d_cyt * P_selfC + u_TAP * TPself

    # Differential equation for TAP
    du[2] = (k_TAP + u_TAP) * TPself - b_TAP_self * P_selfC * TAP

    du[3] = b_TAP_self * P_selfC * TAP - (k_TAP + u_TAP) * TPself

    du[4] = (1 - rho_b) * k_TAP * TPself - (b_A1 * MA1 + b_A2 * MA2 + b_B1 * MB1 + b_B2 * MB2 + b_C1 * MC1 + b_C2 * MC2) * P_nbinder + u_nbinder * (MA1P_nbinder + MA2P_nbinder + MB1P_nbinder + MB2P_nbinder + MC1P_nbinder + MC2P_nbinder) + q * u_nbinder * (TMA1P_nbinder + TMA2P_nbinder + TMB1P_nbinder + TMB2P_nbinder + TMC1P_nbinder + TMC2P_nbinder) - c * (TMA1 + TMA2 + TMB1 + TMB2 + TMC1 + TMC2) * P_nbinder - dP * P_nbinder

    du[5] = rho_b * k_TAP * TPself - (b_A1 * MA1 + b_A2 * MA2 + b_B1 * MB1 + b_B2 * MB2 + b_C1 * MC1 + b_C2 * MC2) * P_binder + u_binder * (MA1P_binder + MA2P_binder + MB1P_binder + MB2P_binder + MC1P_binder + MC2P_binder) + q * u_binder * (TMA1P_binder + TMA2P_binder + TMB1P_binder + TMB2P_binder + TMC1P_binder + TMC2P_binder) - c * (TMA1 + TMA2 + TMB1 + TMB2 + TMC1 + TMC2) * P_binder - dP * P_binder

    # Differential equations for MA1 --> MC2
    du[6] = gM_A1 + u_nbinder * MA1P_nbinder + u_binder * MA1P_binder + uT * TMA1 - b_A1 * (P_binder + P_nbinder) * MA1 - dM * MA1 - bT * T * MA1
    du[7] = gM_A2 + u_nbinder * MA2P_nbinder + u_binder * MA2P_binder + uT * TMA2 - b_A2 * (P_binder + P_nbinder) * MA2 - dM * MA2 - bT * T * MA2
    du[8] = gM_B1 + u_nbinder * MB1P_nbinder + u_binder * MB1P_binder + uT * TMB1 - b_B1 * (P_binder + P_nbinder) * MB1 - dM * MB1 - bT * T * MB1
    du[9] = gM_B2 + u_nbinder * MB2P_nbinder + u_binder * MB2P_binder + uT * TMB2 - b_B2 * (P_binder + P_nbinder) * MB2 - dM * MB2 - bT * T * MB2
    du[10] = gM_C1 + u_nbinder * MC1P_nbinder + u_binder * MC1P_binder + uT * TMC1 - b_C1 * (P_binder + P_nbinder) * MC1 - dM * MC1 - bT * T * MC1
    du[11] = gM_C2 + u_nbinder * MC2P_nbinder + u_binder * MC2P_binder + uT * TMC2 - b_C2 * (P_binder + P_nbinder) * MC2 - dM * MC2 - bT * T * MC2

    # T
    du[12] = uT * (TMA1 + TMA2 + TMB1 + TMB2 + TMC1 + TMC2) + gT + uT * v * (TMA1P_binder + TMA1P_nbinder + TMA2P_binder + TMA2P_nbinder + TMB1P_binder + TMB1P_nbinder + TMB2P_binder + TMB2P_nbinder + TMC1P_binder + TMC1P_nbinder + TMC2P_binder + TMC2P_nbinder) - (bT * (MA1 + MA2 + MB1 + MB2 + MC1 + MC2) + dT) * T

    # TMA1 --> TMC2
    du[13] = bT * T * MA1 + q * (u_nbinder * TMA1P_nbinder + u_binder * TMA1P_binder) - (uT + c * (P_binder + P_nbinder)) * TMA1
    du[14] = bT * T * MA2 + q * (u_nbinder * TMA2P_nbinder + u_binder * TMA2P_binder) - (uT + c * (P_binder + P_nbinder)) * TMA2
    du[15] = bT * T * MB1 + q * (u_nbinder * TMB1P_nbinder + u_binder * TMB1P_binder) - (uT + c * (P_binder + P_nbinder)) * TMB1
    du[16] = bT * T * MB2 + q * (u_nbinder * TMB2P_nbinder + u_binder * TMB2P_binder) - (uT + c * (P_binder + P_nbinder)) * TMB2
    du[17] = bT * T * MC1 + q * (u_nbinder * TMC1P_nbinder + u_binder * TMC1P_binder) - (uT + c * (P_binder + P_nbinder)) * TMC1
    du[18] = bT * T * MC2 + q * (u_nbinder * TMC2P_nbinder + u_binder * TMC2P_binder) - (uT + c * (P_binder + P_nbinder)) * TMC2

    # MA1P_nbinder --> MC2P_binder
    du[19] = b_A1 * MA1 * P_nbinder - (u_nbinder + e) * MA1P_nbinder + uT * v * TMA1P_nbinder
    du[20] = b_A1 * MA1 * P_binder - (u_binder + e) * MA1P_binder + uT * v * TMA1P_binder
    du[21] = b_A2 * MA2 * P_nbinder - (u_nbinder + e) * MA2P_nbinder + uT * v * TMA2P_nbinder
    du[22] = b_A2 * MA2 * P_binder - (u_binder + e) * MA2P_binder + uT * v * TMA2P_binder
    du[23] = b_B1 * MB1 * P_nbinder - (u_nbinder + e) * MB1P_nbinder + uT * v * TMB1P_nbinder
    du[24] = b_B1 * MB1 * P_binder - (u_binder + e) * MB1P_binder + uT * v * TMB1P_binder
    du[25] = b_B2 * MB2 * P_nbinder - (u_nbinder + e) * MB2P_nbinder + uT * v * TMB2P_nbinder
    du[26] = b_B2 * MB2 * P_binder - (u_binder + e) * MB2P_binder + uT * v * TMB2P_binder
    du[27] = b_C1 * MC1 * P_nbinder - (u_nbinder + e) * MC1P_nbinder + uT * v * TMC1P_nbinder
    du[28] = b_C1 * MC1 * P_binder - (u_binder + e) * MC1P_binder + uT * v * TMC1P_binder
    du[29] = b_C2 * MC2 * P_nbinder - (u_nbinder + e) * MC2P_nbinder + uT * v * TMC2P_nbinder
    du[30] = b_C2 * MC2 * P_binder - (u_binder + e) * MC2P_binder + uT * v * TMC2P_binder

    # TMPA1_nbinder --> TMPC2_binder
    du[31] = c * TMA1 * P_nbinder - (q * u_nbinder + uT * v) * TMA1P_nbinder
    du[32] = c * TMA1 * P_binder - (q * u_binder + uT * v) * TMA1P_binder
    du[33] = c * TMA2 * P_nbinder - (q * u_nbinder + uT * v) * TMA2P_nbinder
    du[34] = c * TMA2 * P_binder - (q * u_binder + uT * v) * TMA2P_binder
    du[35] = c * TMB1 * P_nbinder - (q * u_nbinder + uT * v) * TMB1P_nbinder
    du[36] = c * TMB1 * P_binder - (q * u_binder + uT * v) * TMB1P_binder
    du[37] = c * TMB2 * P_nbinder - (q * u_nbinder + uT * v) * TMB2P_nbinder
    du[38] = c * TMB2 * P_binder - (q * u_binder + uT * v) * TMB2P_binder
    du[39] = c * TMC1 * P_nbinder - (q * u_nbinder + uT * v) * TMC1P_nbinder
    du[40] = c * TMC1 * P_binder - (q * u_binder + uT * v) * TMC1P_binder
    du[41] = c * TMC2 * P_nbinder - (q * u_nbinder + uT * v) * TMC2P_nbinder
    du[42] = c * TMC2 * P_binder - (q * u_binder + uT * v) * TMC2P_binder

    # MeA1P_nbinder --> MeC2P_nbinder
    du[43] = e * MA1P_nbinder - u_nbinder * MeA1P_nbinder
    du[44] = e * MA2P_nbinder - u_nbinder * MeA2P_nbinder
    du[45] = e * MB1P_nbinder - u_nbinder * MeB1P_nbinder
    du[46] = e * MB2P_nbinder - u_nbinder * MeB2P_nbinder
    du[47] = e * MC1P_nbinder - u_nbinder * MeC1P_nbinder
    du[48] = e * MC2P_nbinder - u_nbinder * MeC2P_nbinder

    # MeA1P_binder --> MeC2P_binder
    du[49] = e * MA1P_binder - u_binder * MeA1P_binder
    du[50] = e * MA2P_binder - u_binder * MeA2P_binder
    du[51] = e * MB1P_binder - u_binder * MeB1P_binder
    du[52] = e * MB2P_binder - u_binder * MeB2P_binder
    du[53] = e * MC1P_binder - u_binder * MeC1P_binder
    du[54] = e * MC2P_binder - u_binder * MeC2P_binder

    # MeA1 --> MeC2
    du[55] = u_nbinder * MeA1P_nbinder + u_binder * MeA1P_binder - dMe * MeA1
    du[56] = u_nbinder * MeA2P_nbinder + u_binder * MeA2P_binder - dMe * MeA2
    du[57] = u_nbinder * MeB1P_nbinder + u_binder * MeB1P_binder - dMe * MeB1
    du[58] = u_nbinder * MeB2P_nbinder + u_binder * MeB2P_binder - dMe * MeB2
    du[59] = u_nbinder * MeC1P_nbinder + u_binder * MeC1P_binder - dMe * MeC1
    du[60] = u_nbinder * MeC2P_nbinder + u_binder * MeC2P_binder - dMe * MeC2

end


function derivative!(du::AbstractVector, u::AbstractVector, p::Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64, Float64, Float64, Float64, Float64, Float64}, t::Real)
    # Unpack parameters
    gPs, b_TAPs, cytamin_in, cytamin_out, us_A1, us_A2, us_B1, us_B2, us_C1, us_C2, erap1_kcat_in, erap1_kcat_out, b_A1, b_A2, b_B1, b_B2, b_C1, b_C2 = p
    
    # Unpack states
    P_Cs = u[1:n_peptides]
    P_selfC = u[n_peptides + 1]
    TAP = u[n_peptides + 2]
    TPs = u[n_peptides + 3:2 * n_peptides + 2]
    TPself = u[2 * n_peptides + 3]
    P_nbinder = u[2 * n_peptides + 4]
    P_binder = u[2 * n_peptides + 5]
    Ps = u[2 * n_peptides + 6:3 * n_peptides + 5]

    # States for alleles
    MA1 = u[3 * n_peptides + 6]
    MA2 = u[3 * n_peptides + 7]
    MB1 = u[3 * n_peptides + 8]
    MB2 = u[3 * n_peptides + 9]
    MC1 = u[3 * n_peptides + 10]
    MC2 = u[3 * n_peptides + 11]

    T = u[3 * n_peptides + 12]
    
    TMA1 = u[3 * n_peptides + 13]
    TMA2 = u[3 * n_peptides + 14]
    TMB1 = u[3 * n_peptides + 15]
    TMB2 = u[3 * n_peptides + 16]
    TMC1 = u[3 * n_peptides + 17]
    TMC2 = u[3 * n_peptides + 18]

    MA1P_nbinder = u[3 * n_peptides + 19]
    MA1P_binder = u[3 * n_peptides + 20]
    MA2P_nbinder = u[3 * n_peptides + 21]
    MA2P_binder = u[3 * n_peptides + 22]
    MB1P_nbinder = u[3 * n_peptides + 23]
    MB1P_binder = u[3 * n_peptides + 24]
    MB2P_nbinder = u[3 * n_peptides + 25]
    MB2P_binder = u[3 * n_peptides + 26]
    MC1P_nbinder = u[3 * n_peptides + 27]
    MC1P_binder = u[3 * n_peptides + 28]
    MC2P_nbinder = u[3 * n_peptides + 29]
    MC2P_binder = u[3 * n_peptides + 30]

    MA1Ps = u[3 * n_peptides + 31:4 * n_peptides + 30]
    MA2Ps = u[4 * n_peptides + 31:5 * n_peptides + 30]
    MB1Ps = u[5 * n_peptides + 31:6 * n_peptides + 30]
    MB2Ps = u[6 * n_peptides + 31:7 * n_peptides + 30]
    MC1Ps = u[7 * n_peptides + 31:8 * n_peptides + 30]
    MC2Ps = u[8 * n_peptides + 31:9 * n_peptides + 30]

    TMA1P_nbinder = u[9 * n_peptides + 31]
    TMA1P_binder = u[9 * n_peptides + 32]
    TMA2P_nbinder = u[9 * n_peptides + 33]
    TMA2P_binder = u[9 * n_peptides + 34]
    TMB1P_nbinder = u[9 * n_peptides + 35]
    TMB1P_binder = u[9 * n_peptides + 36]
    TMB2P_nbinder = u[9 * n_peptides + 37]
    TMB2P_binder = u[9 * n_peptides + 38]
    TMC1P_nbinder = u[9 * n_peptides + 39]
    TMC1P_binder = u[9 * n_peptides + 40]
    TMC2P_nbinder = u[9 * n_peptides + 41]
    TMC2P_binder = u[9 * n_peptides + 42]

    TMA1Ps = u[9 * n_peptides + 43 : 10 * n_peptides + 42]
    TMA2Ps = u[10 * n_peptides + 43 : 11 * n_peptides + 42]
    TMB1Ps = u[11 * n_peptides + 43 : 12 * n_peptides + 42]
    TMB2Ps = u[12 * n_peptides + 43 : 13 * n_peptides + 42]
    TMC1Ps = u[13 * n_peptides + 43 : 14 * n_peptides + 42]
    TMC2Ps = u[14 * n_peptides + 43 : 15 * n_peptides + 42]

    MeA1P_nbinder = u[15 * n_peptides + 43]
    MeA2P_nbinder = u[15 * n_peptides + 44]
    MeB1P_nbinder = u[15 * n_peptides + 45]
    MeB2P_nbinder = u[15 * n_peptides + 46]
    MeC1P_nbinder = u[15 * n_peptides + 47]
    MeC2P_nbinder = u[15 * n_peptides + 48]

    MeA1P_binder = u[15 * n_peptides + 49]
    MeA2P_binder = u[15 * n_peptides + 50]
    MeB1P_binder = u[15 * n_peptides + 51]
    MeB2P_binder = u[15 * n_peptides + 52]
    MeC1P_binder = u[15 * n_peptides + 53]
    MeC2P_binder = u[15 * n_peptides + 54]

    MeA1Ps = u[15 * n_peptides + 55 : 16 * n_peptides + 54]
    MeA2Ps = u[16 * n_peptides + 55 : 17 * n_peptides + 54]
    MeB1Ps = u[17 * n_peptides + 55 : 18 * n_peptides + 54]
    MeB2Ps = u[18 * n_peptides + 55 : 19 * n_peptides + 54]
    MeC1Ps = u[19 * n_peptides + 55 : 20 * n_peptides + 54]
    MeC2Ps = u[20 * n_peptides + 55 : 21 * n_peptides + 54]

    MeA1 = u[21 * n_peptides + 55]
    MeA2 = u[21 * n_peptides + 56]
    MeB1 = u[21 * n_peptides + 57]
    MeB2 = u[21 * n_peptides + 58]
    MeC1 = u[21 * n_peptides + 59]
    MeC2 = u[21 * n_peptides + 60]


    # Differential equations for P_Cs
    du[1:n_peptides] .= gPs .- b_TAPs .* TAP .* P_Cs .- d_cyt .* P_Cs .+ u_TAP .* TPs .- cytamin_out .* P_Cs .+ cytamin_in .* vcat([0.0], P_Cs[1:end-1])

    # Differential equations for P_selfC
    du[n_peptides + 1] = g_self_cyt - b_TAP_self * TAP * P_selfC - d_cyt * P_selfC + u_TAP * TPself

    # Differential equation for TAP
    du[n_peptides + 2] = (k_TAP + u_TAP) * (sum(TPs) + TPself) - (b_TAP_self * P_selfC + dot(b_TAPs, P_Cs)) * TAP

    # Differential equations for TPs
    du[n_peptides + 3:2 * n_peptides + 2] .= b_TAPs .* TAP .* P_Cs .- (k_TAP + u_TAP) .* TPs
    
    du[2 * n_peptides + 3] = b_TAP_self * P_selfC * TAP - (k_TAP + u_TAP) * TPself

    du[2 * n_peptides + 4] = (1 - rho_b) * k_TAP * TPself - (b_A1 * MA1 + b_A2 * MA2 + b_B1 * MB1 + b_B2 * MB2 + b_C1 * MC1 + b_C2 * MC2) * P_nbinder + u_nbinder * (MA1P_nbinder + MA2P_nbinder + MB1P_nbinder + MB2P_nbinder + MC1P_nbinder + MC2P_nbinder) + q * u_nbinder * (TMA1P_nbinder + TMA2P_nbinder + TMB1P_nbinder + TMB2P_nbinder + TMC1P_nbinder + TMC2P_nbinder) - c * (TMA1 + TMA2 + TMB1 + TMB2 + TMC1 + TMC2) * P_nbinder - dP * P_nbinder

    du[2 * n_peptides + 5] = rho_b * k_TAP * TPself - (b_A1 * MA1 + b_A2 * MA2 + b_B1 * MB1 + b_B2 * MB2 + b_C1 * MC1 + b_C2 * MC2) * P_binder + u_binder * (MA1P_binder + MA2P_binder + MB1P_binder + MB2P_binder + MC1P_binder + MC2P_binder) + q * u_binder * (TMA1P_binder + TMA2P_binder + TMB1P_binder + TMB2P_binder + TMC1P_binder + TMC2P_binder) - c * (TMA1 + TMA2 + TMB1 + TMB2 + TMC1 + TMC2) * P_binder - dP * P_binder

    # Ps
    du[2 * n_peptides + 6 : 3 * n_peptides + 5] .= k_TAP .* TPs .- (b_A1 .* MA1 .+ b_A2 .* MA2 .+ b_B1 .* MB1 .+ b_B2 .* MB2 .+ b_C1 .* MC1 .+ b_C2 .* MC2) .* Ps .+ us_A1 .* MA1Ps .+ us_A2 .* MA2Ps .+ us_B1 .* MB1Ps .+ us_B2 .* MB2Ps .+ us_C1 .* MC1Ps .+ us_C2 .* MC2Ps .+ q .* (us_A1 .* TMA1Ps .+ us_A2 .* TMA2Ps .+ us_B1 .* TMB1Ps .+ us_B2 .* TMB2Ps .+ us_C1 .* TMC1Ps .+ us_C2 .* TMC2Ps) .- c .* (TMA1 .+ TMA2 .+ TMB1 .+ TMB2 .+ TMC1 .+ TMC2) .* Ps .- dP .* Ps .- E0 * (erap1_kcat_out .* Ps / Km) ./ (1 .+ sum(Ps) / Km .+ (P_nbinder + P_binder) / Km) .+ E0 * (erap1_kcat_in .* vcat([0.0], Ps[1:end-1]) / Km) ./ (1 .+ sum(Ps) / Km .+ (P_nbinder + P_binder) / Km)

    # Differential equations for MA1 --> MC2
    du[3 * n_peptides + 6] = gM_A1 + u_nbinder * MA1P_nbinder + u_binder * MA1P_binder + dot(us_A1, MA1Ps) + uT * TMA1 - b_A1 * (P_binder + P_nbinder + sum(Ps)) * MA1 - dM * MA1 - bT * T * MA1
    du[3 * n_peptides + 7] = gM_A2 + u_nbinder * MA2P_nbinder + u_binder * MA2P_binder + dot(us_A2, MA2Ps) + uT * TMA2 - b_A2 * (P_binder + P_nbinder + sum(Ps)) * MA2 - dM * MA2 - bT * T * MA2
    du[3 * n_peptides + 8] = gM_B1 + u_nbinder * MB1P_nbinder + u_binder * MB1P_binder + dot(us_B1, MB1Ps) + uT * TMB1 - b_B1 * (P_binder + P_nbinder + sum(Ps)) * MB1 - dM * MB1 - bT * T * MB1
    du[3 * n_peptides + 9] = gM_B2 + u_nbinder * MB2P_nbinder + u_binder * MB2P_binder + dot(us_B2, MB2Ps) + uT * TMB2 - b_B2 * (P_binder + P_nbinder + sum(Ps)) * MB2 - dM * MB2 - bT * T * MB2
    du[3 * n_peptides + 10] = gM_C1 + u_nbinder * MC1P_nbinder + u_binder * MC1P_binder + dot(us_C1, MC1Ps) + uT * TMC1 - b_C1 * (P_binder + P_nbinder + sum(Ps)) * MC1 - dM * MC1 - bT * T * MC1
    du[3 * n_peptides + 11] = gM_C2 + u_nbinder * MC2P_nbinder + u_binder * MC2P_binder + dot(us_C2, MC2Ps) + uT * TMC2 - b_C2 * (P_binder + P_nbinder + sum(Ps)) * MC2 - dM * MC2 - bT * T * MC2

    # T
    du[3 * n_peptides + 12] = uT * (TMA1 + TMA2 + TMB1 + TMB2 + TMC1 + TMC2) + gT + uT * v * (TMA1P_binder + TMA1P_nbinder + sum(TMA1Ps) + TMA2P_binder + TMA2P_nbinder + sum(TMA2Ps) + TMB1P_binder + TMB1P_nbinder + sum(TMB1Ps) + TMB2P_binder + TMB2P_nbinder + sum(TMB2Ps) + TMC1P_binder + TMC1P_nbinder + sum(TMC1Ps) + TMC2P_binder + TMC2P_nbinder + sum(TMC2Ps)) - (bT * (MA1 + MA2 + MB1 + MB2 + MC1 + MC2) + dT) * T

    # TMA1 --> TMC2
    du[3 * n_peptides + 13] = bT * T * MA1 + q * (u_nbinder * TMA1P_nbinder + u_binder * TMA1P_binder + dot(us_A1, TMA1Ps)) - (uT + c * (P_binder + P_nbinder + sum(Ps))) * TMA1
    du[3 * n_peptides + 14] = bT * T * MA2 + q * (u_nbinder * TMA2P_nbinder + u_binder * TMA2P_binder + dot(us_A2, TMA2Ps)) - (uT + c * (P_binder + P_nbinder + sum(Ps))) * TMA2
    du[3 * n_peptides + 15] = bT * T * MB1 + q * (u_nbinder * TMB1P_nbinder + u_binder * TMB1P_binder + dot(us_B1, TMB1Ps)) - (uT + c * (P_binder + P_nbinder + sum(Ps))) * TMB1
    du[3 * n_peptides + 16] = bT * T * MB2 + q * (u_nbinder * TMB2P_nbinder + u_binder * TMB2P_binder + dot(us_B2, TMB2Ps)) - (uT + c * (P_binder + P_nbinder + sum(Ps))) * TMB2
    du[3 * n_peptides + 17] = bT * T * MC1 + q * (u_nbinder * TMC1P_nbinder + u_binder * TMC1P_binder + dot(us_C1, TMC1Ps)) - (uT + c * (P_binder + P_nbinder + sum(Ps))) * TMC1
    du[3 * n_peptides + 18] = bT * T * MC2 + q * (u_nbinder * TMC2P_nbinder + u_binder * TMC2P_binder + dot(us_C2, TMC2Ps)) - (uT + c * (P_binder + P_nbinder + sum(Ps))) * TMC2

    # MA1P_nbinder --> MC2P_binder
    du[3 * n_peptides + 19] = b_A1 * MA1 * P_nbinder - (u_nbinder + e) * MA1P_nbinder + uT * v * TMA1P_nbinder
    du[3 * n_peptides + 20] = b_A1 * MA1 * P_binder - (u_binder + e) * MA1P_binder + uT * v * TMA1P_binder
    du[3 * n_peptides + 21] = b_A2 * MA2 * P_nbinder - (u_nbinder + e) * MA2P_nbinder + uT * v * TMA2P_nbinder
    du[3 * n_peptides + 22] = b_A2 * MA2 * P_binder - (u_binder + e) * MA2P_binder + uT * v * TMA2P_binder
    du[3 * n_peptides + 23] = b_B1 * MB1 * P_nbinder - (u_nbinder + e) * MB1P_nbinder + uT * v * TMB1P_nbinder
    du[3 * n_peptides + 24] = b_B1 * MB1 * P_binder - (u_binder + e) * MB1P_binder + uT * v * TMB1P_binder
    du[3 * n_peptides + 25] = b_B2 * MB2 * P_nbinder - (u_nbinder + e) * MB2P_nbinder + uT * v * TMB2P_nbinder
    du[3 * n_peptides + 26] = b_B2 * MB2 * P_binder - (u_binder + e) * MB2P_binder + uT * v * TMB2P_binder
    du[3 * n_peptides + 27] = b_C1 * MC1 * P_nbinder - (u_nbinder + e) * MC1P_nbinder + uT * v * TMC1P_nbinder
    du[3 * n_peptides + 28] = b_C1 * MC1 * P_binder - (u_binder + e) * MC1P_binder + uT * v * TMC1P_binder
    du[3 * n_peptides + 29] = b_C2 * MC2 * P_nbinder - (u_nbinder + e) * MC2P_nbinder + uT * v * TMC2P_nbinder
    du[3 * n_peptides + 30] = b_C2 * MC2 * P_binder - (u_binder + e) * MC2P_binder + uT * v * TMC2P_binder

    # MA1Ps --> MC2Ps
    du[3 * n_peptides + 31 : 4 * n_peptides + 30] .= b_A1 .* MA1 .* Ps .- (us_A1 .+ e) .* MA1Ps .+ uT .* v .* TMA1Ps
    du[4 * n_peptides + 31 : 5 * n_peptides + 30] .= b_A2 .* MA2 .* Ps .- (us_A2 .+ e) .* MA2Ps .+ uT .* v .* TMA2Ps
    du[5 * n_peptides + 31 : 6 * n_peptides + 30] .= b_B1 .* MB1 .* Ps .- (us_B1 .+ e) .* MB1Ps .+ uT .* v .* TMB1Ps
    du[6 * n_peptides + 31 : 7 * n_peptides + 30] .= b_B2 .* MB2 .* Ps .- (us_B2 .+ e) .* MB2Ps .+ uT .* v .* TMB2Ps
    du[7 * n_peptides + 31 : 8 * n_peptides + 30] .= b_C1 .* MC1 .* Ps .- (us_C1 .+ e) .* MC1Ps .+ uT .* v .* TMC1Ps
    du[8 * n_peptides + 31 : 9 * n_peptides + 30] .= b_C2 .* MC2 .* Ps .- (us_C2 .+ e) .* MC2Ps .+ uT .* v .* TMC2Ps
    
    # TMPA1_nbinder --> TMPC2_binder
    du[9 * n_peptides + 31] = c * TMA1 * P_nbinder - (q * u_nbinder + uT * v) * TMA1P_nbinder
    du[9 * n_peptides + 32] = c * TMA1 * P_binder - (q * u_binder + uT * v) * TMA1P_binder
    du[9 * n_peptides + 33] = c * TMA2 * P_nbinder - (q * u_nbinder + uT * v) * TMA2P_nbinder
    du[9 * n_peptides + 34] = c * TMA2 * P_binder - (q * u_binder + uT * v) * TMA2P_binder
    du[9 * n_peptides + 35] = c * TMB1 * P_nbinder - (q * u_nbinder + uT * v) * TMB1P_nbinder
    du[9 * n_peptides + 36] = c * TMB1 * P_binder - (q * u_binder + uT * v) * TMB1P_binder
    du[9 * n_peptides + 37] = c * TMB2 * P_nbinder - (q * u_nbinder + uT * v) * TMB2P_nbinder
    du[9 * n_peptides + 38] = c * TMB2 * P_binder - (q * u_binder + uT * v) * TMB2P_binder
    du[9 * n_peptides + 39] = c * TMC1 * P_nbinder - (q * u_nbinder + uT * v) * TMC1P_nbinder
    du[9 * n_peptides + 40] = c * TMC1 * P_binder - (q * u_binder + uT * v) * TMC1P_binder
    du[9 * n_peptides + 41] = c * TMC2 * P_nbinder - (q * u_nbinder + uT * v) * TMC2P_nbinder
    du[9 * n_peptides + 42] = c * TMC2 * P_binder - (q * u_binder + uT * v) * TMC2P_binder

    # TMA1Ps --> TMC2Ps
    du[9 * n_peptides + 43 : 10 * n_peptides + 42] .= c .* TMA1 .* Ps .- (q .* us_A1 .+ uT .* v) .* TMA1Ps
    du[10 * n_peptides + 43 : 11 * n_peptides + 42] .= c .* TMA2 .* Ps .- (q .* us_A2 .+ uT .* v) .* TMA2Ps
    du[11 * n_peptides + 43 : 12 * n_peptides + 42] .= c .* TMB1 .* Ps .- (q .* us_B1 .+ uT .* v) .* TMB1Ps
    du[12 * n_peptides + 43 : 13 * n_peptides + 42] .= c .* TMB2 .* Ps .- (q .* us_B2 .+ uT .* v) .* TMB2Ps
    du[13 * n_peptides + 43 : 14 * n_peptides + 42] .= c .* TMC1 .* Ps .- (q .* us_C1 .+ uT .* v) .* TMC1Ps
    du[14 * n_peptides + 43 : 15 * n_peptides + 42] .= c .* TMC2 .* Ps .- (q .* us_C2 .+ uT .* v) .* TMC2Ps

    # MeA1P_nbinder --> MeC2P_nbinder
    du[15 * n_peptides + 43] = e * MA1P_nbinder - u_nbinder * MeA1P_nbinder
    du[15 * n_peptides + 44] = e * MA2P_nbinder - u_nbinder * MeA2P_nbinder
    du[15 * n_peptides + 45] = e * MB1P_nbinder - u_nbinder * MeB1P_nbinder
    du[15 * n_peptides + 46] = e * MB2P_nbinder - u_nbinder * MeB2P_nbinder
    du[15 * n_peptides + 47] = e * MC1P_nbinder - u_nbinder * MeC1P_nbinder
    du[15 * n_peptides + 48] = e * MC2P_nbinder - u_nbinder * MeC2P_nbinder

    # MeA1P_binder --> MeC2P_binder
    du[15 * n_peptides + 49] = e * MA1P_binder - u_binder * MeA1P_binder
    du[15 * n_peptides + 50] = e * MA2P_binder - u_binder * MeA2P_binder
    du[15 * n_peptides + 51] = e * MB1P_binder - u_binder * MeB1P_binder
    du[15 * n_peptides + 52] = e * MB2P_binder - u_binder * MeB2P_binder
    du[15 * n_peptides + 53] = e * MC1P_binder - u_binder * MeC1P_binder
    du[15 * n_peptides + 54] = e * MC2P_binder - u_binder * MeC2P_binder

    # MeA1Ps --> MeC2Ps
    du[15 * n_peptides + 55 : 16 * n_peptides + 54] .= e .* MA1Ps .- us_A1 .* MeA1Ps
    du[16 * n_peptides + 55 : 17 * n_peptides + 54] .= e .* MA2Ps .- us_A2 .* MeA2Ps
    du[17 * n_peptides + 55 : 18 * n_peptides + 54] .= e .* MB1Ps .- us_B1 .* MeB1Ps
    du[18 * n_peptides + 55 : 19 * n_peptides + 54] .= e .* MB2Ps .- us_B2 .* MeB2Ps
    du[19 * n_peptides + 55 : 20 * n_peptides + 54] .= e .* MC1Ps .- us_C1 .* MeC1Ps
    du[20 * n_peptides + 55 : 21 * n_peptides + 54] .= e .* MC2Ps .- us_C2 .* MeC2Ps

    # MeA1 --> MeC2
    du[21 * n_peptides + 55] = u_nbinder * MeA1P_nbinder + u_binder * MeA1P_binder + dot(us_A1, MeA1Ps) - dMe * MeA1
    du[21 * n_peptides + 56] = u_nbinder * MeA2P_nbinder + u_binder * MeA2P_binder + dot(us_A2, MeA2Ps) - dMe * MeA2
    du[21 * n_peptides + 57] = u_nbinder * MeB1P_nbinder + u_binder * MeB1P_binder + dot(us_B1, MeB1Ps) - dMe * MeB1
    du[21 * n_peptides + 58] = u_nbinder * MeB2P_nbinder + u_binder * MeB2P_binder + dot(us_B2, MeB2Ps) - dMe * MeB2
    du[21 * n_peptides + 59] = u_nbinder * MeC1P_nbinder + u_binder * MeC1P_binder + dot(us_C1, MeC1Ps) - dMe * MeC1
    du[21 * n_peptides + 60] = u_nbinder * MeC2P_nbinder + u_binder * MeC2P_binder + dot(us_C2, MeC2Ps) - dMe * MeC2

end


function endog_to_exog(eqm::Vector{Float64}, n_peptides::Int)
    """Takes in eqm solution of endogenous only model and returns startpoint for full model"""
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
        eqm[11],
        eqm[12],
        eqm[13],
        eqm[14],
        eqm[15],
        eqm[16],
        eqm[17],
        eqm[18],
        eqm[19],
        eqm[20],
        eqm[21],
        eqm[22],
        eqm[23],
        eqm[24],
        eqm[25],
        eqm[26],
        eqm[27],
        eqm[28],
        eqm[29],
        eqm[30],
        zeros(n_peptides),
        zeros(n_peptides),
        zeros(n_peptides),
        zeros(n_peptides),
        zeros(n_peptides),
        zeros(n_peptides),
        eqm[31],
        eqm[32],
        eqm[33],
        eqm[34],
        eqm[35],
        eqm[36],
        eqm[37],
        eqm[38],
        eqm[39],
        eqm[40],
        eqm[41],
        eqm[42],
        zeros(n_peptides),
        zeros(n_peptides),
        zeros(n_peptides),
        zeros(n_peptides),
        zeros(n_peptides),
        zeros(n_peptides),
        eqm[43],
        eqm[44],
        eqm[45],
        eqm[46],
        eqm[47],
        eqm[48],
        eqm[49],
        eqm[50],
        eqm[51],
        eqm[52],
        eqm[53],
        eqm[54],
        zeros(n_peptides),
        zeros(n_peptides),
        zeros(n_peptides),
        zeros(n_peptides),
        zeros(n_peptides),
        zeros(n_peptides),
        eqm[55],
        eqm[56],
        eqm[57],
        eqm[58],
        eqm[59],
        eqm[60],
    )
end

# read in calculated parameter values from Python scripts
gPs = 100.0 .* npzread(joinpath(input_dir, "gPs.npy"))
TAP_BAs = npzread(joinpath(input_dir, "TAP_BAs.npy"))
cytamin_in = npzread(joinpath(input_dir, "cytamin_in.npy"))
cytamin_out = npzread(joinpath(input_dir, "cytamin_out.npy"))
erap1_kcat_in = npzread(joinpath(input_dir, "erap1_kcat_in.npy"))
erap1_kcat_out = npzread(joinpath(input_dir, "erap1_kcat_out.npy"))
bs = 10 .^ npzread(joinpath(input_dir, "bERs.npy"))
us_A1 = bs[:, 1] .* 1e-3 .* v_ER .* npzread(joinpath(input_dir, "mhci_affinities_A1.npy"))
us_A2 = bs[:, 2] .* 1e-3 .* v_ER .* npzread(joinpath(input_dir, "mhci_affinities_A2.npy"))
us_B1 = bs[:, 3] .* 1e-3 .* v_ER .* npzread(joinpath(input_dir, "mhci_affinities_B1.npy"))
us_B2 = bs[:, 4] .* 1e-3 .* v_ER .* npzread(joinpath(input_dir, "mhci_affinities_B2.npy"))
us_C1 = bs[:, 5] .* 1e-3 .* v_ER .* npzread(joinpath(input_dir, "mhci_affinities_C1.npy"))
us_C2 = bs[:, 6] .* 1e-3 .* v_ER .* npzread(joinpath(input_dir, "mhci_affinities_C2.npy"))

b_TAPs = (u_TAP ./ (TAP_BAs .* v_TAP))

n_peptides::Int = 9 # we track 16mers --> 8mers

startp = zeros(60)
startp[2] = T0  # Remembering Julia uses 1-based indexing

# Solving the endogenous model to find approximate equilibrium point
tspan_endog = (0.0, 1_000_000.0)
t_endog = range(0.0, stop=200_000.0, length=1000)
outputs = zeros(size(gPs)[1], 6 * size(gPs)[2])

# solve for every row in gPs (corresponding to a new peptide)
@showprogress for i in 1:size(gPs, 1)

    # solve the endogenous problem (no non-self peptide)
    p_endog = (bs[i, 1], bs[i, 2], bs[i, 3], bs[i, 4], bs[i, 5], bs[i, 6])
    prob_endog = ODEProblem(endogenous_derivative!, startp, tspan_endog, p_endog)
    sol_endog = solve(prob_endog, Rodas4(), saveat=t_endog)

    # Solve for the equilibrium point
    x0 = sol_endog.u[end]
    exog_x0 = endog_to_exog(x0, n_peptides)

    # Define parameters
    p = (
        gPs[i, :],
        b_TAPs[i, :],
        cytamin_in[i, :],
        cytamin_out[i, :],
        us_A1[i, :],
        us_A2[i, :],
        us_B1[i, :],
        us_B2[i, :],
        us_C1[i, :],
        us_C2[i, :],
        erap1_kcat_in[i, :],
        erap1_kcat_out[i, :],
        bs[i, 1],
        bs[i, 2],
        bs[i, 3],
        bs[i, 4],
        bs[i, 5],
        bs[i, 6],
    )

    # Solving the exogenous model
    prob_exog = ODEProblem(derivative!, exog_x0, (0.0, 1_000_000.0), p)
    sol_exog = solve(prob_exog, TRBDF2(), abstol=1e-6, reltol=1e-3, saveat=1_000_000.0)
    outputs[i, :] = sol_exog.u[end][15 * n_peptides + 55 : 21 * n_peptides + 54]

end

# write the final pMHC levels to a numpy formatted file
npzwrite(joinpath(input_dir, "pmhc_levels.npy"), outputs)