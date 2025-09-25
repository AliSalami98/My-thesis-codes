import numpy as np
import math

from utils import (
	get_state,
)
import config
from config import CP
from funcs import (
    init_evaporator_post,
    init_gas_cooler_post,
    init_buffer1_post,
    init_buffer2_post,
    init_internal_heat_exchanger_post,
    init_flash_tank_post,

    calc_evaporator,
    calc_gas_cooler,
    calc_buffer1,
    calc_buffer2,
    calc_internal_heat_exchanger,
    fhx_predict_Q
)

# import os
# import joblib
# GP_models_dir = 'ML_Models/GP_models'
# models_GP = [joblib.load(os.path.join(GP_models_dir, f'GP_model_{i+1}.pkl')) for i in range(3)]

# scalers_dir = 'ML_Models/scalers'
# scaler_X = joblib.load(os.path.join(scalers_dir, 'scaler_X.pkl'))
# scalers_y = [joblib.load(os.path.join(scalers_dir, f'scaler_y_{i+1}.pkl')) for i in range(3)]
# from ML_Models.compressor_PR import (
#     poly1_reg,
#     model1_PR,
#     poly2_reg,
#     model2_PR,
#     poly3_reg,
#     model3_PR,
# )
# from ML_Models.comps import (
#     model1_1_LR,
#     model1_2_LR,
#     model1_3_LR,
#     model2_1_LR,
#     model2_2_LR,
#     model2_3_LR,
#     model3_1_LR,
#     model3_2_LR,
#     model3_3_LR,
# )
# from PR_compressor2 import (
#     model1_comp2_PR,
#     model2_comp2_PR,
#     model3_comp2_PR,
# )
# from PR_compressor3 import (
#     model1_comp3_PR,
#     model2_comp3_PR,
#     model3_comp3_PR,
# )
def post_process(
    y,
    t,
    mw_dot,
    Tw_in,
    Tmpg_in,
    mmpg_dot,
    mc_in_dot,
    mc_out_dot,
    me_in_dot,
    me_out_dot,
    mft_l_dot,
    mft_v_dot,
    mtc2_dot,
    Ttc2_out,
    hc_in,
    Dc_in,
    hbuff1_in,
    Dbuff1_in,
    hTC1_out,
    hTC2_in,
    Pcooler1_pred,
    Pcooler2_pred,
    Pcooler3_pred,
    omegab
    ):

    a_t = []
    a_pc = []
    a_Tw_out = []
    
    a_Qe_conv = []
    a_Qmpg_conv = []
    a_Pevap = []

    a_Qc_conv = []
    a_Qc_w_conv = []
    a_Hpev = []
    a_j = []

    a_Tc_out = []
    a_hc_out = []
    a_pe = []
    a_Te_out = []
    a_he_out = []

    a_me_dot = []
    a_mihx2_dot = []
    a_mbuff1_dot = []
    a_mbuff2_dot = []
    a_mc_dot = []
    a_mihx1_dot = []
    a_Pgc = []
    a_Pevap = []
    a_Pihx1 = []
    a_Pihx2 = []
    a_Pbuff1 = []
    a_Pbuff2 = []
    a_COP = []
    a_Pheat_out = []
    a_Pcoolers = []
    a_Pbuffers = []
    a_mc_in_dot = []
    a_mbuff1_in_dot = []
    a_pft = []
    a_hft = []
    a_mc_out_dot = []
    a_mft_in_dot = []
    a_mft_out_dot = []
    a_me_in_dot = []
    a_me_out_dot = []
    a_mbuff1_out_dot = []
    a_pbuff1 = []
    a_pbuff2 = []
    a_mft_dot = []
    a_hihx1_out = []
    a_hihx2_out = []
    a_Tmpg_out = []
    a_Prec_total = []
    a_Pcomb = []
    a_SH = []
    for k in range(len(t)):

        j = int(np.floor(t[k]))
        a_j.append(j)
        pbuff1_in = config.pbuff1
        pbuff2_in = config.pbuff2
        pft_in = config.pft
        pc_in = config.pc
        pe_in = config.pe
        De_in = config.De[0]
        mw2_dot = 0.7 * mw_dot
        mw1_dot = 0.3 * mw_dot
        mw1_dot_1 = 0.5 * 0.3 * mw_dot
        mw1_dot_2 = mw1_dot_1

        state = get_state(CP.PT_INPUTS, pbuff2_in, Ttc2_out)
        htc2_out = state.hmass()
        hbuff2_in = max(htc2_out, config.hbuff2_v)
        state = get_state(CP.HmassP_INPUTS, hbuff2_in, pbuff2_in)
        Dbuff2_in = state.rhomass()


        hihx1_in = config.hc[-1]
        pihx1_in = config.pc
        state = get_state(CP.HmassP_INPUTS, hihx1_in, pihx1_in)
        Dihx1_in = state.rhomass()
        Tihx1_in = state.T()

        hihx2_in = config.he[-1]
        pihx2_in = config.pe
        state = get_state(CP.HmassP_INPUTS, hihx2_in, pihx2_in)
        Dihx2_in = state.rhomass()
        Tihx2_in = state.T()
        # mbuff2_in_dot =  10**(-3) * (5.66e-3 * omegab - 4.6e+1 * Pr1 + 1.23e-2 * omega1 + 50.3)
        hft_in = config.hihx1
        he_in = config.hft_l

        mft_in_dot = mc_out_dot
        mft_out_dot = mft_l_dot + mft_v_dot
        
        mbuff1_in_dot = mft_v_dot + me_out_dot
        mbuff1_out_dot = mc_in_dot

        mbuff2_in_dot = mtc2_dot
        mbuff2_out_dot = mc_in_dot

        mihx1_in_dot = mc_in_dot
        mihx1_out_dot = mihx1_in_dot

        mihx2_in_dot = me_out_dot
        mihx2_out_dot = mihx2_in_dot

        # print(Pcooler1_pred)
        init_gas_cooler_post(y, k, pc_in)
        calc_gas_cooler(mc_in_dot, mc_out_dot, Dc_in, mw2_dot, Tw_in)

        init_evaporator_post(y, k, pe_in, he_in, config.pbuff1)
        calc_evaporator(me_in_dot, me_out_dot, De_in, mmpg_dot, Tmpg_in)
        
        init_buffer1_post(y, k, pbuff1_in, config.pe)
        init_buffer2_post(y, k, pbuff2_in, config.pbuff1)
        init_internal_heat_exchanger_post(y, k, pihx1_in, pihx2_in, hihx1_in, hihx2_in)
        init_flash_tank_post(y, k, pft_in, config.pe)
        
        calc_buffer1(mbuff1_in_dot, mbuff1_out_dot, Dbuff1_in, mw1_dot_1, Tw_in, me_out_dot)
        calc_buffer2(mbuff2_in_dot, mbuff2_out_dot, Dbuff2_in, mw1_dot_2, Tw_in)
        calc_internal_heat_exchanger(mihx1_in_dot, mihx1_out_dot, Dihx1_in, mihx2_in_dot, mihx2_out_dot, Dihx2_in)
        
        a_Qe_conv.append(np.sum(config.Qe_conv))
        a_Qmpg_conv.append(np.sum(config.Qmpg_conv))
        a_Qc_conv.append(np.sum(config.Qc_conv))
        a_Qc_w_conv.append(np.sum(config.Qc_w_conv))

        a_pe.append(config.pe * 10**(-5))
        a_Te_out.append(config.Te[-1] - 273.15)
        a_pc.append(config.pc * 10**(-5))
        a_Tc_out.append(config.Tc[-1]- 273.15)
        a_he_out.append(config.he[-1])
        a_hc_out.append(config.hc[-1])

        a_me_dot.append(me_in_dot - me_out_dot)
        a_mihx2_dot.append(mihx2_in_dot - mihx2_out_dot)
        a_mbuff1_dot.append(mbuff1_in_dot - mbuff1_out_dot)
        a_mft_dot.append(mft_in_dot - mft_out_dot)
        a_mc_dot.append(mc_in_dot - mc_out_dot)
        a_mihx1_dot.append(mihx1_in_dot - mihx1_out_dot)
        a_t.append(t[k])

        a_Pgc.append(mw2_dot * config.cpw * (config.Tc_w[0] - Tw_in)) #mc_in_dot * hc_in - mc_out_dot * config.hc[-1]) #
        a_Pevap.append(mmpg_dot * config.cp_mpg * (Tmpg_in - config.Tmpg[0])) #me_out_dot * config.he[-1] - me_in_dot * he_in) #
        # a_Pihx1.append(mihx1_in_dot*(hihx1_in - config.hihx1[0]))
        # a_Pihx2.append(mihx2_in_dot*(config.hihx2[-1] - hihx2_in))
        a_Pbuff1.append(me_out_dot*(hbuff1_in - config.hbuff1))
        a_Pbuff2.append(mbuff2_in_dot*(hbuff2_in - config.hbuff2))

        mCH4_dot = (0.0022 * omegab - 2.5965) * 0.657/60000
        Pcomb = mCH4_dot * 50e6

        # Pcooler3_pred = min(Pcomb - Pcooler1_pred - Pcooler2_pred - me_out_dot * (hTC1_out - config.hihx2)- mtc2_dot * (htc2_out - hTC2_in) - mc_in_dot * (hc_in - config.hbuff2), Pcooler3_pred)
        # print(Pcooler3_pred)
        Tfume_in = 0.056 * omegab + 236.65
        Trwo_target = (a_Pbuff1[-1] + a_Pbuff2[-1] + Pcooler1_pred + Pcooler2_pred + Pcooler3_pred)/(config.cpw * mw1_dot) + Tw_in
        config.Trwo = Trwo_target #config.Trwo + 1/50 * (Trwo_target - config.Trwo)
        Tfhx_w_in = (mw2_dot * config.Tc_w[0] + mw1_dot * config.Trwo)/mw_dot
        mg_dot = 18.125 * mCH4_dot

        # 3) AU_fhx from your tuned correlation (note: correlation defined in °C)
        Tfhx_w_in_C = Tfhx_w_in - 273.15
        AU_fhx = 5.98e-4 * omegab + 0.687 * Tfhx_w_in_C - 16.34  # W/K
        AU_fhx = max(AU_fhx, 0.02)  # clip to non-negative
        Pfhx = 2 * (Tfume_in - Tfhx_w_in)
        Prec_max = Pcomb + 400 - (Pcooler1_pred + Pcooler2_pred + Pcooler3_pred + me_out_dot * (hTC1_out - config.hihx2) + mtc2_dot * (htc2_out - hTC2_in) + mc_in_dot * (hc_in - htc2_out))
        Pfhx = min(Pfhx, Prec_max)
        Tfhx_w_out = Pfhx/(mw_dot * config.cpw) + Tfhx_w_in
        # 4) Use LMTD closure to solve for Q, Tf_out, Tw_out (K, W)
        # out = fhx_predict_Q(
        #     Tf_in=Tfume_in, Tw_in=Tfhx_w_in,
        #     mdot_f=mg_dot,    cp_f=1100.0,   # adjust cp_f if you have a better value
        #     mdot_w=mw_dot,    cp_w=config.cpw,
        #     AU_fhx=AU_fhx,    F=1.0,         # or a correction factor
        #     tol_rel=1e-4, max_iter=100, under_relax=0.5
        # )

        # Pfhx   = out["Q"],                # W
        # Tfume_out = out["Tf_out"]          # K
        # Tfhx_w_out = out["Tw_out"]          # K
        
        Pheat_out_target = a_Pgc[-1] + a_Pbuff1[-1] + a_Pbuff2[-1] + Pcooler1_pred + Pcooler2_pred + Pcooler3_pred + Pfhx
        config.Pheating_out = Pheat_out_target #config.Pheating_out + 1/20 * (Pheat_out_target - config.Pheating_out)
        a_Tw_out.append(Tfhx_w_out - 273.15) #+ config.Pheating_out/(mw_dot * config.cpw) + Tw_in - 273.15)

        # a_Tw_out.append(Tw_rec_in + (Pfhx)/(mw_dot * config.cpw) - 273.15) #+ config.Pheating_out/(mw_dot * config.cpw) + Tw_in - 273.15)
        a_Tmpg_out.append(config.Tmpg[0] - 273.15) #- a_Pevap[-1]/(mmpg_dot * config.cpw) + Tmpg_in - 273.15)
        a_Pcoolers.append(Pcooler1_pred + Pcooler2_pred + Pcooler3_pred)
        a_Pbuffers.append(a_Pbuff1[-1] + a_Pbuff2[-1])
        a_Pcomb.append(Pcomb)
        a_Prec_total.append(Pcooler1_pred + Pcooler2_pred + Pcooler3_pred + Pfhx + a_Pbuff1[-1] + a_Pbuff2[-1])
        # print(Pfhx)
        a_Pheat_out.append(config.Pheating_out)
        state = get_state(CP.PQ_INPUTS, config.pe, 1)
        a_SH.append(config.Te[-1] - state.T())
        a_COP.append(config.Pheating_out/Pcomb)
        a_mc_in_dot.append(mc_in_dot)
        a_mc_out_dot.append(mc_out_dot)
        a_mft_in_dot.append(mft_in_dot)
        a_mft_out_dot.append(mft_out_dot)
        a_me_in_dot.append(me_in_dot)
        a_me_out_dot.append(me_out_dot)
        a_mbuff1_in_dot.append(mbuff1_in_dot)
        a_mbuff1_out_dot.append(mbuff1_out_dot)
        a_pbuff1.append(config.pbuff1 * 10**(-5))
        a_pbuff2.append(config.pbuff2 * 10**(-5))
        a_pft.append(config.pft * 10**(-5))
        a_hft.append(config.hft)
        a_hihx1_out.append(config.hihx1)
        a_hihx2_out.append(config.hihx2)

    return (
        a_t,
        a_Pgc,
        a_Pevap,
        a_Pihx1,
        a_Pihx2,
        a_Pbuff1,
        a_Pbuff2,
        Pfhx,
        a_me_dot,
        a_mihx2_dot,
        a_mbuff1_dot,
        a_mbuff2_dot,
        a_mc_dot,
        a_mihx1_dot,
        a_COP,
        a_Pheat_out,
        a_Pcoolers,
        a_Pbuffers,
        a_mc_in_dot,
        a_mbuff1_in_dot,
        a_pft,
        a_pc,
        a_pe,
        a_Tc_out,
        a_Te_out,
        a_hc_out,
        a_he_out,
        a_mc_out_dot,
        a_mft_in_dot,
        a_mft_out_dot,
        a_me_in_dot,
        a_me_out_dot,
        a_hft,
        a_mbuff1_out_dot,
        a_pbuff1,
        a_pbuff2,
        a_mft_dot,
        a_hihx1_out,
        a_hihx2_out,
        a_Tw_out,
        a_Tmpg_out,
        a_SH,
        a_Pcomb,
        a_Prec_total,
    )

