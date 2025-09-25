import numpy as np

def compute_cycle_inputs(
    *,
    # time step and actuator commands
    dt: float,
    # previous  states (for first-order lag)
    hpev: float,
    lpev: float,
    # operating inputs
    Tw_in: float,
    Theater1: float,
    Theater2: float,
    omega1: float,
    omega2: float,
    omega3: float,
    omegab: float,
    # model/context objects
    config,
    CP,
    get_state,
    scaler_X,
    models_GP,       # list of regressors for stage-1: [m_dot, Pcooler1, TTC1_out] in your order
    scalers_y,       # list of inverse scalers aligned with models_GP
    model1_1_LR,
    model1_2_LR,
    model1_3_LR,
    model2_1_LR,
    model2_2_LR,
    model2_3_LR,
    model3_1_LR,
    model3_2_LR,
    model3_3_LR,
):
    # ---------- 2) Valve coefficients ----------
    f_hpv = 7.5e-9 * hpev #min(max(4.765e-9 * hpev + 6.3625e-8, 1e-7), 4.5e-7) #
    f_lpv = 5e-9 * min(lpev, 60) #min(max(6.023e-9 * lpev - 1e-7, 0.2e-7), 3e-7) #

    # ---------- 3) Stage 1 GP predictions ----------
    Pr1 = np.clip(config.pbuff1 / config.pe, 1.02, 1.65)
    Tr1 = Theater1 / Tw_in
    pcharged1 = np.sqrt(config.pbuff1 * config.pe)

    test1 = np.array([[config.pe, config.pbuff1, omega1, Theater1, Tw_in, config.Tihx2]])  # shape (1, 6)
    test1_scaled = scaler_X.transform(test1)

    predictions_GP = [model.predict(test1_scaled) for model in models_GP]
    y_pred_real_GP = [
        scaler.inverse_transform(np.array(pred).reshape(-1, 1)).flatten()[0]
        for pred, scaler in zip(predictions_GP, scalers_y)
    ]

    a1_T = np.array([[Pr1, Theater1, Tw_in, omega1]])
    me_out_dot = 0.8 * y_pred_real_GP[0]  #  1e-3 * model1_1_LR.predict(a1_T)[0] # 
    Pcooler1_pred = y_pred_real_GP[1] #model1_3_LR.predict(a1_T)[0] #
    TTC1_out = y_pred_real_GP[2] #model1_2_LR.predict(a1_T)[0] #

    # mass out of compressor 1 (HPV) using throttling relation
    mc_out_dot = f_hpv * np.sign(config.pc - config.pft) * np.sqrt(2 * config.Dc[-1] * np.abs(config.pc - config.pft))

    # ---------- 4) Thermo states around TC1/buffer1 ----------
    state = get_state(CP.PT_INPUTS, config.pbuff1, TTC1_out)
    hTC1_out = state.hmass()
    hbuff1_in = max(hTC1_out, config.hbuff1_v)
    state = get_state(CP.HmassP_INPUTS, hbuff1_in, config.pbuff1)
    Tbuff1_in = state.T()
    Dbuff1_in = state.rhomass()

    # ---------- 5) Stage 3 LR predictions (mc_in_dot, Tc_in) ----------
    Pr3 = config.pc / config.pbuff2
    a3_T = np.array([[Pr3, Theater2, Tw_in, omega3]])

    Tc_in = model3_2_LR.predict(a3_T)[0]
    mc_in_dot = 1e-3 * model3_1_LR.predict(a3_T)[0]

    state = get_state(CP.PT_INPUTS, config.pc, Tc_in)
    hc_in = state.hmass()
    Dc_in = state.rhomass()

    # ---------- 6) Flash tank flows (liquid/vapor) ----------
    if config.hft_l > config.hft:
        mft_l_dot = f_lpv * np.sign(config.pft - config.pe) * np.sqrt(2 * config.Dft * np.abs(config.pft - config.pe))
        mft_v_dot = 1e-6
    if config.hft > config.hft_v:
        mft_l_dot = 1e-6
        mft_v_dot = 1e-6 * np.sign(config.pft - config.pbuff1) * np.sqrt(2 * config.Dft * np.abs(config.pft - config.pbuff1))
    else:
        mft_l_dot = f_lpv * np.sign(config.pft - config.pe) * np.sqrt(2 * config.Dft_l * np.abs(config.pft - config.pe))
        mft_v_dot = 1e-6 * np.sign(config.pft - config.pbuff1) * np.sqrt(2 * config.Dft_v * np.abs(config.pft - config.pbuff1))

    # ---------- 7) TC2 inlet mix (vap from FT + buffer1 out) ----------
    eps = 1e-12
    mix_den = max(mft_v_dot + me_out_dot, eps)
    hTC2_in = np.clip(
        (mft_v_dot * config.hft_v + me_out_dot * config.hbuff1) / mix_den,
        config.hft_v + 5e3,
        config.hbuff1,
    )

    state = get_state(CP.HmassP_INPUTS, hTC2_in, config.pbuff1)
    TTC2_in = state.T()

    # ---------- 8) Stage 2 predictions (LR + GP-style features) ----------
    Pr2 = config.pbuff2 / config.pbuff1
    Tr2 = Theater2 / Tw_in
    pcharged2 = np.sqrt(config.pbuff2 * config.pbuff1)

    test2 = np.array([[config.pbuff1, config.pbuff2, omega2, Theater2, Tw_in, TTC2_in]])  # (1,6)
    test2_scaled = scaler_X.transform(test2)

    # You computed GP2 but then used LR in your final values; we keep the LR outputs as in your code.
    predictions_GP2 = [model.predict(test2_scaled) for model in models_GP]
    y_pred_real_GP2 = [
        scaler.inverse_transform(np.array(pred).reshape(-1, 1)).flatten()[0]
        for pred, scaler in zip(predictions_GP2, scalers_y)
    ]

    a2_T = np.array([[Pr2, Theater2, Tw_in, omega2]])
    mtc2_dot = 0.8 * 1e-3 * model2_1_LR.predict(a2_T)[0] #y_pred_real_GP2[0] #
    Ttc2_out = model2_2_LR.predict(a2_T)[0] #y_pred_real_GP2[2] #

    me_in_dot = mft_l_dot

    Pcooler2_pred = 0.9 * model2_3_LR.predict(a2_T)[0] #y_pred_real_GP2[1] #
    Pcooler3_pred = 0.9 * model3_3_LR.predict(a3_T)[0]



    return (
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
    )
