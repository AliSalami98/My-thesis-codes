import numpy as np
import matplotlib.pyplot as plt
from data_filling_ss import data

def plot_and_print(
    y,
    theta,
    hin,
    a_Pout,
    a_mck_dot,
    a_meh_dot,
    a_mkr_dot,
    a_mrh_dot,
    a_mint_dot,
    a_mout_dot,
    a_W,
    a_Deltap,
    a_Deltapk,
    a_Deltapkr,
    a_Deltapr,
    a_Deltaphr,
    a_Deltaph,
    a_mdot,
    a_Te,
    a_Th,
    a_Tr,
    a_Tc,
    a_Tk,
    a_Tk_wall,
    a_Tr_wall,
    a_Th_wall,
    a_Vc,
    a_Ve,
    a_pc,
    a_pk,
    a_pr,
    a_ph,
    a_pe,
    a_Tout,
    a_hout,
    a_theta,
    a_Dc,
    a_Dk,
    a_Dkr,
    a_Dr,
    a_Dhr,
    a_Dh,
    a_De,
    a_mc,
    a_mk,
    a_mkr,
    a_mr,
    a_mhr,
    a_mh,
    a_me,
    a_vk,
    a_vkr,
    a_vr,
    a_vhr,
    a_vh,
    a_Qc,
    a_Qk,
    a_Qkr,
    a_Qr,
    a_Qhr,
    a_Qh,
    a_Qe,
    a_alpha,
    Pheat_ss,
    Pcool_ss,
    eff_ss,
    k
):
    print('T_out sim', np.mean(a_Tout))
    print('T_out real', data['Tout [K]'][k])

    print("mdot sim [g/s]", (np.mean(a_mout_dot) + np.mean(a_mint_dot))/2)
    print("mdot real [g/s]", data['mdot [g/s]'][k])

    print('Tout2 [°C]', np.mean(a_Tout) - 273.15)
    print("Average mass flow rate[g/s]", np.mean(a_mdot) * 10**3)

    print("Average power output [W]", np.mean(a_Pout))
    print("Pout 2 [W]", np.mean(a_mdot)*(np.mean(a_hout) - hin))
    print("Qc [W]", np.mean(a_Qc))
    print("Qk [W]", np.mean(a_Qk))
    # print("Qkr [W]", np.mean(a_Qkr))
    print("Qr [W]", np.mean(a_Qr))
    # print("Qhr [W]", np.mean(a_Qhr))
    print("Qh [W]", np.mean(a_Qh))
    print("Qe [W]", np.mean(a_Qe))
    print("alpha", np.mean(a_alpha))
    print("The average Work done by motor [W]", np.mean(a_W))
    print("pressure drop[bar]", np.mean(np.abs(a_Deltap)))
    print("Temperature of CO2 in E [°C]", np.mean(a_Te) - 273.15)
    print("Temperature of CO2 in H [°C]", np.mean(a_Th) - 273.15)
    print("Temperature of CO2 in C [°C]", np.mean(a_Tc) - 273.15)
    print("Temperature of CO2 in K [°C]", np.mean(a_Tk) - 273.15)
    print("Density of CO2 in C [kg/m^3]", np.mean(a_Dc))

    print("Pheat [W]", Pheat_ss)
    print("Pcool [W]", Pcool_ss)
    print("Efficiency 1", eff_ss)

    theta = theta * 180 / np.pi
    
    plt.figure(1)
    plt.plot(a_Vc, a_pc,color='b', label='compression space')
    plt.plot(a_Ve, a_pe,color='r', label='expansion space')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Volume [$cm^3$]", fontsize = 12)
    plt.ylabel("Pressure [bar]", fontsize = 12)
    plt.legend()
    plt.grid()

    plt.figure(2)
    plt.plot(a_theta, a_Tc, color='b', label = "Compression")
    plt.plot(a_theta, a_Tk, color='cyan', label = "Cooler")
    plt.plot(a_theta, a_Tr, color='g', label = "Regenerator")
    plt.plot(a_theta, a_Th, color='orange', label = "Heater")
    plt.plot(a_theta, a_Te, color='r', label = "Expansion")
    # plt.plot(a_theta, a_Tk_wall, label = "Cooler wall")
    # plt.plot(a_theta, a_Tr_wall, color='darkgreen', label = "Regenerator wall")
    # plt.plot(a_theta, a_Th_wall, color='m', label = "Heater wall")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("theta [°]", fontsize=12)
    plt.ylabel("Temperature [K]", fontsize=12)
    plt.legend()
    plt.grid()

    plt.figure(3)
    plt.plot(a_theta, a_mc, color='b', label = "Compression")
    plt.plot(a_theta, a_mk, color='cyan', label = "Cooler")
    plt.plot(a_theta, a_mr, color='g', label = "Regenerator")
    plt.plot(a_theta, a_mh, color='orange', label = "Heater")
    plt.plot(a_theta, a_me, color='r', label = "Expansion")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Theta [°]", fontsize=12)
    plt.ylabel("Mass [g]", fontsize=12)
    plt.legend()
    plt.grid()

    plt.figure(4)
    plt.plot(a_theta, a_mint_dot, label = "In")
    plt.plot(a_theta, a_mout_dot, label = "Out")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("theta [°]", fontsize=12)
    plt.ylabel("Mass flow rate [g/s]", fontsize=12)
    plt.legend()
    plt.grid()

    plt.figure(5)
    plt.plot(a_theta, a_pc, color='b', label = "Compression")
    plt.plot(a_theta, a_pk, color='cyan', label = "Cooler")
    plt.plot(a_theta, a_pr, color='g', label = "Regenerator")
    plt.plot(a_theta, a_ph, color='orange', label = "Heater")
    plt.plot(a_theta, a_pe, color='r', label = "Expansion")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("theta [°]", fontsize=12)
    plt.ylabel("pressure [pa]", fontsize=12)
    plt.legend()
    plt.grid()

    plt.figure(6)
    plt.plot(a_theta, a_mck_dot, label = "ck")
    plt.plot(a_theta, a_meh_dot, label = "eh")
    plt.plot(a_theta, a_mkr_dot, label = "kr")
    plt.plot(a_theta, a_mrh_dot, label = "rh")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("theta [°]", fontsize=12)
    plt.ylabel("Mass flow rate [g/s]", fontsize=12)
    plt.legend()
    plt.grid()

    plt.figure(7)
    plt.plot(
        a_theta,
        a_Qc,
        a_theta,
        a_Qk,
        a_theta,
        a_Qkr,
        a_theta,
        a_Qr,
        a_theta,
        a_Qhr,
        a_theta,
        a_Qh,
        a_theta,
        a_Qe,
        a_theta,
        a_W,
        a_theta,
        a_Pout,
    )
    plt.xlabel("Theta [°]")
    plt.ylabel("Heat flow[W]")
    plt.legend([r"Qc", r"Qk", r"Qkr", r"Qr", r"Qhr", r"Qh", r"Qe", r"W", "P"])
    plt.grid()

    plt.figure(8)
    plt.plot(a_theta, a_Dc, color='b', label = "Compression")
    plt.plot(a_theta, a_Dk, color='cyan', label = "Cooler")
    plt.plot(a_theta, a_Dr, color='g', label = "Regenerator")
    plt.plot(a_theta, a_Dh, color='orange', label = "Heater")
    plt.plot(a_theta, a_De, color='r', label = "Expansion")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("theta [°]", fontsize=12)
    plt.ylabel("density [kg/m^3]", fontsize=12)
    plt.legend()
    plt.grid()
    plt.show()

    # plt.figure(2)
    # plt.plot(a_theta, a_Dc, a_theta, a_Dk, a_theta, a_Dkr, a_theta, a_Dr, a_theta, a_Dhr, a_theta, a_Dh, a_theta, a_De)
    # plt.xlabel("Theta [°]")
    # plt.ylabel("Density [kg/m^3]")
    # plt.legend([r"Dc", r"Dk", r"Dkr", r"Dr", r"Dhr", r"Dh", r"De"])
    # plt.grid()

    # plt.figure(3)
    # plt.plot(theta, a_pc, theta, a_pe)
    # plt.xlabel("Theta [°]")
    # plt.ylabel("pressure of CO2[pa]")
    # plt.legend([r"pc", r"pe"])
    # plt.grid()
    # plt.figure(6)
    # plt.plot(a_theta, a_Deltap)
    # plt.xlabel("theta [°]")
    # plt.ylabel("pressure difference [bar]")
    # plt.legend([r"Dp"])
    # plt.grid()

    # plt.figure(5)
    # plt.plot(a_theta, a_eff)
    # plt.xlabel("theta [°]")
    # plt.ylabel("efficiency [%]")
    # # plt.legend([r""])
    # plt.grid()

    # plt.figure(6)
    # plt.plot(theta, a_Deltapk, theta, a_Deltapkr, theta, a_Deltapr, theta, a_Deltaphr, theta, a_Deltaph)
    # plt.xlabel("theta [°]")
    # plt.ylabel("pressure drops [bar]")
    # plt.legend([r"k", r"kr", r"r", r"hr", r"h"])
    # plt.grid()

    # plt.figure(8)
    # plt.plot(theta, a_vk, theta, a_vkr, theta, a_vr, theta, a_vhr, theta, a_vh)
    # plt.xlabel("Theta [°]")
    # plt.ylabel("mass of CO2[kg]")
    # plt.legend([r"vk", r"vkr", r"vr", r"vhr", r"vh"])
    # plt.grid()

    # plt.figure(3)
    # plt.plot(a_theta, a_mint_dot, a_theta, a_mout_dot)
    # plt.xlabel("theta [°]")
    # plt.ylabel("mdot [kg/s]")
    # plt.legend([r"mint_dot", r"mout_dot"])
    # plt.grid()
    
    # plt.figure(4)
    # plt.plot(a_theta, a_mck_dot, a_theta, a_meh_dot)
    # plt.xlabel("theta [°]")
    # plt.ylabel("mdot [kg/s]")
    # plt.legend([r"mck_dot", r"meh_dot"])
    # plt.grid()

    # plt.figure(8)
    # plt.plot(a_theta, a_mc_dot, a_theta, a_me_dot, a_theta, a_mint_dot, a_theta, a_mout_dot)
    # plt.xlabel("Theta [°]")
    # plt.ylabel("mass flow rate of CO2[kg/s]")
    # plt.legend([r"mc_dot", r"me_dot", r"mint_dot", r"mout_dot"])
    # plt.grid()