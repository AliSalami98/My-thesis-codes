import csv
import numpy as np

# Initialize lists
Tc1, Tk1, Tr1, Th1, Te1 = [], [], [], [], []
pc1, pk1, pr1, ph1, pe1 = [], [], [], [], []
Dc1, Dk1, Dr1, Dh1, De1 = [], [], [], [], []
mc1, mk1, mr1, mh1, me1 = [], [], [], [], []
mint_dot1, mout_dot1 = [], []
Vc1, Ve1, t1 = [], [], []

# --- Read CSV ---
with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\Model validation\compressor\cfd.csv') as csv_file:    
    csv_reader = csv.DictReader(csv_file, delimiter=';')
    for row in csv_reader:
        Tc1.append(float(row['Tc']))
        Tk1.append(float(row['Tk']))
        Tr1.append(float(row['Tr']))
        Th1.append(float(row['Th']))
        Te1.append(float(row['Te']))
        pc1.append(float(row['pc']) * 1e-5)
        pk1.append(float(row['pk']) * 1e-5)
        pr1.append(float(row['pr']) * 1e-5)
        ph1.append(float(row['ph']) * 1e-5)
        pe1.append(float(row['pe']) * 1e-5)
        Dc1.append(float(row['Dc']))
        Dk1.append(float(row['Dk']))
        Dr1.append(float(row['Dr']))
        Dh1.append(float(row['Dh']))
        De1.append(float(row['De']))
        mc1.append(float(row['mc']) * 1000)
        mk1.append(float(row['mk']) * 1000)
        mr1.append(float(row['mr']) * 1000)
        mh1.append(float(row['mh']) * 1000)
        me1.append(float(row['me']) * 1000)
        Ve1.append(float(row['Ve']) * 1e6)
        Vc1.append(float(row['Vc']) * 1e6)
        t1.append(float(row['Time2']))
        mint_dot1.append(float(row['mint_dot']) * 1000)
        mout_dot1.append(float(row['mout_dot']) * 1000)

# --- Convert all to numpy arrays ---
Tc1, Tk1, Tr1, Th1, Te1 = map(np.array, (Tc1, Tk1, Tr1, Th1, Te1))
pc1, pk1, pr1, ph1, pe1 = map(np.array, (pc1, pk1, pr1, ph1, pe1))
Dc1, Dk1, Dr1, Dh1, De1 = map(np.array, (Dc1, Dk1, Dr1, Dh1, De1))
mc1, mk1, mr1, mh1, me1 = map(np.array, (mc1, mk1, mr1, mh1, me1))
Vc1, Ve1, t1 = map(np.array, (Vc1, Ve1, t1))
mint_dot1, mout_dot1 = map(np.array, (mint_dot1, mout_dot1))

# --- Convert time to crank angle ---
theta1 = np.linspace(0, 360, len(t1))
