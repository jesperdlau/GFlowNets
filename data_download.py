import numpy as np
import requests
import os 

# the public url to objects available for download "https://storage.googleapis.com/design-bench"

# Change folder if needed
DATA_FOLDER = "GFlowNets/data/"
#DATA_FOLDER = "./data/"

def download(download_target, disk_target):
    response = requests.get(download_target, allow_redirects=True)
    valid_response = response.status_code < 400

    if valid_response:
        with open(DATA_FOLDER + disk_target, "wb") as file:
            file.write(response.content)
    return valid_response


### TF_BIND_8 ###

# TF_BIND_8_FILES = ['tf_bind_8-PITX2_T114P_R1/tf_bind_8-x-0.npy', 'tf_bind_8-NR2E3_R76W_R1/tf_bind_8-x-0.npy', 'tf_bind_8-NR2E3_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ESX1_K193R_R1/tf_bind_8-x-0.npy', 'tf_bind_8-OVOL2_D228E_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ARX_T333N_R2/tf_bind_8-x-0.npy', 'tf_bind_8-KLF11_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PITX2_R108H_R1/tf_bind_8-x-0.npy', 'tf_bind_8-NR2E3_R77Q_R1/tf_bind_8-x-0.npy', 'tf_bind_8-SIX6_T165A_R2/tf_bind_8-x-0.npy', 'tf_bind_8-NKX2-8_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-EGR2_E412K_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PROP1_R99Q_R1/tf_bind_8-x-0.npy', 'tf_bind_8-WT1_F392L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-VAX2_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-CRX_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-VSX1_G160D_R1/tf_bind_8-x-0.npy', 'tf_bind_8-WT1_R366L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-CRX_R90W_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ZNF655_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-NKX2-5_R161P_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ZNF655_S393L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX3_N47H_R1/tf_bind_8-x-0.npy', 'tf_bind_8-GFI1_L400F_R1/tf_bind_8-x-0.npy', 'tf_bind_8-POU4F3_K277R_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PHOX2B_R100L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-POU3F4_A312V_R1/tf_bind_8-x-0.npy', 'tf_bind_8-CRX_E80A_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PITX2_R130W_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX6_F258S_R1/tf_bind_8-x-0.npy', 'tf_bind_8-KLF1_R328H_R2/tf_bind_8-x-0.npy', 'tf_bind_8-VAX2_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-MSX2_R172H_R2/tf_bind_8-x-0.npy', 'tf_bind_8-POU6F2_E639K_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ZNF655_E327G_R2/tf_bind_8-x-0.npy', 'tf_bind_8-KLF1_E325K_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ARX_T333N_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PITX2_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-HOXC4_N178S_R1/tf_bind_8-x-0.npy', 'tf_bind_8-MSX2_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PHOX2B_R141Q_R1/tf_bind_8-x-0.npy', 'tf_bind_8-CRX_R41W_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ZNF200_S265Y_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX6_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-MSX2_P148H_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX6_R242T_R1/tf_bind_8-x-0.npy', 'tf_bind_8-POU4F3_L289F_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX4_REF_R3/tf_bind_8-x-0.npy', 'tf_bind_8-HESX1_E149K_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ARX_L343Q_R2/tf_bind_8-x-0.npy', 'tf_bind_8-HOXB7_K191R_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HOXB7_K191R_R2/tf_bind_8-x-0.npy', 'tf_bind_8-HESX1_N125S_R2/tf_bind_8-x-0.npy', 'tf_bind_8-BCL6_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-NR1H4_C144R_R1/tf_bind_8-x-0.npy', 'tf_bind_8-KLF1_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-POU3F4_R323G_R1/tf_bind_8-x-0.npy', 'tf_bind_8-SNAI2_T234I_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PHOX2B_R141G_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX4_R192S_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX7_P112L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-POU3F4_V322M_R1/tf_bind_8-x-0.npy', 'tf_bind_8-EGR2_D383Y_R1/tf_bind_8-x-0.npy', 'tf_bind_8-GFI1B_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-FOXC1_I126M_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX3_P50L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-VAX2_L139M_R2/tf_bind_8-x-0.npy', 'tf_bind_8-HOXB7_M190L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX3_G48R_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HESX1_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-CRX_R41Q_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PBX4_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ZNF655_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-VSX1_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HOXD13_S316C_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HOXD13_I322L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HOXB7_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-WT1_R394L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-POU3F4_A237G_R1/tf_bind_8-x-0.npy', 'tf_bind_8-NKX2-5_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ESX1_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PROP1_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX6_R128C_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HOXC4_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-FOXC1_S131L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-KLF1_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-NR2E3_G56R_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX3_Y90H_R1/tf_bind_8-x-0.npy', 'tf_bind_8-KLF1_R328H_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ZNF200_H322Y_R1/tf_bind_8-x-0.npy', 'tf_bind_8-POU4F3_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX6_P118R_R1/tf_bind_8-x-0.npy', 'tf_bind_8-POU6F2_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ISX_R83Q_R2/tf_bind_8-x-0.npy', 'tf_bind_8-PITX2_L100Q_R2/tf_bind_8-x-0.npy', 'tf_bind_8-PAX4_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PHOX2B_Q143R_R1/tf_bind_8-x-0.npy', 'tf_bind_8-KLF1_H299Y_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX4_R183C_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HESX1_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-HOXD13_R306W_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ARX_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-SIX6_H141N_R2/tf_bind_8-x-0.npy', 'tf_bind_8-FOXC1_F112S_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PITX2_L100Q_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX4_R192S_R2/tf_bind_8-x-0.npy', 'tf_bind_8-PAX3_R56L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-MSX2_R172H_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX6_R26G_R1/tf_bind_8-x-0.npy', 'tf_bind_8-NKX2-5_T178M_R1/tf_bind_8-x-0.npy', 'tf_bind_8-FOXC1_L130F_R1/tf_bind_8-x-0.npy', 'tf_bind_8-OVOL2_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-WT1_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PITX2_R108H_R2/tf_bind_8-x-0.npy', 'tf_bind_8-ARX_R332H_R1/tf_bind_8-x-0.npy', 'tf_bind_8-NKX2-5_K183E_R1/tf_bind_8-x-0.npy', 'tf_bind_8-POU3F4_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ARX_L343Q_R1/tf_bind_8-x-0.npy', 'tf_bind_8-NKX2-5_R190C_R1/tf_bind_8-x-0.npy', 'tf_bind_8-EGR2_R409W_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ARX_P353L_R2/tf_bind_8-x-0.npy', 'tf_bind_8-HOXC4_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-CRX_V66I_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HESX1_R160C_R1/tf_bind_8-x-0.npy', 'tf_bind_8-FOXC1_S82T_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX4_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-POU4F3_N316K_R1/tf_bind_8-x-0.npy', 'tf_bind_8-NKX2-8_A94T_R1/tf_bind_8-x-0.npy', 'tf_bind_8-VSX1_R166Q_R1/tf_bind_8-x-0.npy', 'tf_bind_8-VENTX_E101K_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ARX_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-VAX2_L139M_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX6_A79E_R1/tf_bind_8-x-0.npy', 'tf_bind_8-FOXC1_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-BCL6_H676Y_R1/tf_bind_8-x-0.npy', 'tf_bind_8-GFI1_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-SIX6_H141N_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX3_R270C_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HOXB7_M190L_R2/tf_bind_8-x-0.npy', 'tf_bind_8-ZNF655_S393L_R2/tf_bind_8-x-0.npy', 'tf_bind_8-VSX1_Q175H_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX6_P68S_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HOXC4_R158L_R2/tf_bind_8-x-0.npy', 'tf_bind_8-ARX_R332H_R2/tf_bind_8-x-0.npy', 'tf_bind_8-ARX_P353R_R2/tf_bind_8-x-0.npy', 'tf_bind_8-FOXC1_P79L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX3_N47K_R1/tf_bind_8-x-0.npy', 'tf_bind_8-GFI1_N382S_R1/tf_bind_8-x-0.npy', 'tf_bind_8-MSX2_P148H_R2/tf_bind_8-x-0.npy', 'tf_bind_8-KLF11_R402Q_R1/tf_bind_8-x-0.npy', 'tf_bind_8-WT1_R394W_R1/tf_bind_8-x-0.npy', 'tf_bind_8-SIX6_T165A_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HOXB7_R189C_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PROP1_R112Q_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PHOX2B_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PBX4_R215Q_R2/tf_bind_8-x-0.npy', 'tf_bind_8-SIX6_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-VENTX_R143C_R1/tf_bind_8-x-0.npy', 'tf_bind_8-POU6F2_L500M_R1/tf_bind_8-x-0.npy', 'tf_bind_8-SNAI2_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HOXD13_Q325R_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HOXD13_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX6_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-ARX_P353R_R1/tf_bind_8-x-0.npy', 'tf_bind_8-VENTX_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PBX4_R215Q_R1/tf_bind_8-x-0.npy', 'tf_bind_8-WT1_H373Y_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX4_R192H_R1/tf_bind_8-x-0.npy', 'tf_bind_8-VSX2_R200Q_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX6_V126D_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HOXC4_R158L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-NR1H4_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-GFI1B_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX7_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ISX_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ISX_R83Q_R1/tf_bind_8-x-0.npy', 'tf_bind_8-WT1_R366C_R1/tf_bind_8-x-0.npy', 'tf_bind_8-GFI1B_A204T_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX6_S119R_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ARX_P353L_R1/tf_bind_8-x-0.npy', 'tf_bind_8-VSX2_R200P_R1/tf_bind_8-x-0.npy', 'tf_bind_8-SNAI2_D119E_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ZNF200_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-PAX4_R192H_R2/tf_bind_8-x-0.npy', 'tf_bind_8-PAX3_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HOXB7_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HOXC4_N178S_R2/tf_bind_8-x-0.npy', 'tf_bind_8-PITX2_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-EGR2_R359W_R1/tf_bind_8-x-0.npy', 'tf_bind_8-KLF1_H299Y_R2/tf_bind_8-x-0.npy', 'tf_bind_8-SIX6_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-PBX4_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-ISX_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-GFI1B_A204T_R2/tf_bind_8-x-0.npy', 'tf_bind_8-MSX2_REF_R2/tf_bind_8-x-0.npy', 'tf_bind_8-EGR2_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-VSX2_REF_R1/tf_bind_8-x-0.npy', 'tf_bind_8-ZNF655_E327G_R1/tf_bind_8-x-0.npy', 'tf_bind_8-HESX1_N125S_R1/tf_bind_8-x-0.npy', 'tf_bind_8-NR2E3_R76Q_R1/tf_bind_8-x-0.npy']

# Chosen TRANSCRIPTION_FACTOR is 'SIX6_REF_R1'. Has to be changed here and in data_load.py if other TF is wanted. 
TRANSCRIPTION_FACTOR = "SIX6_REF_R1"
try:
    os.mkdir(DATA_FOLDER + "tf_bind_8/")
    os.mkdir(DATA_FOLDER + "tf_bind_8/" + TRANSCRIPTION_FACTOR + "/")
    print("TF_Bind_8 Directory created")
except OSError:
    print("TF_Bind_8 Directory already created")


if not os.path.exists(DATA_FOLDER + "tf_bind_8/SIX6_REF_R1/tf_bind_8-x.npy"):
    x_file, x_url = 'tf_bind_8/SIX6_REF_R1/tf_bind_8-x.npy', 'https://storage.googleapis.com/design-bench/tf_bind_8-SIX6_REF_R1/tf_bind_8-x-0.npy'
    y_file, y_url = 'tf_bind_8/SIX6_REF_R1/tf_bind_8-y.npy', 'https://storage.googleapis.com/design-bench/tf_bind_8-SIX6_REF_R1/tf_bind_8-y-0.npy'

    success = download(x_url, x_file)
    success = download(y_url, y_file)

    print("TF_Bind_8 Data downloaded")
else:
    print("TF_Bind_8 Data already downloaded")


### GFP ###
try:
    os.mkdir(DATA_FOLDER + "/gfp/")
    print("GFP Directory created")
except OSError:
    print("GFP Directory already created")

GFP_FILES = ['gfp/gfp-x-0.npy',
             'gfp/gfp-x-1.npy',
             'gfp/gfp-x-2.npy',
             'gfp/gfp-x-3.npy',
             'gfp/gfp-x-4.npy',
             'gfp/gfp-x-5.npy',
             'gfp/gfp-x-6.npy',
             'gfp/gfp-x-7.npy',
             'gfp/gfp-x-8.npy',
             'gfp/gfp-x-9.npy',
             'gfp/gfp-x-10.npy',
             'gfp/gfp-x-11.npy']


# Check if data already is downloaded
if not os.path.exists(DATA_FOLDER + "/gfp/gfp-x.npy"):
    x_files = [(file, f"https://storage.googleapis.com/design-bench/{file}") for file in GFP_FILES]
    y_files = [(file.replace('-x-', '-y-'), f"https://storage.googleapis.com/design-bench/{file.replace('-x-', '-y-')}") for file in GFP_FILES]
    GFP_LOC_X = [DATA_FOLDER + file for file in GFP_FILES]
    GFP_LOC_Y = [DATA_FOLDER + file.replace('-x-', '-y-') for file in GFP_FILES]

    for file, file_url in x_files:
        success = download(file_url, file)
        print(f"Download x: {success}")

    for file, file_url in y_files:
        success = download(file_url, file)
        print(f"Download y: {success}")
    print("GFP data downloaded")

    # Concatenate separate files into one
    X = np.concatenate([np.load(file) for file in GFP_LOC_X])
    Y = np.concatenate([np.load(file) for file in GFP_LOC_Y])

    np.save(DATA_FOLDER + "/gfp/gfp-x.npy", X)
    np.save(DATA_FOLDER + "/gfp/gfp-y.npy", Y)
    print("GFP saved as one file")

    # Remove residues
    for x, y in zip(GFP_LOC_X, GFP_LOC_Y):
        os.remove(x) 
        os.remove(y) 
    print("Removed residues")

else:
    print("GFP Data already downloaded")



