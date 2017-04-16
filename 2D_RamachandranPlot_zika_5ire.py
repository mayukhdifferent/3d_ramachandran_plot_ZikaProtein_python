import matplotlib.pyplot as plt
import chimera, numpy as np

def ramaPlot(res_L, saveFilePath = ""):
    plt.ion()
    fig = plt.figure(figsize = (8,8))

    ax = fig.add_subplot(111, aspect = 'equal')
    ax.set_title("2D Ramachandran Plot  of ZIKA VIRUS PDB: " + saveFilePath, fontsize = 16)
    ax.set_xlabel("PHI (degrees)", fontsize = 12)
    ax.set_ylabel("PSI (degrees)", fontsize = 12)
    ax.set_xlim([-180, 180])
    ax.set_ylim([-180, 180])
    ax.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    ax.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    ax.grid(True, linestyle = '-', color = '0.75')
    ax.hlines(0, -180, 180, colors='k', linestyles='dashed')
    ax.vlines(0, -180, 180, colors='k', linestyles='dashed')

    # Initialization:
    # List indicies: 0 for Helix, 1 for Strand, and 2 for Coil.
    colors_T = ('k', 'g', 'b') # Marker colors for Helix, Strand, and Coil.
    shapes_T = ('o', 's', 'd') # Marker shapes for Helix, Strand, and Coil.
    phi_L = [[], [], []]    # Phi lists for Helix, Strand, and Coil
    psi_L = [[], [], []]    # Psi lists for Helix, Strand, and Coil 

        
    # Generate the phi, psi coordinate lists:
    for r in res_L:
        if r.phi is None or r.psi is None: continue # Skip if residue has no phi, psi attributes.
        if r.isHelix:
            phi_L[0].append(r.phi)
            psi_L[0].append(r.psi)
        if r.isStrand:
            phi_L[1].append(r.phi)
            psi_L[1].append(r.psi)    
        if not (r.isHelix or r.isStrand):
            phi_L[2].append(r.phi)
            psi_L[2].append(r.psi)

    # Drop in the markers:
    for i in range(3):
        ax.scatter(phi_L[i], psi_L[i], s = 20, color = colors_T[i], marker = shapes_T[i])

    # Save file if the saveFilePath parameter is not the empty string.
    if saveFilePath != "":
        fig.savefig(saveFilePath + "_RamachandranPlot", dpi = fig.dpi)
        
    #raw_input("Hit Enter to dismiss the plot.")
    #plt.close()


#==============================================================================

#pdbIDchars = raw_input("Type in 4 character PDB ID or PDB file name: \n")
prot = chimera.openModels.open("5ire", type="PDB")[0]

ramaPlot(prot.residues, "5ire")
