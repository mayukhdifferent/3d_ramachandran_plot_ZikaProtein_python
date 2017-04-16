import chimera, numpy as np
from chimera import runCommand

from StructBio.Scenographics.solids import Solids
from StructBio.Scenographics.labels import LabelGroups

#=====================
maxHeight = 64.
# maxHeight does not affect the plot unless the largest bucket value
# is higher than maxHeight in which case the output is scaled so that
# the "tallest spire" is only maxHeight units high.
#=====================

# -----------------------------------------------------------------------------
# Return VRML string that draws a set of boxes.
def vr_plot(bucketCounts):                   
    baseColorCodes = [[2,3,3,3,3,3,3,3,3,3,3,2,2,2,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,2],
                      [2,3,3,3,3,3,3,3,3,3,3,3,2,2,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,2,2],
                      [2,3,3,3,3,3,3,3,3,3,3,3,2,2,2,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,2,2],
                      [2,2,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,2,2],
                      [2,2,3,3,3,3,3,3,3,3,3,3,3,3,2,2,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2],
                      [2,2,3,3,3,3,3,3,3,3,3,3,3,2,2,2,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,2],
                      [2,2,2,3,3,3,3,3,3,3,3,3,2,2,2,2,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,0,1,1,2],
                      [2,2,2,2,3,3,3,3,3,3,3,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,2],
                      [2,2,2,2,2,3,2,2,3,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,2,2,1,1,0,0,0,0,0,1,1,2],
                      [2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,2,2,2,2,1,1,1,0,0,0,0,0,1,1,1],
                      [2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,2,2,2,2,1,1,1,0,0,0,0,0,1,1,1],
                      [1,2,2,2,2,2,2,2,2,2,2,2,2,1,1,0,0,0,0,1,1,1,2,2,2,1,1,1,1,0,0,0,0,1,1,1],
                      [1,1,2,2,2,2,2,2,2,2,2,2,2,1,1,0,0,0,0,1,1,2,2,2,2,2,2,1,1,0,0,0,0,1,1,1],
                      [1,1,2,2,2,2,2,2,2,2,2,2,1,1,1,0,0,0,0,1,1,2,2,3,2,2,1,1,1,0,0,0,0,1,1,1],
                      [1,2,2,2,2,2,2,2,2,2,2,2,1,1,1,0,0,0,0,1,1,1,2,3,3,2,1,1,1,0,0,0,0,1,1,1],
                      [1,2,2,2,2,2,2,3,2,2,2,2,2,1,1,1,0,0,0,1,1,1,2,2,2,2,2,1,1,0,0,0,0,1,1,1],
                      [1,2,2,2,2,2,3,3,3,3,2,2,2,1,1,1,1,0,0,0,1,1,1,2,2,2,1,1,1,0,0,0,0,1,1,1],
                      [1,2,2,2,2,2,3,3,3,3,3,2,2,2,1,1,1,0,0,0,1,1,1,1,2,2,2,1,1,0,0,0,0,1,1,1],
                      [1,2,2,2,2,3,3,3,3,3,3,3,2,2,2,1,1,1,0,0,1,1,1,2,1,2,2,1,1,0,0,0,0,1,1,1],
                      [1,2,2,2,2,2,3,3,3,3,3,3,3,2,2,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1],
                      [1,1,2,2,2,2,3,3,3,3,3,3,3,3,2,2,1,1,1,0,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1],
                      [1,1,2,2,2,2,2,3,3,3,3,3,3,3,2,2,2,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
                      [1,1,1,2,2,2,2,2,3,3,3,3,3,3,3,2,2,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
                      [1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,2,2,2,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
                      [2,2,1,2,2,2,2,2,2,2,2,3,3,3,3,2,2,2,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
                      [1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
                      [1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
                      [0,1,1,1,1,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,1,1,1,1,2,2,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [1,1,1,1,1,2,2,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
                      [1,1,1,1,1,1,2,2,2,2,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
                      [1,2,2,2,1,2,2,2,2,2,1,1,1,1,1,0,0,0,0,0,0,1,1,2,1,1,1,0,0,0,0,0,0,0,0,0],
                      [1,2,2,2,2,2,2,2,2,2,2,1,1,1,1,0,0,0,0,0,0,1,1,2,1,1,1,0,0,0,0,0,0,1,1,1],
                      [1,2,2,2,2,2,2,2,2,2,2,2,2,1,1,0,0,0,0,0,0,1,1,2,2,1,1,0,0,0,0,0,0,1,1,1],
                      [2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,2]]

    baseBoxes = Solids("Base Boxes")
    histoBoxes = Solids("Histogram Boxes")
    axes = Solids("Axes")
    ticks = Solids("Ticks")
    plotLabels = LabelGroups("Tick Labels")
    
    baseColors = ((1.0, 1.0, 0.5), (1.0, 0.8, 0.3), (0.8, 0.6, 0.1), (1.0, 0.0, 0.0))
    baseHeights = (0.6, 0.7, 0.8, 1.00)
    
    for ii in range(36):
        for jj in range(36):
            # Put in cubes making up the base:
            bCC = baseColorCodes[ii][jj]
            baseBoxes.addBox(np.array((jj, 35-ii, 0.0)), (1, 1, baseHeights[bCC]), baseColors[bCC])

            # Put in boxes for bucket values:
            boxHeight = bucketCounts[ii, jj]
            histoBoxes.addBox(np.array((jj, 35-ii, boxHeight/2.0)), (1, 1, boxHeight), baseColors[bCC])

    axes.addBox(np.array((17.5, 17.0, 0.5)), (0.1, 37, .1), (0,0,0))
    axes.addBox(np.array((17.0, 17.5, 0.5)), (37, 0.1, .1), (0,0,0))

    for ii in range(9):
        xpos = -0.5 + ii*4.5
        ticks.addBox(np.array((xpos, 17.0, 0.4)), (0.06, 37, .1), (.2,.2,.2))
        ypos = -0.5 + ii*4.5
        ticks.addBox(np.array((17.0, ypos, 0.4)), (37, 0.06, .1), (.2,.2,.2))
    
    baseBoxes.display()
    histoBoxes.display()
    axes.display()
    ticks.display()

# -----------------------------------------------------------------------------
# Put in axes as a separate model.

    plotLabels.addLabelGroup("AxesLabels")
    plotLabels.addLabelGroup("TickLabels")

    xPosOffset = -2.0
    plotLabels.addLabel("AxesLabels", "Phi", (36.0, 18.0, 0.0))
    plotLabels.addLabel("AxesLabels", "Psi", (18.0, 36.0, 1.0))
    plotLabels.addLabel("AxesLabels", "  0", (xPosOffset + 18.0, -4.0, 0.6))    
    plotLabels.addLabel("AxesLabels", "  0", (-4.0, 18.0, 0.6))

    pAxVals = (' 45', ' 90', ' 135', ' 180')
    nAxVals = ('-45', '-90', '-135', '-180')
    for ii in range(4):
        plotLabels.addLabel("TickLabels", pAxVals[ii], (xPosOffset + 22.5 + 4.5*ii, -4.0, 0.6))
        plotLabels.addLabel("TickLabels", nAxVals[3-ii], (xPosOffset - 0.5 + 4.5*ii, -4.0, 0.6))
        plotLabels.addLabel("TickLabels", pAxVals[ii], (-4.0, 22.5 + 4.5*ii,  0.6))
    for ii in range(3):
        plotLabels.addLabel("TickLabels", nAxVals[2-ii], (-4.0, 4. + 4.5*ii,  0.6))
        
    plotLabels.setLabels("AxesLabels", "black")
    plotLabels.setLabels("TickLabels", "black")
    plotLabels.showLabels()
    
#==============================================================================
# Mainline:
#==============================================================================
#pdbIDchars = raw_input("5ire")
model = chimera.openModels.open("5iy3", type="PDB")
prot = model[0]

# To keep it simple and look at all residues in the protein
# (no attempt to work with a single chain).

buckets = np.zeros((36,36))
maxCount = 0

for res in prot.residues:
    if not (res.phi and res.psi): continue  # Skip if we are missing either angle.
    rowIndex = int(np.floor(36 - (res.psi + 180.)/10.))
    colIndex = int(np.floor((res.phi + 180.)/10.))
    buckets[rowIndex, colIndex] += 1
    if buckets[rowIndex, colIndex] > maxCount:
        maxCount = buckets[rowIndex, colIndex]

if maxCount > maxHeight:
    for ii in range(36):
        for jj in range(36):
            buckets[ii, jj] = buckets[ii, jj]*maxHeight/maxCount
            
chimera.openModels.close(prot)

vr_plot(buckets)
runCommand("set bg_color white")

# Center the plot in the display:
chimera.viewer.viewAll()
