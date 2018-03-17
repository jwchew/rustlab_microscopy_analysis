"""
Justin Chew
Rust Lab

This script takes the output from Guillaume's cell tracker and plots all of the
cell lineages from the very first cells to the very last cells, storing data as
linked lists in the forward (mother -> daughters) direction and reverse (daugh-
ter -> mother) direction.

Guillaume's "trP.npy" file contains data with the following format:
Column: 0       1       2       3       4       5       6       7           8               9
Data:   xPos    yPos    time    cellID  PoleID  Length  Width   Orientation Cell intensity  LeftMother information

Column: 10                      11          12          13          14
Data:   RightMother information FamilyID    NewPoleAge  OldPoleAge  Elongation rate

Data is stored in the following way for mother/daughter cells:
For each individual cell, the first frame in which it has split from its
sister cell lists the mother cell ID in columns 9 and 10 (left/right
mother information).  The last frame before it splits into two daughter
cells lists the cell IDs for the two daughter cells in the left/right
mother information.  All of the intermediate frames hold a value of zero
for the mother information in these two columns.

Example for a cell (cellID = 2) that descended from a mother cell
(cellID = 1), splitting into two daughter cells (cellID = 3 and 4) where
the cell data for cellID = 2 has been pulled out and lists columns 9 and
10 in time-sorted order:

Column
9   10
------
1   1
0   0
0   0
0   0
3   4
"""

import numpy
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import os
from scipy.optimize import curve_fit
import shutil
import sys
import errno
import pickle

# save a copy of this script in the data directory
#base = 071715
#shutil.copy(sys.argv[0], directory[1:] + "/" + base + "_" + sys.argv[0])

# adjust figure major ticks
majorLocator = MultipleLocator(24)
majorFormatter = FormatStrFormatter("%d")
minorLocator = MultipleLocator(12)

expt_name = "2.23.18 368 92 23 uM theo 16 8 LD cycles"

pickle_lineage = True   # flag for whether the lineages should be saved to file
wait_time = 1.0 # time (h) between each time point
start_time = 21.0    # most of the time this should be 0 if all beginning frames of expt used

def makedir(dir_name):
    """
    makedir creates the specified directory path if it doesn't exist
    """
    try:
        os.makedirs(dir_name)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


# generate list of folders
original_cwd = os.getcwd()
folders = os.listdir(expt_name + "/raw data/")
folders = ["73-2-23-6", "73-2-92-6", "73-2-368-3", "YFP-6"]

for fPath in folders:

    try:
        trP = numpy.load(original_cwd + "/" + expt_name + "/raw data/" + fPath + "/trP.npy")
    except:
        print "Error: could not load trP.npy for", fPath
    else:        
        output_dir = original_cwd + "/output/" + expt_name + "/lineages/" + fPath
        makedir(output_dir)
                
        os.chdir(output_dir)
        
        # extract data for individual cellIDs
        cell_id = trP[:,3]      # grab cell_id column from full dataset
        cell_set = set(cell_id) # make a list of all unique cell elements in pole_id

        cell_data = {}    # store the data (value) for each cell ID (key)

        for cell in cell_set:
            cell_bool = cell_id == cell
            cell_data[cell] = trP[cell_bool == True,:]

        # build lineage tree in the forward direction
        forward_tree = {} # dict to hold forward lineage
        reverse_tree = {} # dict to hold reverse lineage
        endnodes = []     # cells without children

        def traverse(current_cellID):

            # check to see if left and right nodes exist
            # pull out the cellIDs that contain the current cellID as the first entry in their mother column
            leftnode_check, rightnode_check = False, False
            leftnode = cell_data[current_cellID][:,9][-1]
            rightnode = cell_data[current_cellID][:,10][-1]

            if leftnode != 0:
                leftnode_check = True
            if rightnode != 0:
                rightnode_check = True

            # sanity check in case somehow a cell has one daughter but not another (cell should either have two daughters or none)
            if leftnode_check != rightnode_check:
                print "Weird node behavior at ", current_cellID
                print leftnode_check, rightnode_check

            # if node has no children, add it to endnodes
            if not(leftnode_check and rightnode_check):
                endnodes.append(current_cellID)
                #print current_cellID, reverse_tree[current_cellID]

            if leftnode_check and rightnode_check:
                forward_tree[current_cellID] = (leftnode, rightnode)
                reverse_tree[leftnode] = current_cellID
                reverse_tree[rightnode] = current_cellID

                traverse(leftnode)
                traverse(rightnode)

        # construct and plot actual lineages
        mother_cells = []   # initial population of mother cells at beginning of experiment

        for cell in cell_set:
            # only add mother cell if it has no parent and it is present at t=0
            if (cell_data[cell][:,9][0] == 0) and (cell_data[cell][:,2][0] == start_time):
                mother_cells.append(cell)

        for cell in mother_cells:
            traverse(cell)

        # find set of end nodes with no children, i.e. keys that don't appear in values
        endnodes = sorted(set(endnodes))

        lineages = []   # holds a list of lineages where each lineage is itself a list of cell_IDs that form that single lineage

        for node in endnodes:
            
            templineage = [node]
            currentnode = node
            
            while currentnode in reverse_tree:
                currentnode = reverse_tree[currentnode]
                templineage.append(currentnode)

            lineages.append(templineage)

        # divides lineages and cells up amongst the original mother cells
        mother_lineages = {}        # stores all cell data for a mother lineage
        mother_descendants_raw = {} # stores all descendant cell IDs of mother

        for mother in mother_cells:
            mother_lineages[mother] = []
            mother_descendants_raw[mother] = []

        for item in lineages:
            cell_data_array = [cell_data[x] for x in item]
            cell_data_array.reverse()   # reverse data array back to normal order since data was read in from reverse_tree
            lineage_data = numpy.concatenate(cell_data_array, axis=0)

            mother_lineages[lineage_data[:,3][0]].append(lineage_data)    # for each lineage, put the data into the appropriate mother cell lineage
            mother_descendants_raw[lineage_data[:,3][0]].extend(item)

        mother_descendants = {} # holds set of all cells descended from a mother
        for mother in mother_cells:
            mother_descendants[mother] = set(mother_descendants_raw[mother])

        # finds average cell intensity over individual mother lineages
        lineage_averages = {}
        timepoints = list(set(trP[:,2]))
        max_time = max(timepoints)

        # finds any gaps in the time series when images are not acquired due to darkness
        gaps = []
        for i, time in enumerate(timepoints[:-1]):
            if (time+1) != timepoints[i+1]:
                gaps.append((timepoints[i], timepoints[i+1]))

        for mother in mother_cells:

            # next pull all the cell IDs in this set and average over each time point in the whole time course
            average_intensity = []

            # for each time point, check each descendant and add the cell intensity at that time point if it is available
            for time in timepoints:

                cell_intensities = []

                for descendant in mother_descendants[mother]:
                    for datarow in cell_data[descendant]:
                        if datarow[2] == time:
                            cell_intensities.append(datarow[8])
                
                average_intensity.append(numpy.mean(cell_intensities))

            lineage_averages[mother] = average_intensity

        # Note: modified on 11/15/16 to test out manual cell segmentation
        #start_fit = 20.0 / wait_time
        #end_fit = 100.0 / wait_time
        start_fit = 0.0
        end_fit = len(timepoints)

        for mother in mother_cells:
            
            for item in mother_lineages[mother]:

                time_data = item[:,2] * wait_time
                intensity_data = item[:,8]

                plt.plot(time_data, intensity_data, color = "0.75", zorder=1)

            plt.plot(numpy.array(timepoints)*wait_time, numpy.array(lineage_averages[mother]), color = "black", linewidth = 1, zorder=1)

            # plot white spans over gaps in data to hide the transitions
            for gap in gaps:
                plt.axvspan(gap[0], gap[1], facecolor="white", edgecolor="None", zorder=2)

            plt.gca().get_xaxis().set_major_locator(majorLocator)
            plt.gca().get_xaxis().set_major_formatter(majorFormatter)

            plt.title("All lineages test: cell lineage " + str(mother))
            plt.xlabel("Time (h)")
            plt.ylabel("Intensity")

            plt.savefig("cell lineage " + str(mother) + " " + fPath + ".png", format="png")
            plt.clf()
            
            # output lineage data to pickle file if specified
            if pickle_lineage:
                outputfile = open("cell lineage " + str(mother) + " " + fPath + ".pickle", "wb")
                pickle.dump((mother_lineages[mother], (timepoints, lineage_averages[mother])), outputfile)
                outputfile.close()
            
            # plot all lineages individually against the average lineage for each mother cell in its own folder
            cwd = os.getcwd()
            try:
                os.mkdir(cwd + "/" + "cell lineage " + str(mother) + " " + fPath)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise
                    
            os.chdir(cwd + "/" + "cell lineage " + str(mother) + " " + fPath)
            count = 0
            for item in mother_lineages[mother]:
            
                # generate list of cellIDs in the lineage, from beginning to end
                cellID_lineage = []
                for cell in item[:,3]:
                    if cell not in cellID_lineage:
                        cellID_lineage.append(cell)
                
                # convert cellID lineage to string for plot title
                lineage_as_string = ", ".join([str(x) for x in cellID_lineage])
                
                # compile list of mother->daughter division events and cellIDs, one example entry might be:
                # "1.0 -> (2.0, 3.0)" to show a mother cell (cellID = 1) dividing into two cells (IDs 2 and 3)
                division_IDs = []
                for i in range(len(cellID_lineage)-1):  # leave out last cell in lineage since it doesn't divide
                    cell = cellID_lineage[i]
                    div_event = str(cell) + " ->\n(" + str(forward_tree[cell][0]) + ", " + str(forward_tree[cell][1]) + ")"
                    division_IDs.append(div_event)
            
                # extract all cell division events, i.e. last timepoint for each unique cell ID
                division_times = []
                for i in range(len(cellID_lineage)-1):
                    cell = cellID_lineage[i]
                    division_times.append(cell_data[cell][:,2][-1])
                
                time_data = item[:,2] * wait_time
                intensity_data = item[:,8]

                # plot end of lineage track (blue line)
                plt.axvline(cell_data[cellID_lineage[-1]][:,2][-1], color="#ccccff")

                # plot division times (red lines) and the mother/daughter splits
                for i in range(len(division_times)):
                    plt.axvline(division_times[i], color="#ffcccc")
                    
                    # stagger div labels so consecutive ones are readable
                    if i % 2 == 0:
                        div_label_y = 1000
                    elif i % 2 == 1:
                        div_label_y = 750
                    plt.text(division_times[i], div_label_y, division_IDs[i], fontsize=7, horizontalalignment='center', verticalalignment='center')

                plt.plot(time_data, intensity_data, color = "0.75", zorder=1)
                plt.ylim(500,2800)
                plt.plot(numpy.array(timepoints)*wait_time, numpy.array(lineage_averages[mother]), color = "black", linewidth = 1, zorder=1)
                # plot white spans over gaps in data to hide the transitions
                for gap in gaps:
                    plt.axvspan(gap[0], gap[1], facecolor="white", edgecolor="None", zorder=2)
                plt.title(fPath + " " + "mother " + str(mother) + "\n" + "lineage: " + lineage_as_string)
                plt.savefig("lineage " + str(count) + ".png", format="png")
                plt.clf()
                count += 1
            os.chdir(cwd)

        os.chdir(original_cwd + "/" + expt_name + "/raw data/" + fPath)
            
        print "Finished plotting for", fPath
