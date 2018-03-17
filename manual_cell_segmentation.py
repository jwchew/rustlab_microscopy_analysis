"""
Justin Chew
November 15, 2016

This script allows for manual cell segmentation for use with Guillaume's
celltracker.py script and the modified ProcessImages.py script.  As input, it
takes in the same folders and images as ProcessImages.py would (a directory
containing brightfield, chlorophyll, and YFP images), and from the brightfield
images, it produces a segmented cell mask from user-directed cell segmentation,
specifically a numpy array containing the information stored in the "bwImg"
variable in the ProcessImages.py script.

Run this script before running the modified ProcessImages_manual_segmentation.py
script.

RECOMMENDED: Open System Monitor to check on RAM usage since this program eats a
lot of memory!  Typically can process 10-50 images in one sitting.

Controls
-----------
D: next image
A: previous image
S: save labeled image
W: undo (up to 10 times)
Q: quit and save all images
F: copy cell segmentation data from previous frame to current frame
G: copy cell segmentation data from subsequent frame to current frame
R: reset frame to original state without user input
T: toggle cell selection transparency
Z: copy previous frame's cell selections to current frame to initiate guess, hit Z again to cancel
UHJK: move a cell selection guess up, left, down, right, respectively
X: initiate guess
P: toggle timekeeping pause (press once to pause, press again to unpause)
L: load selection from specified file
Ctrl: hold while clicking on a cell to delete its selection
Shift: hold to draw without optimizing user cell selection
Esc: quit without saving
Left mouse button: click and drag to draw a cell outline

Inputs
-----------
A folder containing brightfield images

Outputs
-----------
A tif file corresponding to the original "processed.tif" output image as well as
a numpy array containing bwL for each image (the uniquely labeled cells)

Misc
-----------
Keyvalues for opencv2 waitKey:
Q: 113
W: 119
A: 97
S: 115
D: 110
E: 101
R: 114
F: 102
G: 103
T: 116

Z: 122
X: 120
U: 117
H: 104
J: 106
K: 107

L: 108
"""

import cv2
import numpy as np
import random
import os
import re
from datetime import datetime
from datetime import timedelta
import math

expt_dir = ""       # name of experimental directory with microscope images
bounds = (1, 10)    # bounds are inclusive and start from 1
load_file = ""      # should be a "cellcoords.npy" file

# TEMPORARY VARIABLES FOR DRAWING
refPt = []          # holds starting and ending coordinates of mouse button when drawing
drawing = False     # indicates whether mouse button is down
transparency = True # indicates whether cell selection should be transparent
guessing = False    # indicates whether the guessing procedure is ongoing

# CONSTANTS FOR PROGRAM FUNCTION
cell_radius = 6     # typical half-width of cell in pixels
undo_depth = 11     # the number of times + 1 a cell drawing can be undone
search_radius = 2   # the radius of the circle searched for optimizing cell selection
tr_alpha = 0.5      # specifies degree of transparency overlay (0 = transparent, 1 = opaque)

# TEMPORARY VARIABLES STORING USER-INPUT DRAWING INFO
undo_buffer = []    # holds the images and bwL arrays in buffer memory for undo operations, each entry is a tuple containing (image you're seeing, bwL array behind it)
used_colors = []    # holds list of colors already used to draw cells
cell_number = 1     # holds current cell label number for generating output image
cell_coords = []    # holds the refPt coordinate pairs for each cell

# VARIABLES FOR HOLDING INFO FOR EACH IMAGE
current_image_index = 0 # index of current loaded image, used for switching between images
image_buffer = []   # holds both the color and bw arrays of current image state for each image
data_buffer = []    # holds tuple of (undo_buffer, used_colors, cell_number, cell_coords) for each image

# MISC VARIABLES
start_times = []    # stores all start points for timekeeping
end_times = []      # stores all end points for timekeeping
paused = False      # stores whether the program timekeeping is paused or not
create_log = True  # specifies whether an output log file should be generated

# mouse callback function
def mouse_input(event, x, y, flags, param):

    global refPt, drawing, img, bwL, predraw_img, cell_number

    if event == cv2.EVENT_LBUTTONDOWN:
        # don't initiate drawing if click is for deleting a cell selection
        if flags == cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_LBUTTON:
            return
        # otherwise, begin the process of drawing a new cell selection
        predraw_img = img.copy()
        refPt = [(x, y)]
        drawing = True

    if event == cv2.EVENT_MOUSEMOVE and drawing:

        img = predraw_img.copy()

        # draw cell between the two indicated points in refPt
        cv2.circle(img, refPt[0], cell_radius, (0, 255, 0), -1)
        cv2.line(img, refPt[0], (x, y), (0, 255, 0), cell_radius*2)
        cv2.circle(img, (x, y), cell_radius, (0, 255, 0), -1)
        
    if event == cv2.EVENT_LBUTTONUP:

        # detect if Ctrl is held down to decide whether to draw or delete cell
        if flags == cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_LBUTTON:
            delete_cell(x, y)
            return

        refPt.append((x, y))
        drawing = False

        # use manual input instead of optimization if shift key pressed while drawing
        if flags == cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_FLAG_LBUTTON:
            draw_cell(refPt)
        else:
            draw_cell(circle_search(refPt))


def draw_cell(coords, redraw_flag = True):
    """
    Draws a cell into the selections and bwL images, taking as input a pair of (x, y)
    coordinates.  The flag redraw_img specifies whether to redraw the image after drawing
    the cell.
    """

    global selections, bwL, undo_buffer, cell_number, cell_coords

    # generate random unique color for the cell
    cell_color = generate_color()

    # draw cell between the two indicated points in refPt
    for point in coords:
        cv2.circle(selections, point, cell_radius, cell_color, -1)
        cv2.circle(bwL, point, cell_radius, cell_number, -1)
    cv2.line(selections, coords[0], coords[1], cell_color, cell_radius*2)
    cv2.line(bwL, coords[0], coords[1], cell_number, cell_radius*2)

    undo_buffer.pop(0)
    undo_buffer.append((selections.copy(), bwL.copy()))
    cell_number += 1
    cell_coords.append(coords)

    if redraw_flag:
        redraw_img(selections, original_img, cell_coords)


def generate_color():
    """
    Generates new color that is not in the used_colors list, ensuring that each new
    color is above a certain brightness/vibrant level (prevents dark colors from being
    assigned to cells, as these are harder to see on transparencies).

    Returns a tuple with three values for each channel in BGR.
    """
    new_color = False
    max_value = 70
    while not new_color:
        cell_color = (random.randint(0,max_value), random.randint(0,max_value), random.randint(0,max_value))
        if cell_color not in used_colors and sum(cell_color) > max_value*2.16 and sum(cell_color) < max_value*2.6:
            used_colors.append(cell_color)
            new_color = True
    return cell_color


def mat2gray(img, scale=1):
    """
    imgOut = mat2gray(img, scale) return a rescaled matrix from 1 to scale
    """

    imgM = img - img.min()

    imgOut = imgM*np.double(scale)/imgM.max()

    return imgOut


def _sort_nicely(l):
    """ Sort the given iterable in the way that humans expemt."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c)
                                for c in re.split('([0-9]+)',  key)]
    return sorted(l, key=alphanum_key)


def load_image(index):
    """
    Loads an image into active memory.  Takes as input the index
    of the image that should be loaded.

    Below are some variable definitions, hope this is helpful:
    -------------------------
    img: the main image that makes up the workspace, containing all user input and cell selections
    bwL: mask containing all cell selections with each individual cell assigned a unique cell number
    selections: holds all colored cell selections on a copy of original_img for transparency function
    s_chan_img: the original single channel version of the image, remains unchanged
    original_img: the original, autocontrasted, color version of img before any user input
    predraw_img: the previous version of img before a user-input modification, useful for drawing

    Note that any user input is divided into a separate selections channel (selections, bwL) and the
    original image (original_img), and the output to the screen (img) is a combination of these two.
    Transparency alters how these two are combined and displayed.
    """

    global img, bwL, selections, s_chan_img, original_img, predraw_img, undo_buffer, used_colors, cell_number, cell_coords
    
    if not image_buffer[index]:
        reset_image(index)
    else:
        selections = image_buffer[index][0]
        bwL = image_buffer[index][1]
        s_chan_img = image_buffer[index][2]
        original_img = image_buffer[index][3]
        undo_buffer = data_buffer[index][0]
        used_colors = data_buffer[index][1]
        cell_number = data_buffer[index][2]
        cell_coords = data_buffer[index][3]
        
        redraw_img(selections, original_img, cell_coords)


def save_image(i):
    """
    Saves the current image and associated info into memory
    """
    
    global image_buffer, data_buffer

    image_buffer[i] = (selections.copy(), bwL.copy(), s_chan_img.copy(), original_img.copy())
    data_buffer[i] = (list(undo_buffer), list(used_colors), cell_number, list(cell_coords))


def save_image_to_file(i):
    """
    Saves current image to disk in the output folder, taking the current image index
    as input i.

    Note: the output "segmented.npy" array is cast as float64 since that is what Guillaume's
    code had bwImg as.  Not sure exactly why it is this way considering it is just 0s and 1s.
    """

    file_prefix = "_".join(input_images[i].split("_")[:-1])

    redraw_img(image_buffer[i][0], image_buffer[i][3], data_buffer[i][3])

    cv2.imwrite(expt_dir + "/" + file_prefix + "_labeled.tif", img)
    cv2.imwrite(expt_dir + "/" + file_prefix + "_presegment.tif", image_buffer[i][1])
    segmented_bin, segmented_bin_rescaled = segment_by_erosion(image_buffer[i][1])
    np.save(expt_dir + "/" + file_prefix + "_segmented.npy", np.float64(segmented_bin))
    cv2.imwrite(expt_dir + "/" + file_prefix + "_segmented-rescaled.tif", segmented_bin_rescaled)
    np.save(expt_dir + "/" + file_prefix + "_cellcoords.npy", data_buffer[i][3])
    

def segment_by_erosion(bwL):
    """
    Takes each cell by its individual label value and erodes it to separate cells that
    are touching each other.  Sacrifices some data since the mask area for each cell is
    smaller but guarantees segmented cells.

    Input is an image in which each cell is labeled with individual values, essentially
    the result of the celltracker.py bwlabel function.

    Output is two verisons of the erosion mask (i.e. a binarized version of the cells
    after segmentation), one binarized to 0/1, the other to 0/255.
    """

    eroded_mask = np.zeros(bwL.shape)
    
    # use a 3x3 square kernel for the erosion
    kernel = np.ones((3,3), np.uint8)

    # from bwL, isolate individual cells and erode them with the kernel, add result to eroded_mask
    for i in range(1,bwL.max()+1):
        isolated = np.zeros(bwL.shape)
        isolated[bwL == i] = 1
        cell_erosion = cv2.erode(isolated, kernel, iterations=1)
        eroded_mask = eroded_mask + cell_erosion

    # rescale eroded_mask to have max value of 255
    mask_rescale = np.zeros(eroded_mask.shape)
    mask_rescale[eroded_mask == 1] = 255

    return eroded_mask, mask_rescale


def copy_image(i):
    """
    Copies all user-specified drawing data from the specified frame into
    the current frame.
    """

    global selections, bwL, used_colors, undo_buffer, cell_number, cell_coords

    # copy cell selections over
    selections = image_buffer[i][0].copy()
    bwL = image_buffer[i][1].copy()
    # redraw all previous cell selections
    copy_cell_coords = list(data_buffer[i][3])
    for coord in copy_cell_coords:
        draw_cell(coord)


def reset_image(index):
    """
    Resets image to original loaded state before any cell selections were made.
    """

    global img, bwL, selections, s_chan_img, original_img, used_colors, undo_buffer, cell_number, cell_coords

    # hold a single channel version of the image to speed up optimization calculations
    s_chan_img = cv2.imread(input_dir + "/" + input_images[index], -1)
    # autocontrast image so that cells are visible
    bwimg = np.uint8(mat2gray(s_chan_img, 255))        
    # convert to color to allow for easy visualization of cell labeling
    original_img = cv2.cvtColor(bwimg, cv2.COLOR_GRAY2RGB)
    # write the image info at the top left of the image
    cv2.putText(original_img, input_images[index], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    text = str(index+1) + "/" + str(len(input_images))
    cv2.putText(original_img, text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    # create color image that holds cell selections on a black background
    selections = cv2.cvtColor(np.zeros(s_chan_img.shape, np.uint8), cv2.COLOR_GRAY2RGB)
    # create mirror array to hold labeled cells, uses uint16 to be able to label all the cells adequately
    bwL = np.zeros(bwimg.shape, np.uint16)

    # initialize the temporary variables for each image
    for i in range(undo_depth):
        undo_buffer.append((selections.copy(), bwL.copy()))
    used_colors = []
    cell_number = 1
    cell_coords = []

    redraw_img(selections, original_img, cell_coords)


def bounding_box(refPt, padding=12):
    """
    Given a pair of points (refPt), returns the corners of the bounding box
    plus some extra room, to account for the fact that the cells have a certain
    thickness beyond the points defined by refPt (the padding, in px).

    Practically, this function speeds up the circle_search and get_intensity
    functions by processing only a small fraction of the entire image instead of
    arrays that are as large as the images themselves.
    """

    x_coords = (refPt[0][0], refPt[1][0])
    y_coords = (refPt[0][1], refPt[1][1])
    max_x = max(x_coords) + padding
    min_x = min(x_coords) - padding
    max_y = max(y_coords) + padding
    min_y = min(y_coords) - padding
    
    return (min_x, max_x, min_y, max_y)


def get_intensity(img, refPt):
    """
    Calculates the average intensity of the brightfield image underneath the
    mask defined by the given cell selection (demarcated by the coordinates in
    refPt).
    """

    min_x, max_x, min_y, max_y = bounding_box(refPt)

    # create mask to calculate average intensity
    mask = np.zeros((max_y-min_y, max_x-min_x))
    
    # translate the input refPt points to the coordinate system of the bounding box
    new_refPt = []
    for point in refPt:
        new_refPt.append((point[0]-min_x, point[1]-min_y))

    for point in new_refPt:
        cv2.circle(mask, point, cell_radius, 1, -1)
    cv2.line(mask, refPt[0], refPt[1], 1, cell_radius*2) 

    try:    # this code fails sometimes for reasons I still need to dig into
        intensity = np.average(img[min_y:max_y, min_x:max_x], None, mask)
    except Exception as e:
        print "Error in getting cell intensity: cell selection not valid"
        print e
        return 0
    else:
        return np.average(img[min_y:max_y, min_x:max_x], None, mask)


def circle_search(refPt, int_threshold = 1.0028):
    """
    Optimizes the user-specified cell selection by attempting to find a similar
    cell selection that minimizes the average brightness under the cell selection,
    which relies on the cell interior being much darker than its borders and the
    exterior environment.

    Takes as input a pair of coordinates demarcating the ends of the cell, and for
    each coordinate, searches in a circular area around each originally specified
    end to find an optimal minimum. In essence, it "fits" the cell selection to the
    image given a user-provided initial guess.

    The parameter int_threshold describes the tolerance for finding the longest
    selection that fits the cell. Without this parameter (e.g. if int_threshold = 1),
    then the function will return the intensity that is the absolute minimum of all
    of the tested fits, which often results in a shorter cell than the selection.
    Over subsequent frames, this leads to selections that are much shorter than the
    actual cell.
    """

    min_x, max_x, min_y, max_y = bounding_box(refPt)

    # translate the input refPt points to the coordinate system of the bounding box
    new_refPt = []
    for point in refPt:
        new_refPt.append((point[0]-min_x, point[1]-min_y))

    search_area = []    # two indices, one for each circle
    for point in new_refPt:
        temp_mask = np.zeros((max_y-min_y, max_x-min_x))
        cv2.circle(temp_mask, point, search_radius, 255, -1)
        nonzero = np.where(temp_mask == 255)
        search_area.append([(x+min_x, y+min_y) for x, y in zip(nonzero[1], nonzero[0])])

    # create a table of cell lengths vs. intensities for later sorting
    fit_info = None
    flag = False    # to prompt creation of first numpy array for vstacking
    for point1 in search_area[0]:
        for point2 in search_area[1]:
            length = math.sqrt((point2[1]-point1[1])*(point2[1]-point1[1]) + (point2[0]-point1[0])*(point2[0]-point1[0]))   # x*x is faster in python than x**2
            intensity = get_intensity(s_chan_img, (point1, point2))
            table_entry = np.array([point1, point2, length, intensity])
            if not flag:
                fit_info = table_entry
                flag = True
            else:
                fit_info = np.vstack((fit_info, table_entry))
    
    # sort table by length, from http://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column, and find longest cell that still is near min intensity
    fit_info = fit_info[fit_info[:,2].argsort()]
    min_intensity = fit_info[:,3].min()
    best_fit = refPt
    for i in range(fit_info.shape[0]):
        if fit_info[i][3] <= min_intensity * int_threshold:
            best_fit = (fit_info[i][0], fit_info[i][1])

    return best_fit


def delete_cell(x, y):
    """
    Deletes a given cell selection based on the user Ctrl-clicking the cell they
    wish to delete.

    Takes as input the x and y coordinates of a mouse click.
    """

    global bwL, selections

    # identify pixels of cell selection from bwL, do nothing if there is no cell to select
    del_value = bwL[y][x]
    if not del_value:
        return
    del_indices = np.where(bwL == del_value)
    del_pixels = ([(x, y) for x, y in zip(del_indices[1], del_indices[0])])

    # reset pixels of deleted cell in both img and bwL
    for coord in del_pixels:
        bwL[coord[1]][coord[0]] = 0
        selections[coord[1]][coord[0]] = (0, 0, 0)

    # delete the corresponding cell in cell_coords
    for pair in cell_coords:
        if pair[0] in del_pixels or pair[1] in del_pixels:
            cell_coords.remove(pair)

    redraw_img(selections, original_img, cell_coords)

    undo_buffer.pop(0)
    undo_buffer.append((selections.copy(), bwL.copy()))


def redraw_img(selections, original_img, cell_coords):
    """
    This function combines the cell selection mask and the original image into a single
    image for display, taking into account whether transparency is toggled.
    """

    global img

    # return original image if there are no cell selections
    if not cell_coords:
        img = original_img.copy()
        return

    img = cv2.addWeighted(selections, tr_alpha, original_img, (1-tr_alpha), 0)*2


    """ old code which was much much slower than cv2 builtin addWeighted
    # creates bounding box for each coordinate pair to speed up redraw
    bounding_indices = []
    for pair in cell_coords:
        x_coords = [pt[0] for pt in pair]
        y_coords = [pt[1] for pt in pair]
        max_x = max(x_coords) + cell_radius
        min_x = min(x_coords) - cell_radius
        max_y = max(y_coords) + cell_radius
        min_y = min(y_coords) - cell_radius
        bounding_indices.append((max_x, min_x, max_y, min_y))

    # use temp_img variable since python calls to global variables are very slow!
    temp_img = original_img.copy()
    for item in bounding_indices:
        for y in range(item[3], item[2]):
            for x in range(item[1], item[0]):
                if np.any(selections[y][x]):
                    if transparency:
                        temp_img[y][x][0] = np.uint8(tr_alpha*selections[y][x][0] + (1-tr_alpha)*original_img[y][x][0])
                        temp_img[y][x][1] = np.uint8(tr_alpha*selections[y][x][1] + (1-tr_alpha)*original_img[y][x][1])
                        temp_img[y][x][2] = np.uint8(tr_alpha*selections[y][x][2] + (1-tr_alpha)*original_img[y][x][2])
                    else:
                        temp_img[y][x] = np.uint8(selections[y][x])

    img = temp_img.copy()
"""


def create_guess():
    """
    Copies previous frame's segmentations into current frame as a guess for the
    cell segmentation optimizer.
    """
    
    global selections, cell_coords

    selections = image_buffer[current_image_index-1][0].copy()
    cell_coords = list(data_buffer[current_image_index-1][3])
    redraw_img(selections, original_img, cell_coords)


def move_guess(input_key):
    """
    Moves selection guess to allow for adjustments for field of view movements.
    """

    global selections, cell_coords

    if input_key == 117:    # Press "U" to move selection up one pixel
        selections = np.roll(selections, -1, axis=0)
        cell_coords = [((pair[0][0], pair[0][1]-1), (pair[1][0], pair[1][1]-1)) for pair in cell_coords] 
    if input_key == 104:    # Press "H" to move selection left one pixel
        selections = np.roll(selections, -1, axis=1)
        cell_coords = [((pair[0][0]-1, pair[0][1]), (pair[1][0]-1, pair[1][1])) for pair in cell_coords] 
    if input_key == 106:    # Press "J" to move selection down one pixel
        selections = np.roll(selections, 1, axis=0)
        cell_coords = [((pair[0][0], pair[0][1]+1), (pair[1][0], pair[1][1]+1)) for pair in cell_coords] 
    if input_key == 107:    # Press "K" to move selection right one pixel
        selections = np.roll(selections, 1, axis=1)
        cell_coords = [((pair[0][0]+1, pair[0][1]), (pair[1][0]+1, pair[1][1])) for pair in cell_coords]

    redraw_img(selections, original_img, cell_coords)
    

def execute_guess():
    """
    Uses the currently loaded cell selection as guesses for the cell selection
    optimization function circle_search().
    """

    global selections, cell_coords

    # store the current selection guess into temporary variables
    g_selections = selections.copy()
    g_cell_coords = list(cell_coords)

    # create a modified list of guess cell_coords with 2.93% increased cell length (for cell growth)
    # 2.93% is derived from 2^(1/24), assuming one cell doubling every 24 hours.
    # EDIT: changed it to 4% since guesses still underestimated cell length many times
    adj_cell_coords = []
    for pair in g_cell_coords:
        midpoint = ((pair[0][0] + pair[1][0])/2.0, (pair[0][1] + pair[1][1])/2.0)
        new_x1 = np.int64((pair[0][0] - midpoint[0])*0.04 + pair[0][0])
        new_y1 = np.int64((pair[0][1] - midpoint[1])*0.04 + pair[0][1])
        new_x2 = np.int64((pair[1][0] - midpoint[0])*0.04 + pair[1][0])
        new_y2 = np.int64((pair[1][1] - midpoint[1])*0.04 + pair[1][1])
        adj_cell_coords.append(((new_x1, new_y1), (new_x2, new_y2)))

    # restore selections and cell_coords variables to blank initial states
    selections = cv2.cvtColor(np.zeros(s_chan_img.shape, np.uint8), cv2.COLOR_GRAY2RGB)
    cell_coords = []

    # execute optimization function for each guessed cell
    for index, pair in enumerate(adj_cell_coords):
        draw_cell(circle_search(pair), redraw_flag = False)
        print "Finished guess for cell", index+1, "out of", len(adj_cell_coords)

    redraw_img(selections, original_img, cell_coords)

    print "Guess optimization complete."


def toggle_pause():
    """
    Toggles pause function for timekeeping.  Press "P" to pause/unpause timekeeping.
    """

    global paused

    if not paused:
        end_times.append(datetime.now())
        print "Program paused on", datetime.now().strftime("%c")
    if paused:
        start_times.append(datetime.now())
        print "Program unpaused on", datetime.now().strftime("%c")

    paused = not paused


def load_guess():
    """
    This function loads a cell selection saved from a previous session (specifically
    loading the data saved in the "cellcoords.npy" file) into the current frame. This
    is useful for starting a new session after completing a previous one.
    """

    global img, selections, cell_coords

    loaded_coords = np.load(expt_dir + "/" + load_file)
    cell_coords = [((pair[0][0], pair[0][1]), (pair[1][0], pair[1][1])) for pair in loaded_coords]
    selections = cv2.cvtColor(np.zeros(s_chan_img.shape, np.uint8), cv2.COLOR_GRAY2RGB)

    for pair in cell_coords:
        # generate random unique color for the cell
        cell_color = generate_color()

        # draw cell between the two indicated points in pair
        for point in pair:
            cv2.circle(selections, point, cell_radius, cell_color, -1)
        cv2.line(selections, pair[0], pair[1], cell_color, cell_radius*2)

    redraw_img(selections, original_img, cell_coords)
    

start_times.append(datetime.now())

# get list of input images
cwd = os.getcwd()
input_dir = cwd + "/" + expt_dir
input_images = _sort_nicely([x for x in os.listdir(input_dir) if "Brightfield-550nm.tif" in x])
input_images = input_images[bounds[0]-1:bounds[1]]

# initialize image storage variables
for i in range(len(input_images)):
    image_buffer.append(None)
    data_buffer.append(None)

# load first image and initialize window
initial_image = input_images[0]

load_image(current_image_index)

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_input)

# initialize the undo buffer
for i in range(undo_depth):
    undo_buffer.append((img.copy(), bwL))

# allow for cells to be drawn until "Esc" or "Q" key pressed
while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 100:    # press "D" to cycle to next image
        save_image(current_image_index)
        if current_image_index + 1 < len(input_images):
            current_image_index += 1
            load_image(current_image_index)
    if k == 97:     # press "A" to cycle to previous image
        save_image(current_image_index)
        if current_image_index - 1 >= 0:
            current_image_index -= 1
            load_image(current_image_index)
    if k == 115:    # press "S" to save current image to file
        save_image(current_image_index)
        save_image_to_file(current_image_index)
    if k == 119:    # press "W" to undo last change
        selections = undo_buffer[-2][0]
        bwL = undo_buffer[-2][1]
        undo_buffer.pop()
        undo_buffer.insert(0, (undo_buffer[0][0].copy(), undo_buffer[0][1].copy()))
        if len(cell_coords):
            cell_coords.pop()
        redraw_img(selections, original_img, cell_coords)
    if k == 102:    # press "F" to copy previous frame into current frame
        if current_image_index - 1 >= 0:
            reset_image(current_image_index)
            copy_image(current_image_index - 1)
        else:
            print "Cannot copy previous frame: at first frame already."
    if k == 103:    # press "G" to copy subsequent frame into current frame
        if current_image_index + 1 < len(input_images):
            if not image_buffer[current_image_index + 1]:
                print "Make a selection on the subsequent frame first."
            else:
                reset_image(current_image_index)
                copy_image(current_image_index + 1)
        else:
            print "Cannot copy subsequent frame: at last frame already."
    if k == 113:    # press "Q" to quit and save all images, including those not marked yet
        print "Quitting and saving..."
        save_image(current_image_index)
        # load the rest of the images into memory that haven't been loaded yet so they can be saved
        for i in range(len(input_images)):
            if not image_buffer[i]:
                load_image(i)
                save_image(i)
            save_image_to_file(i)
            output_text = "Saved image " + str(i+1) + " out of " + str(len(input_images))
            print output_text
        break
    if k == 114:    # press "R" to reset current image to initially loaded state
        reset_image(current_image_index)
    if k == 116:    # press "T" to toggle cell selection transparency
        transparency = not transparency
        redraw_img(selections, original_img, cell_coords)
    if k == 122:    # press "Z" to copy previous segmentations as guess for current frame
        if current_image_index - 1 >= 0:
            create_guess()
            guessing = True
    if k == 117 or k == 104 or k == 106 or k == 107:    # Press UHJK to move selection guess
        if guessing:
            move_guess(k)
    if k == 120:    # press "X" to execute guess optimization
        if guessing:
            execute_guess()
    if k == 112:    # press "P" to pause timekeeping
        toggle_pause()
    if k == 108:    # press "L" to load cell selection guess from a file
        load_guess()
        guessing = True
    if k == 27:     # press "Esc" to quit without saving changes
        break

cv2.destroyAllWindows()

# generate end of program statistics
if not paused:
    end_times.append(datetime.now())
total_runtime = timedelta(seconds=0)
for start, end in zip(start_times, end_times):
    total_runtime += end - start
total_cell_count = 0
for item in data_buffer:
    if item:
        total_cell_count += len(item[3])
if total_cell_count:
    s_per_selection = total_runtime.total_seconds()/total_cell_count
else:
    s_per_selection = "N/A"

print ""
print "End program statistics:"
print "---------------------------"
print "Runtime:", total_runtime, "seconds."
print "Cells selected:", total_cell_count
print "Seconds per selection:", s_per_selection

# output end of program statistics
if create_log:
    with open(expt_dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S.log"), "wb") as outputfile:
        outputfile.write("Expt directory:\n")
        outputfile.write(expt_dir + "\n\n")
        outputfile.write("Images analyzed:\n")
        for image in input_images:
            outputfile.write(image)
            outputfile.write("\n")
        outputfile.write("\n")
        outputfile.write("Cell search radius: " + str(search_radius) + "\n\n")
        outputfile.write("Image bounds: (" + str(bounds[0]) + ", " + str(bounds[1]) + ")\n\n")
        outputfile.write("End program statistics\n")
        outputfile.write("---------------------------\n")
        outputfile.write("Runtime: " + str(total_runtime) + " seconds.\n")
        outputfile.write("Cells selected: " + str(total_cell_count) + "\n")
        outputfile.write("Seconds per selection: " + str(s_per_selection))

