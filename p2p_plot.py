"""
Justin Chew
Rust Lab

This script plots the peak to peak statistics for a given experiment.

Similar to sync_index_plot.py, this must be run after lineage_plot.py
with pickle_lineage = True to work.
"""

import matplotlib.pyplot as plt
import peakdetect
import cPickle as pickle
import numpy as np
import os
import errno

expt_name = "12.4.16 pJC073-2 23 92 368 uM theo redo"

export_peak_imgs = False
max_interpeak = 4

# Suppresses any saved output and prints outlier p2p interval information
# Use this flag to look at whether there was any errant peak identification
# for outliers in the p2p histogram for 73-2-368 and to verify for the
# other conditions that the longer period p2ps are not errors either.
p2p_diagnostic = False

# Toggle this if doing an antibiotic cell elongation expt to discard peaks
# where the first peak happens before antibiotic administration
ab_flag = False
ab_time = 36


################################
## Function definitions below ##
################################

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    # from http://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def makedir(dir_name):
    """
    makedir creates the specified directory path if it doesn't exist
    """
    try:
        os.makedirs(dir_name)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def calc_stats(p2p_diff_list):
    """
    Returns statistics about the input peak to peak distribution list,
    specifically calculating the mean, standard deviation, and the
    coefficient of variation.
    """

    mean = np.mean(p2p_diff_list)
    std = np.std(p2p_diff_list)
    cov = std/mean

    return (mean, std, cov)


def extract_conditions(folders):
    """
    extract_conditions returns a list of microscope conditions for
    aggregating data, e.g. "53-2-6" or "YFP-181".  For this function to
    work properly, all folder names MUST be in the following format:
    condition_name-pos_num, where the position number is separated by a
    hyphen.
    
    Hence, the following folders are grouped together into a condition:
    "53-2-6-1", "53-2-6-2", "53-2-6-3" all become "53-2-6".
    """

    suffixes_removed = []
    
    for item in folders:
        parsed_item = item.split("-")
        suffixes_removed.append("-".join(parsed_item[:-1]))
    
    conditions = list(set(suffixes_removed))
    
    return conditions


def get_avg_len(lineage_data, p2p_ends):
    """
    Given a set of peak to peaks, this function returns the average cell
    length over the time period between the two peaks.
    """

    times1 = lineage_data[lineage_data[:,2] >= p2p_ends[0]]
    times2 = times1[times1[:,2] <= p2p_ends[1]]

    return np.mean(times2[:,5])
    

###############################
## Begin code execution here ##
###############################

original_cwd = os.getcwd()

# get all the input folder names
folders = [x for x in os.listdir("output/" + expt_name + "/lineages/") if not "sync" in x and not "p2p" in x and not "div" in x]

for fPath in folders:

    work_dir = original_cwd + "/output/" + expt_name + "/lineages/" + fPath
    os.chdir(work_dir)

    # get list of pickle files in fPath
    lineage_files = [x for x in os.listdir(work_dir) if (".pickle" in x and not "p2p" in x and not "SI" in x and not "div" in x)]

    for lfile in lineage_files:
        with open(lfile, "rb") as inputfile:
            pickle_contents = pickle.load(inputfile)

        # open lineage and extract lineage information
        lineages = pickle_contents[0]

        # hold p2p data tuples for 1, 2, and 3 peak interpeak time measurements
        p2p_data = []
        for i in range(max_interpeak):
            p2p_data.append([])

        if export_peak_imgs:
            makedir(work_dir + "/" + "p2p output " + lfile[:-7])

        for lcounter, lineage in enumerate(lineages):

            time = lineage[:,2]
            intensity = lineage[:,8]

            # smooth the lineage intensity data for peak detection
            smoothed = savitzky_golay(intensity, window_size=11, order=3)
            
            # calculate peak to peak data
            _max, _min = peakdetect.peakdetect(smoothed, time, lookahead=1, delta=100)
            peak_t = [p[0] for p in _max]
            peak_val = [p[1] for p in _max]
            trough_t = [p[0] for p in _min]
            trough_val = [p[1] for p in _min]

            # plot peaks if export_peak_imgs is True
            if export_peak_imgs:
                fig, ax = plt.subplots()
                ax.plot(time, intensity)
                ax.plot(time, smoothed)
                ax.plot(peak_t, peak_val, "r+")
                fig.savefig(work_dir + "/" + "p2p output " + lfile[:-7] + "/lineage " + str(lcounter) + " peak plot.png", format="png")
                plt.close()

            # calculate peak to peak measurements
            for p in range(1, max_interpeak+1):
                for i in range(p, len(peak_val)):
                    p2p_diff = peak_t[i] - peak_t[i-p]
                    # select row based on time, from http://stackoverflow.com/questions/1962980/selecting-rows-from-a-numpy-ndarray
                    cellID1 = lineage[lineage[:,2] == peak_t[i-p]][0][3]
                    cellID2 = lineage[lineage[:,2] == peak_t[i]][0][3]
                    avg_cell_len = get_avg_len(lineage, (peak_t[i-p], peak_t[i]))
                    # grab the trough in between the two peaks, doesn't work for interpeak > 1
                    trough = None
                    for index, t in enumerate(trough_t):
                        if t > peak_t[i-p] and t < peak_t[i]:
                            trough = trough_val[index]
                    amplitude = peak_val[i] - trough
                    p2p_tuple = (p2p_diff, peak_t[i], cellID1, cellID2, avg_cell_len, amplitude)
                    if p2p_tuple not in p2p_data[p-1]:
                        if ab_flag and peak_t[i-p] > ab_time:
                            p2p_data[p-1].append(p2p_tuple)
                            if p2p_diagnostic:
                                if p == 1:
                                    lfile_short = " ".join(lfile.split(" ")[0:3])
                                    p2p_list = list(p2p_tuple)
                                    p2p_list.insert(1, str(peak_t[i-1]))
                                    p2p_tuple_str = " ".join([str(x) for x in p2p_list])
                                    print fPath, lfile_short, lcounter, p2p_tuple
                                    makedir(work_dir + "/" + "p2p diagnostic " + lfile[:-7])
                                    fig, ax = plt.subplots()
                                    ax.plot(time, intensity)
                                    ax.plot(time, smoothed)
                                    ax.axvline(peak_t[i])
                                    ax.axvline(peak_t[i-p])
                                    #ax.plot(peak_t, peak_val, "r+")
                                    ax.set_title("p2p, t_i, t_f, cell1, cell2\n" + p2p_tuple_str)
                                    fig.savefig(work_dir + "/" + "p2p diagnostic " + lfile[:-7] + "/lineage " + str(lcounter) + " " + p2p_tuple_str + " peak plot.png", format="png")
                                    plt.close()
                        elif not ab_flag:
                            p2p_data[p-1].append(p2p_tuple)
                            #if p == 1:
                                #print p2p_tuple
                        """ # Use for figuring out whether long p2p intervals are legit
                        if p2p_diagnostic:
                            if p == 1 and p2p_diff > 40:
                                lfile_short = " ".join(lfile.split(" ")[0:3])
                                p2p_list = list(p2p_tuple)
                                p2p_list.insert(1, str(peak_t[i-1]))
                                p2p_tuple_str = " ".join([str(x) for x in p2p_list])
                                print fPath, lfile_short, lcounter, p2p_tuple
                                makedir(work_dir + "/" + "p2p diagnostic " + lfile[:-7])
                                fig, ax = plt.subplots()
                                ax.plot(time, intensity)
                                ax.plot(time, smoothed)
                                ax.plot(peak_t, peak_val, "r+")
                                ax.set_title("p2p, t_i, t_f, cell1, cell2\n" + p2p_tuple_str)
                                fig.savefig(work_dir + "/" + "p2p diagnostic " + lfile[:-7] + "/lineage " + str(lcounter) + " " + p2p_tuple_str + " peak plot.png", format="png")
                                plt.close()
                        """

        # calculate some statistics about the p2p distributions
        lineage_p2p_diffs = []
        lineage_p2p_celllens = []
        lineage_p2p_amps = []
        lin_stats = []
        for i in range(max_interpeak):
            temp_p2p_diffs = [x[0] for x in p2p_data[i]]
            temp_p2p_celllens = [x[4] for x in p2p_data[i]]
            temp_p2p_amps = [x[5] for x in p2p_data[i]]
            lineage_p2p_diffs.append(temp_p2p_diffs)
            lineage_p2p_celllens.append(temp_p2p_celllens)
            lineage_p2p_amps.append(temp_p2p_amps)
            lin_stats.append(calc_stats(temp_p2p_diffs))

        if not p2p_diagnostic:
            # create plot for single peak to peak data
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.hist(lineage_p2p_diffs[0], 30, range=(0.0, 60.0), normed=True)
            ax.set_ylabel("Frequency")
            ax.set_xlabel("Peak to peak time (h)")
            ax.set_xlim(0.0, 60.0)
            title = "Peak to peak distribution test for " + lfile[:-7] + "\nTotal number of peak to peak differences = " + str(len(lineage_p2p_diffs[0])) + "\nMean = " + str(round(lin_stats[0][0], 3)) + ", std = " + str(round(lin_stats[0][1],3)) + ", cov = " + str(round(lin_stats[0][2],3))
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(work_dir + "/" + lfile[:-7] + " p2p histogram.png", format="png")
            plt.close()

            # create plot for multiple peak to peak data
            n = [len(x) for x in lineage_p2p_diffs]
            means = [x[0] for x in lin_stats]
            stdevs = [x[1] for x in lin_stats]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.errorbar(range(1, max_interpeak+1), means, yerr=stdevs, marker="o")
            for i, j in enumerate(n):
                ax.annotate("n = " + str(j), (i+1, means[i]), textcoords="offset pixels", xytext=(8, -12), fontsize=10)
            ax.set_title("Peak to peak time means and stdevs for " + lfile[:-7])
            ax.set_ylabel("Time (h)")
            ax.set_xlabel("Number of interpeaks")
            ax.set_xlim(0, max_interpeak+1)
            ax.set_ylim(0, max(means)*1.2)
            fig.savefig(work_dir + "/" + lfile[:-7] + " p2p means and stdevs.png", format="png")
            plt.close()
            
            # create plot of stdev vs number of interpeaks
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(range(1, max_interpeak+1), stdevs, marker="o")
            for i, j in enumerate(n):
                ax.annotate("n = " + str(j), (i+1, stdevs[i]), textcoords="offset pixels", xytext=(8, -12), fontsize=10)
            ax.set_title("Number of interpeaks vs stdevs for " + lfile[:-7])
            ax.set_ylabel("Time (h)")
            ax.set_xlabel("Number of interpeaks")
            ax.set_xlim(0, max_interpeak+1)
            ax.set_ylim(0, max(stdevs)*1.1)
            fig.savefig(work_dir + "/" + lfile[:-7] + " p2p stdevs.png", format="png")
            plt.close()
            
            # dump pickle file with p2p data/statistics
            with open("p2p data " + lfile, "wb") as outputfile:
                pickle.dump((lineage_p2p_diffs, lin_stats, lineage_p2p_celllens, lineage_p2p_amps), outputfile)

    print "Finished calculating p2p data for", fPath

########################################
## Calculate aggregate p2p statistics ##
########################################

if not p2p_diagnostic:

    # create output directory
    total_output_dir = original_cwd + "/output/" + expt_name + "/lineages/total_p2p_stats"
    makedir(total_output_dir)

    # generate conditions
    condition_input_folders = [x for x in os.listdir(original_cwd + "/output/" + expt_name + "/lineages/") if not "avg_sync_index" in x and not "p2p" in x and not "div" in x]
    conditions = extract_conditions(condition_input_folders)

    # start plot for all the mean/stdev data together
    msfig = plt.figure()
    msax = msfig.add_subplot(111)
    msax.set_title("Peak to peak time means and stdevs for all conditions")
    msax.set_ylabel("Time (h)")
    msax.set_xlabel("Number of interpeaks")
    msax.set_xlim(0, max_interpeak+1)

    # start plot for all stdev data together
    sfig = plt.figure()
    sax = sfig.add_subplot(111)
    sax.set_title("Number of interpeaks vs stdevs for all conditions")
    sax.set_ylabel("Time (h)")
    sax.set_xlabel("Number of interpeaks")
    sax.set_xlim(0, max_interpeak+1)

    for condition in conditions:
        
        folders = [x for x in os.listdir(original_cwd + "/output/" + expt_name + "/lineages/") if condition in x]

        # calculate some statistics about the p2p distributions
        all_p2p_diffs = []
        all_p2p_celllens = []
        all_p2p_amps = []
        for i in range(max_interpeak):
            all_p2p_diffs.append([])
            all_p2p_celllens.append([])
            all_p2p_amps.append([])

        for fPath in folders:        
            work_dir = original_cwd + "/output/" + expt_name + "/lineages/" + fPath
            p2p_files = [x for x in os.listdir(work_dir) if (".pickle" in x and "p2p" in x)]

            for p2p_file in p2p_files:
                with open(work_dir + "/" + p2p_file, "rb") as inputfile:
                    loaded_data = pickle.load(inputfile)
                for i in range(max_interpeak):
                    try:
                        all_p2p_diffs[i].extend(loaded_data[0][i])
                        all_p2p_celllens[i].extend(loaded_data[2][i])
                        all_p2p_amps[i].extend(loaded_data[3][i])
                    except:
                        pass

        # calculate statistics for total histogram
        all_stats = []
        for i in range(max_interpeak):
            all_stats.append(calc_stats(all_p2p_diffs[i]))

        # create plot for all mother cell data pooled together for single peak to peak histogram
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(all_p2p_diffs[0], 30, range=(0.0, 60.0), normed=True)
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Peak to peak time (h)")
        ax.set_xlim(0.0, 60.0)
        title = "Peak to peak distribution for " + condition + "\nTotal number of peak differences = " + str(len(all_p2p_diffs[0])) + "\nMean = " + str(round(all_stats[0][0],3)) + ", std = " + str(round(all_stats[0][1],3)) + ", cov = " + str(round(all_stats[0][2],3))
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(total_output_dir + "/" + condition + " total p2p histogram.png", format="png")

        # create plot for multiple peak to peak data
        n = [len(x) for x in all_p2p_diffs]
        means = [x[0] for x in all_stats]
        stdevs = [x[1] for x in all_stats]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for item in [ax, msax]:
            item.errorbar(range(1, max_interpeak+1), means, yerr=stdevs, marker="o", label=condition)
            for i, j in enumerate(n):
                item.annotate("n = " + str(j), (i+1, means[i]), textcoords="offset pixels", xytext=(8, -12), fontsize=10)
        ax.set_title("Peak to peak time means and stdevs for " + condition)
        ax.set_ylabel("Time (h)")
        ax.set_xlabel("Number of interpeaks")
        ax.set_xlim(0, max_interpeak+1)
        ax.set_ylim(0, max(means)*1.2)
        fig.savefig(total_output_dir + "/" + condition + " p2p means and stdevs.png", format="png")
        
        # create plot of stdev vs number of interpeaks
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for item in [ax, sax]:
            item.plot(range(1, max_interpeak+1), stdevs, marker="o", label=condition)
            for i, j in enumerate(n):
                item.annotate("n = " + str(j), (i+1, stdevs[i]), textcoords="offset pixels", xytext=(8, -12), fontsize=10)
        ax.set_title("Number of interpeaks vs stdevs for " + condition)
        ax.set_ylabel("Time (h)")
        ax.set_xlabel("Number of interpeaks")
        ax.set_xlim(0, max_interpeak+1)
        ax.set_ylim(0, max(stdevs)*1.1)
        fig.savefig(total_output_dir + "/" + condition + " p2p stdevs.png", format="png")

        print total_output_dir + "/" + condition + " p2p data.pickle"

        # dump pickle file with total p2p data/statistics
        with open(total_output_dir + "/" + condition + " p2p data.pickle", "wb") as outputfile:
            pickle.dump((all_p2p_diffs, all_stats, all_p2p_celllens, all_p2p_amps), outputfile)

    msax.set_ylim(ymin=0)
    msax.legend(loc="lower right")
    msfig.savefig(total_output_dir + "/" + "all p2p means and stdevs.png", format="png")

    sax.set_ylim(ymin=0)
    sax.legend(loc="center right")
    sfig.savefig(total_output_dir + "/" + "all p2p stdevs.png", format="png")
    plt.close()
