# Import pandas 
import pandas as pd 

# for plotting
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


import numpy as np
# for work with directories
import os
# for work withstrings and substrings:
import re

from experiment import src_results_folder as src_folder

# csv reader (epreriment results reader) function:
def get_experiment_results(src_folder):

    
    # all experiments in main dirname
    all_experiments = os.listdir(src_folder)
    # results array
    pd_results=[]
    #  labels(fftLengths) array
    fftLengths=[]
    for experiment in all_experiments:
        results = pd.read_csv(str(src_folder)+"/"+str(experiment)) 
        pd_results.append(results)
        # let's fetch fftlengths from file names
        fftLength = re.search('demo_dataset_results(.*).csv', experiment)
        fftLength = fftLength.group(1)
        fftLengths.append(fftLength)
    return fftLengths, pd_results

fftLengths, pd_results = get_experiment_results(src_folder)

# 
# Make data.
# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
Time = [] # X
Allow_true = [] # Y1
Allow_false = [] # Y2
FftLength = [] # Z
All_Count = []
for index, experiment in enumerate(pd_results):
    Time = np.append(Time, experiment.TIME_TO_SPLIT, axis=0)
    Allow_true = np.append(Allow_true, experiment.Allow_true, axis=0)
    Allow_false = np.append(Allow_false, experiment.Allow_false, axis=0)
    All_Count = np.append(All_Count, experiment.all_count, axis=0)


    fftLength_part = np.full(len(experiment.Allow_true), fftLengths[index])
    FftLength = np.append(FftLength, fftLength_part, axis=0)

FftLength = FftLength.astype(np.float)
# let's create one aggrigate pandas:
experimant_results = pd.DataFrame({'Time':Time,'FftLength':FftLength, 'Allow_true':Allow_true, 'Allow_false':Allow_false, 'All_Count':All_Count})
# let's calculate optimal values from experiment
# max value for Allow_true through all dataset
true_max = np.max(experimant_results.Allow_true)
false_min = np.min(experimant_results.Allow_false)
# optimal parameters:
experimant_results_optimal = experimant_results[(experimant_results.Allow_true>=(true_max-2)) & (experimant_results.Allow_false<=(false_min+2))]
# print(experimant_results_optimal)

# 3D:
fig = plt.figure()
ax = fig.gca(projection='3d')

# # plot scatter (points):
# ax.scatter(Time, FftLength, Allow_true, marker="o")

# plot trisurf:
x=Time
y=FftLength
z1=Allow_true
z2= Allow_false
surf = ax.plot_trisurf(x, y, z2, cmap=cm.jet, linewidth=0.1)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)


# ax.set_ylim(0, 33000)
ax.set_xlabel('Time Label')
ax.set_ylabel('FftLength Label')
ax.set_zlabel('Allow_false Label')

plt.show()



# # 2D:
# # reading csv file  
# results = pd.read_csv("demo_dataset_results.csv") 

# # data to plot
# # count of columns
# n_groups = len(results.TIME_TO_SPLIT)

# # create plot
# fig, ax = plt.subplots()
# index = np.arange(n_groups)
# bar_width = 0.35
# opacity = 0.8

# # allow-true
# rects1 = plt.bar(index, results.Allow_true, bar_width,
# alpha=opacity,
# color='b',
# label='Allow-true')
# # allow-false
# rects2 = plt.bar(index + bar_width, results.Allow_false, bar_width,
# alpha=opacity,
# color='g',
# label='Allow-false')

# plt.xlabel('Довжина фрази (сек)')
# plt.ylabel('Кількість експериментів із загальної - 300')
# plt.title('Статистика відносно довжини фрази')
# plt.xticks(index, results.TIME_TO_SPLIT)
# plt.legend()

# plt.tight_layout()
# plt.show()


