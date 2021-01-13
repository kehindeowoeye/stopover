import matplotlib
#matplotlib.use('TKAgg',warn=False, force=True)
import matplotlib.pyplot as plt
import matplotlib.style
plt.style.use('classic')
matplotlib.style.use('ggplot')
import pandas as pd
from collections import OrderedDict
import seaborn as sns
from matplotlib import rc,rcParams
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import sem, t
from matplotlib import rc,rcParams
#rc('text', usetex=True)
#rc('axes', linewidth=2)
#rc('font', weight='bold')
rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

plt.rcParams["figure.figsize"] = [15, 7]
rc('text', usetex=True)
rc('axes', linewidth=2)
rc('font', weight='bold')
rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

blue_patch = mpatches.Patch(color='blue', label='Coastal Lagoons, Estuaries, Sea and ocean')
red_patch = mpatches.Patch(color='red',   label='Coniferous forest')
green_patch = mpatches.Patch(color='lime', label='Mixed forest')
magenta_patch = mpatches.Patch(color='magenta', label='Water courses, Water bodies')




df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[2], 'Coniferous forest': [0], 'Mixed forest': [0.28], 'Water courses, Water bodies':[0]} )

fig, ax = plt.subplots()
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.01, position=1.5, colormap="brg", ax=ax, alpha=0.7)


df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[0.84], 'Coniferous forest': [0], 'Mixed forest': [0.28], 'Water courses, Water bodies':[0.28]} )
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest','Water courses, Water bodies']].plot.bar(stacked=True, width=0.01, position=2.5, color = ('b','r','chartreuse','m'), ax=ax, alpha=0.7)


df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[1.4], 'Coniferous forest': [0], 'Mixed forest': [0], 'Water courses, Water bodies':[0]} )

df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest','Water courses, Water bodies']].plot.bar(stacked=True, width=0.01, position=3.5, colormap="brg", ax=ax, alpha=0.7)


df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[2], 'Coniferous forest': [0], 'Mixed forest': [0.28], 'Water courses, Water bodies':[0]} )


df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.01, position=4.5, colormap="brg", ax=ax, alpha=0.7)


df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[1.68], 'Coniferous forest': [0.28], 'Mixed forest': [0.28], 'Water courses, Water bodies':[0]} )


df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.01, position=5.5, colormap="brg", ax=ax, alpha=0.7)


df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[1.82], 'Coniferous forest': [0.28], 'Mixed forest': [0.28], 'Water courses, Water bodies':[0]} )
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.01, position=6.5, colormap="brg", ax=ax, alpha=0.7)


df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[0.84], 'Coniferous forest': [0], 'Mixed forest': [0], 'Water courses, Water bodies':[0.28]} )

df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest','Water courses, Water bodies']].plot.bar(stacked=True, width=0.01, position=7.5, color = ('b','m'), ax=ax, alpha=0.7)


df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[1.54], 'Coniferous forest': [0.28], 'Mixed forest': [0.42], 'Water courses, Water bodies':[0]} )

df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.01, position=8.5, colormap="brg", ax=ax, alpha=0.7)



df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[1.792], 'Coniferous forest': [0.28], 'Mixed forest': [0.28], 'Water courses, Water bodies':[0]} )

df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.01, position=9.5, colormap="brg", ax=ax, alpha=0.7)




handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
legend_properties = {'weight':'bold','size':12}
plt.legend(handles=[blue_patch, red_patch, green_patch, magenta_patch])
plt.title('Spring Migration- Turkey Vulture',fontsize=14, fontweight="bold")
plt.ylim([0, 2.99])
#plt.yticks([], [])
plt.xticks([], [])
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(2.8))
plt.savefig('springtk.eps',dpi = 140)



plt.rcParams["figure.figsize"] = [15,7]
rc('text', usetex=True)
rc('axes', linewidth=2)
rc('font', weight='bold')
rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']




df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[1.4], 'Coniferous forest': [0.28], 'Mixed forest': [0.28], 'Water courses, Water bodies':[0]} )
fig, ax = plt.subplots()
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.01, position=1.5, colormap="brg", ax=ax, alpha=0.7)


df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[1.6], 'Coniferous forest': [0.28], 'Mixed forest': [0.28], 'Water courses, Water bodies':[0]} )
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.01, position=2.5, colormap="brg", ax=ax, alpha=0.7)


df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[1.484], 'Coniferous forest': [0.28], 'Mixed forest': [0.28], 'Water courses, Water bodies':[0]} )
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.01, position=3.5, colormap="brg", ax=ax, alpha=0.7)


df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[1.4], 'Coniferous forest': [0], 'Mixed forest': [0], 'Water courses, Water bodies':[0]} )
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.01, position=4.5, colormap="brg", ax=ax, alpha=0.7)


df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[1.54], 'Coniferous forest': [0], 'Mixed forest': [0], 'Water courses, Water bodies':[0]} )
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.01, position=5.5, colormap="brg", ax=ax, alpha=0.7)


df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[1.6], 'Coniferous forest': [0.28], 'Mixed forest': [0.28], 'Water courses, Water bodies':[0]} )
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.01, position=6.5, colormap="brg", ax=ax, alpha=0.7)


df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[2], 'Coniferous forest': [0], 'Mixed forest': [0.28], 'Water courses, Water bodies':[0]} )
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.01, position=7.5, colormap="brg", ax=ax, alpha=0.7)



df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[1.736], 'Coniferous forest': [0.28], 'Mixed forest': [0.28], 'Water courses, Water bodies':[0]} )
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.01, position=8.5, colormap="brg", ax=ax, alpha=0.7)


df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[2.2092], 'Coniferous forest': [0], 'Mixed forest': [0], 'Water courses, Water bodies':[0]} )
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.01, position=9.5, colormap="brg", ax=ax, alpha=0.7)


df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[1.68], 'Coniferous forest': [0.28], 'Mixed forest': [0.392], 'Water courses, Water bodies':[0]} )
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.01, position=10.5, colormap="brg", ax=ax, alpha=0.7)


df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[0.84], 'Coniferous forest': [0], 'Mixed forest': [0], 'Water courses, Water bodies':[0.28]} )
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest','Water courses, Water bodies']].plot.bar(stacked=True, width=0.01, position=11.5, color = ('b','r','chartreuse','m'), ax=ax, alpha=0.7)


df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[1.54], 'Coniferous forest': [0.336], 'Mixed forest': [0.42], 'Water courses, Water bodies':[0]} )
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.01, position=12.5, colormap="brg", ax=ax, alpha=0.7)



df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[1.792], 'Coniferous forest': [0.28], 'Mixed forest': [0.364], 'Water courses, Water bodies':[0]} )
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.01, position=13.5, colormap="brg", ax=ax, alpha=0.7)


handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
legend_properties = {'weight':'bold','size':8}
plt.legend(handles=[blue_patch, red_patch, green_patch, magenta_patch])
plt.title('Fall Migration- Turkey Vulture',fontsize=14, fontweight="bold")
plt.ylim([0, 2.99])
#plt.yticks([], [])
plt.xticks([], [])
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(2.8))
plt.ylabel('\% Composition of each habitat',fontsize=18, fontweight="bold")
plt.savefig('falltk.eps',dpi = 140)


