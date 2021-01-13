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
plt.rcParams["figure.figsize"] = [15,5]
#rc('text', usetex=True)
#rc('axes', linewidth=2)
#rc('font', weight='bold')
rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

legend_properties = {'weight':'bold'}
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


#sns.set(style="whitegrid")
rc('font', weight='bold')
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


#df = pd.DataFrame({'a':[2, 2], 'b': [0, 25], 'c': [35, 40], 'd':[45, 50]}, index=['john', 'bob'])
df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[1.736, 1.12, 1.736, 1.064, 1.176], 'Coniferous forest': [0.28, 0.28, 0.28, 0.28, 0.28], 'Mixed forest': [0.28, 0.28, 0.28, 0.28, 0.28], 'Water courses, Water bodies':[1, 1, 1, 1, 1]},  index=['2006', '2007', '2008', '2009', '2013'] )

fig, ax = plt.subplots()
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.05, position=1.5, colormap="brg", ax=ax, alpha=0.7)


##################

df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[1.484, 1.736, 1.736, 1.064, 0], 'Coniferous forest': [0.28, 0.28, 0.28, 0.28, 0], 'Mixed forest': [0.28, 0.28, 0.28, 0.28, 0], 'Water courses, Water bodies':[1, 1, 1, 1, 1]},  index=['2006', '2007', '2008', '2009', '2013'] )


df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.05, position=2.5, colormap="brg", ax=ax, alpha=0.7)


##################


df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[1.596, 1.736, 0, 1.736, 0], 'Coniferous forest': [0.28, 0.28, 0, 0.28, 0], 'Mixed forest': [0.28, 0.28, 0, 0.28, 0], 'Water courses, Water bodies':[1, 1, 1, 1, 1]},  index=['2006', '2007', '2008', '2009', '2013'] )

df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.05, position=3.5, colormap="brg", ax=ax, alpha=0.7)


##################


df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[1.064, 0, 0, 0, 0], 'Coniferous forest': [0.28, 0, 0, 0, 0], 'Mixed forest': [0.28, 0, 0, 0, 0], 'Water courses, Water bodies':[1, 1, 1, 1, 1]},  index=['2006', '2007', '2008', '2009', '2013'] )

df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.05, position=4.5, colormap="brg", ax=ax, alpha=0.7)

##################


df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[1.12, 0, 0, 0, 0], 'Coniferous forest': [0.28, 0, 0, 0, 0], 'Mixed forest': [0.28, 0, 0, 0, 0], 'Water courses, Water bodies':[1, 1, 1, 1, 1]},  index=['2006', '2007', '2008', '2009', '2013'] )

df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.05, position=5.5, colormap="brg", ax=ax, alpha=0.7)

##################


df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[1.736, 0, 0, 0, 0], 'Coniferous forest': [0.28, 0, 0, 0, 0], 'Mixed forest': [0.28, 0, 0, 0, 0], 'Water courses, Water bodies':[1, 1, 1, 1, 1]},  index=['2006', '2007', '2008', '2009', '2013'] )

df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.05, position=6.5, colormap="brg", ax=ax, alpha=0.7)

##################
##################
#positioning

df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[2.128, 2.156, 0, 0, 0], 'Coniferous forest': [0.28, 1, 0, 1, 1], 'Mixed forest': [0.28, 1, 0, 1, 1], 'Water courses, Water bodies':[1, 1, 1, 1, 1]},  index=['2006', '2007', '2008', '2009', '2013'] )
df[['Coastal Lagoons, Estuaries, Sea and ocean']].plot.bar(stacked=False, width=0.05, position=7.5, colormap="brg", ax=ax, alpha=0.7)


df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[0, 2.156, 0, 0, 0], 'Coniferous forest': [1, 1, 1, 1, 1], 'Mixed forest': [1, 1, 1, 1, 1], 'Water courses, Water bodies':[1, 1, 1, 1, 1]},  index=['2006', '2007', '2008', '2009', '2013'] )
df[['Coastal Lagoons, Estuaries, Sea and ocean']].plot.bar(stacked=False, width=0.05, position=4.5, colormap="brg", ax=ax, alpha=0.7)


df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[0, 2, 0, 0, 0], 'Coniferous forest': [1, 1, 1, 1, 1], 'Mixed forest': [1, 1, 1, 1, 1], 'Water courses, Water bodies':[1, 1, 1, 1, 1]},  index=['2006', '2007', '2008', '2009', '2013'] )
df[['Coastal Lagoons, Estuaries, Sea and ocean']].plot.bar(stacked=False, width=0.05, position=5.5, colormap="brg", ax=ax, alpha=0.7)



df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[0, 0, 0, 2.072, 0], 'Coniferous forest': [1, 1, 1, 1, 1], 'Mixed forest': [1, 1, 1, 1, 1], 'Water courses, Water bodies':[1, 1, 1, 1, 1]},  index=['2006', '2007', '2008', '2009', '2013'] )
df[['Coastal Lagoons, Estuaries, Sea and ocean']].plot.bar(stacked=False, width=0.05, position=4.5, colormap="brg", ax=ax, alpha=0.7)


df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[0, 0, 0, 0, 2.072], 'Coniferous forest': [1, 1, 1, 1, 1], 'Mixed forest': [1, 1, 1, 1, 1], 'Water courses, Water bodies':[1, 1, 1, 1, 1]},  index=['2006', '2007', '2008', '2009', '2013'] )
df[['Coastal Lagoons, Estuaries, Sea and ocean']].plot.bar(stacked=False, width=0.05, position=2.5, colormap="brg", ax=ax, alpha=0.7)

#####################################################################
#positioning
df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[0, 1.96, 0, 0, 0], 'Coniferous forest': [0, 0, 0, 0, 0], 'Mixed forest': [0, 0.28, 0, 0, 0], 'Water courses, Water bodies':[1, 1, 1, 1, 1]},  index=['2006', '2007', '2008', '2009', '2013'] )


df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Mixed forest']].plot.bar(stacked=True, width=0.05, position=6.5, colormap="brg", ax=ax, alpha=0.7)
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Mixed forest']].plot.bar(stacked=True, width=0.05, position=7.5, colormap="brg", ax=ax, alpha=0.7)
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Mixed forest']].plot.bar(stacked=True, width=0.05, position=8.5, colormap="brg", ax=ax, alpha=0.7)
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Mixed forest']].plot.bar(stacked=True, width=0.05, position=9.5, colormap="brg", ax=ax, alpha=0.7)



df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[0, 0, 1.848, 0, 0], 'Coniferous forest': [0, 0, 0.28, 0, 0], 'Mixed forest': [0, 0, 0.28, 0, 0], 'Water courses, Water bodies':[1, 1, 1, 1, 1]},  index=['2006', '2007', '2008', '2009', '2013'] )
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Mixed forest']].plot.bar(stacked=True, width=0.05, position=3.5, colormap="brg", ax=ax, alpha=0.7)
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Mixed forest']].plot.bar(stacked=True, width=0.05, position=4.5, colormap="brg", ax=ax, alpha=0.7)
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Mixed forest']].plot.bar(stacked=True, width=0.05, position=5.5, colormap="brg", ax=ax, alpha=0.7)


df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[0, 0, 0, 0, 1.876], 'Coniferous forest': [1, 0, 0, 0, 0], 'Mixed forest': [0, 0, 0, 0, 0.28], 'Water courses, Water bodies':[1, 1, 1, 1, 1]},  index=['2006', '2007', '2008', '2009', '2013'] )
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Mixed forest']].plot.bar(stacked=True, width=0.05, position=3.5, colormap="brg", ax=ax, alpha=0.7)

##################################################################################

df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[0.784, 0, 0, 0, 0], 'Coniferous forest': [0.28, 0, 0, 0, 0], 'Mixed forest': [0.28, 0, 0, 0, 0], 'Water courses, Water bodies':[0.392, 0, 0, 0, 0]},  index=['2006', '2007', '2008', '2009', '2013'] )
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Water courses, Water bodies']].plot.bar(stacked=True, width=0.05, position=8.5, color = ('b','m'), ax=ax, alpha=0.7)



handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
legend_properties = {'weight':'bold','size':18}
plt.legend(handles=[blue_patch, red_patch, green_patch, magenta_patch])
#plt.legend(by_label.values(), by_label.keys(), prop=legend_properties )
#plt.legend(loc="upper center")
plt.title('Spring Migration',fontsize=18, fontweight="bold")
plt.ylim([0, 2.99])
#plt.yticks([], [])
#plt.xticks([], [])
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(2.8))
plt.xlabel('Year',fontsize=18, fontweight="bold")
plt.ylabel('\% Composition of each habitat',fontsize=18, fontweight="bold")
#ax.yaxis.set_tick_params(labelsize=18)
ax.xaxis.set_tick_params(labelsize=18)
#ax.set_xticklabels(xlabels, rotation=45, ha='right')
plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha = 'center')
#ax.legend().remove()

plt.tick_params(labelsize = 20)
for tick in ax.xaxis.get_majorticklabels():
    tick.set_horizontalalignment("right")

plt.savefig('spring.eps',dpi = 140)













plt.rcParams["figure.figsize"] = [15, 7]
rc('text', usetex=True)
rc('axes', linewidth=2)
rc('font', weight='bold')
rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
#########################################################################################

df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[1.316, 1.456, 1.4, 1.4, 1.68], 'Coniferous forest': [0.28, 0.28, 0.28, 0.28, 0.28], 'Mixed forest': [0.28, 0.28, 0.28, 0.28, 0.28], 'Water courses, Water bodies':[1, 1, 1, 1, 1]},  index=['2006', '2007', '2008', '2009', '2013'] )


fig, ax = plt.subplots()
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.05, position=1.5, colormap="brg", ax=ax, alpha=0.7)


df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[1.68, 0, 1.54, 0, 0], 'Coniferous forest': [0.28, 0, 0.28, 0, 0], 'Mixed forest': [0.28, 0, 0.28, 0, 0], 'Water courses, Water bodies':[1, 1, 1, 1, 1]},  index=['2006', '2007', '2008', '2009', '2013'] )
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.05, position=2.5, colormap="brg", ax=ax, alpha=0.7)
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.05, position=3.5, colormap="brg", ax=ax, alpha=0.7)
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Coniferous forest', 'Mixed forest']].plot.bar(stacked=True, width=0.05, position=4.5, colormap="brg", ax=ax, alpha=0.7)
####################################################
#positioning
df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[1.96, 2.044, 0, 2.1, 1.96], 'Coniferous forest': [0.28, 0.28, 0, 0.28, 0.28], 'Mixed forest': [0.28, 0.28, 0, 0.28, 0.28], 'Water courses, Water bodies':[1, 1, 1, 1, 1]},  index=['2006', '2007', '2008', '2009', '2013'] )

df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Mixed forest']].plot.bar(stacked=True, width=0.05, position=2.5, colormap="brg", ax=ax, alpha=0.7)



df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[1.96, 0, 0, 0, 0], 'Coniferous forest': [0.28, 0, 0, 0, 0], 'Mixed forest': [0.28, 0, 0, 0, 0], 'Water courses, Water bodies':[1, 1, 1, 1, 1]},  index=['2006', '2007', '2008', '2009', '2013'] )

df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Mixed forest']].plot.bar(stacked=True, width=0.05, position=5.5, colormap="brg", ax=ax, alpha=0.7)





df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[0, 0, 1.876, 0, 0], 'Coniferous forest': [0, 0, 0.28, 0, 0], 'Mixed forest': [0, 0, 0.28, 0, 0], 'Water courses, Water bodies':[1, 1, 1, 1, 1]},  index=['2006', '2007', '2008', '2009', '2013'] )

df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Mixed forest']].plot.bar(stacked=True, width=0.05, position=5.5, colormap="brg", ax=ax, alpha=0.7)
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Mixed forest']].plot.bar(stacked=True, width=0.05, position=6.5, colormap="brg", ax=ax, alpha=0.7)
df[['Coastal Lagoons, Estuaries, Sea and ocean', 'Mixed forest']].plot.bar(stacked=True, width=0.05, position=7.5, colormap="brg", ax=ax, alpha=0.7)

#########################################################################################

df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[0, 0, 0, 2.1, 2.0272], 'Coniferous forest': [0.28, 0.28, 0.28, 1, 1], 'Mixed forest': [0.28, 0.28, 0.28, 1, 1], 'Water courses, Water bodies':[1, 1, 1, 1, 1]},  index=['2006', '2007', '2008', '2009', '2013'] )
df[['Coastal Lagoons, Estuaries, Sea and ocean']].plot.bar(stacked=True, width=0.05, position=3.5, colormap="brg", ax=ax, alpha=0.7)

df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[0, 0, 1.4, 0, 0], 'Coniferous forest': [0, 0, 0.28, 0, 0], 'Mixed forest': [0, 0, 0.28, 0, 0], 'Water courses, Water bodies':[1, 1, 1, 1, 1]},  index=['2006', '2007', '2008', '2009', '2013'] )
df[['Coastal Lagoons, Estuaries, Sea and ocean']].plot.bar(stacked=True, width=0.05, position=8.5, colormap="brg", ax=ax, alpha=0.7)



df = pd.DataFrame({'Coastal Lagoons, Estuaries, Sea and ocean':[2, 0, 0, 0, 0], 'Coniferous forest': [1, 0, 0, 0, 0], 'Mixed forest': [1, 0, 0, 0, 0], 'Water courses, Water bodies':[1, 1, 1, 1, 1]},  index=['2006', '2007', '2008', '2009', '2013'] )
df[['Coastal Lagoons, Estuaries, Sea and ocean']].plot.bar(stacked=True, width=0.05, position=6.5, colormap="brg", ax=ax, alpha=0.7)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
legend_properties = {'weight':'bold','size':18}
plt.legend(handles=[blue_patch, red_patch, green_patch, magenta_patch])
plt.ylim([0, 2.99])
#plt.yticks([], [])
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(2.8))
plt.title('Fall Migration',fontsize=18, fontweight="bold")
#ax.text(0.4, 0.9, 'Fall Migration',transform=ax.transAxes, ha="left",fontsize = 24,fontweight="bold")
plt.xlabel('Year',fontsize=18, fontweight="bold")
#plt.ylabel('Percentage composition of each habitat',fontsize=18, fontweight="bold")
plt.ylabel('\% Composition of each habitat',fontsize=18, fontweight="bold")
ax.xaxis.set_tick_params(labelsize=18)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=-45, ha = 'center')
ax.legend().remove()

plt.tick_params(labelsize = 20)
for tick in ax.xaxis.get_majorticklabels():
    tick.set_horizontalalignment("right")



plt.savefig('fall.eps',dpi = 140)
