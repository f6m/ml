import numpy as np
import pandas as pd #Pandas or csv library reads a csv file
import numpy as np
import scipy.stats
#import pingouin as pg #para el test anova no lo carga google
from scipy.stats import f_oneway
from scipy.stats import kruskal

df = pd.read_csv('varsind.csv')
produ = df['oilproduction'][0:27]
pibmx = df['pibmx'][0:27]
wti = df['wti'][0:27]

#MATRIX DE CORRELACION
matrix = np.corrcoef([produ, pibmx, wti])
print(matrix)
#Las entradas de la matriz de correlación el coef. de Pearson.
print(scipy.stats.pearsonr(produ, pibmx))   # Pearson's r and correlation
print(scipy.stats.spearmanr(produ, pibmx)) # Spearman's rho
print(scipy.stats.kendalltau(produ, pibmx))  #  Kendall's tau

#Las entradas de la matriz de correlación el coef. de Pearson.
print(scipy.stats.pearsonr(produ, pibmx)[0])   # Pearson's r
print(scipy.stats.spearmanr(produ, pibmx)[0]) # Spearman's rho
print(scipy.stats.kendalltau(produ, pibmx)[0])  #  Kendall's tau

#TEST ANOVA ANALISYS OF VARIANCE para variables NORMALES
f_oneway(produ,pibmx,wti)

#Por lo tanto como p-value = 6.850244172472871e-58 < 0.05 se rechaza H0
#y al menos existe una media que difiere del resto

hres = kruskal(produ,pibmx,wti)
print(hres)

#Por lo tanto como p-value = 3.777639908118289e-16 < 0.05 se rechaza H0
#y al menos existe una mediana de entre las tres que difiere del resto
