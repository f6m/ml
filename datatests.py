import pandas as pd #Pandas or csv library reads a csv file
import matplotlib.pyplot as plt #plot
import numpy as np
import pylab
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import norm
from scipy.stats import probplot
from scipy.stats import mode
from seaborn import *

#Cargamos el archivo de variables independientes
df = pd.read_csv('varsind.csv')
price = df['oilproduction']
price28 = df['oilproduction'][0:27]
print(price28)
#print(df.head(10)) imprime 10 lineas
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

#Creamos un HISTORGRAMA
axes[0].hist(df['oilproduction'], edgecolor='black', bins=30)
#axes[1].hist(df['pibmx'],edgecolor='red', bins=30)

#Creamos un QQPLOT - plot QQ quartile-quartile
probplot(df['oilproduction'], dist="norm", plot=pylab)
pylab.show()

#BOX PLOT
#df.boxplot(by = 'anio1', column = ['oilproduction'], grid = True) 
#boxplot(x = 'pibmx', y = 'oilproduction', data = df)
import plotly.express as px
fig = px.box(df, y="oilproduction")
fig.show()

#Realizamos la prueba estadística de Shapiro-Wilk para normalidad (test)
alpha = 0.05 #Nivel de significancia para la prueba
if (len(price28)) < 70:
    print("Se aplicará un test de normalidad Shapiro-Wilk")
    stat,p=shapiro(price28)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    if p > alpha:
      print('De acuerdo con el test de SW la muestra parece Normal (Gaussiana) (se fallo a rechazar H0)')
    else:
      print('De acuerdo con el test de SW la muestra NO parece Normal (Gaussiana) (se rechaza H0)')
else:
  print("Se aplicará un test de normalidad Kolmogorov-Smirnov")
  pvalu = kstest(price28,'norm').pvalue
  if (pvalu > alpha):
    print('De acuerdo con el test de KS la muestra parece Normal (Gaussiana) (se fallo a rechazar H0)')
  else:
    print('De acuerdo con el test de KS la muestra NO parece Normal (Gaussiana) (se rechaza H0)')

print('Estadisticos descriptivos')
print(np.mean(price28))
print(np.var(price28))
print(mode(price28)) 
print(np.median(price28))
print(np.percentile(price28, [25, 50, 75]))

#var(price28), mod(price28), median(price28))
#Puesto que ShapiroResult(statistic=0.959795566172879, pvalue=0.3875267716578847)
#Puesto que 0.38 > 0.05 se hacepta H0 y los datos se consideran distribuidos
#de forma normal

#Realizamos la prueba estadística de Kolmogorov-Smirnov para normalidad (test)
#Puesto que ShapiroResult(statistic=0.959795566172879, pvalue=0.3875267716578847)
#Puesto que 0.38 > 0.05 se hacepta H0 y los datos se consideran distribuidos
#De forma normal
