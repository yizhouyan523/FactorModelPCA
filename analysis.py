import pandas 
import os
import numpy
#from sklearn.decomposition import PCA
import plotly
import cufflinks
import numpy.linalg as linalg
import datetime
import statsmodels.api as sm
import numpy.ma as ma
import pymongo
from pymongo import MongoClient

#import matplotlib.pyplot as plt

# create connection
client = MongoClient(host='Prefix')

# create database object
db = client['YizhouDatabase']

# create a collection (table)
coll = db['HybridModel']

#outdir='Output'

outdir = 'XXX'

class plots_pointintime:
    
	
    def _init_(self, stat_m, datestr):
        self.stat_m=stat_m
        self.datestr=datestr
   
    def plot_pcastat(self,name):
        a=[]
        x=[]
        cursor=db.coll.find()
        for document in cursor:
            if (self.datestr==document['dt'].strftime('%Y%m%d')):
                for i in range(self.stat_m):
                    a.append(document[name]['StatFactor-%d'%i])
                    x.append('statfactor-%d'%i)
		#print a
        if (name=='propVarExplained'):
            plotdf = pandas.DataFrame({name: a,
                                   'Cumulative%s'%name: numpy.cumsum(a)},index=x)
        else:
            plotdf = pandas.DataFrame({name:a},index=x)
        fig = plotdf.iplot(asFigure=True)
        plotly.offline.plot(fig, filename=os.path.join(outdir, 'Pointintime-%s' %name + '-%s.html' % self.datestr),
                        auto_open=False)
	
    def plot_tstat(self):
        cursor = db.coll.find()
        for document in cursor:
            if (self.datestr==document['dt'].strftime('%Y%m%d')):
                fundf=document['stats']['fundTstats'].keys()
		fundt=map(abs,document['stats']['fundTstats'].values())
                hybridf=document['stats']['hybridTstats'].keys()
		hybridt=map(abs,document['stats']['hybridTstats'].values())
        fundamental = plotly.graph_objs.Bar(x=fundf,y=fundt,name='Fundamental')
        hybrid = plotly.graph_objs.Bar(x=hybridf,y=hybridt, name='Hybrid')
        data=[fundamental,hybrid]
        layout=plotly.graph_objs.Layout(barmode='group')
        fig=plotly.graph_objs.Figure(data,layout)
        plotly.offline.plot(fig, filename=os.path.join(outdir, 'Pointintime-tstatcomparison-%s.html' % self.datestr),
                        auto_open=False)
	
    def plot_r2(self):
        cursor = db.coll.find()
        for document in cursor:
            if (self.datestr==document['dt'].strftime('%Y%m%d')):
                adjr2fund=document['adjR2']['adjR2Fund']
                adjr2hybrid=document['adjR2']['adjR2Hybrid']
        data = [plotly.graph_objs.Bar(x=['adjR2Fund','adjR2Hybrid'],
                                     y=[adjr2fund,adjr2hybrid])]
        fig = plotly.graph_objs.Figure(data)
        #plotdf = pandas.DataFrame([[adjr2fund,adjr2hybrid]],index=['adjR2Fund','adjR2Hybrid'])
        #fig = plotdf.iplot(asFigure=True)
        plotly.offline.plot(fig, filename=os.path.join(outdir, 'Pointintime-R2comparison-%s.html' % self.datestr),
                        auto_open=False)  
        

class plots_overtime:
    def _init_(self, stat_m, datestr, radays):
        self.cur=cursor
        self.stat_m=stat_m
        self.datestr=datestr
        self.radays=radays
   
    def rolling_avg(self,lists):
        ra=[]
        for i in range(len(lists)):
            if(i==0):
                ra.append(lists[i])
            elif(i<self.radays):
                ra.append(numpy.mean(lists[:i]))
            else:
                ra.append(numpy.mean(lists[i-self.radays:i]))
        return ra
    
    def rolling_std(self,lists):
        rstd=[]
        for i in range(len(lists)):
            if(i==0):
                rstd.append(0)
            elif(i<self.radays):
                rstd.append(numpy.std(lists[:i]))
            else:
                rstd.append(numpy.std(lists[i-self.radays:i]))
        return rstd

		
#cursor = db.coll.find({})
p1=plots_pointintime()
p1.stat_m=3
p1.datestr='20080818'
p1.plot_pcastat('eigenValues')
p1.plot_pcastat('propVarExplained')
p1.plot_pcastat('StatFactorReturns')
p1.plot_tstat()
p1.plot_r2()
'''
cursor = db.coll.find()
for d in cursor:
    print d
#from IPython import embed; embed()
'''
