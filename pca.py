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

indir = 'XXX'

outdir = 'XXX'
if not os.path.exists(outdir):
    os.makedirs(outdir)

dateList = [
	(datetime.date(2008, 8, 18), datetime.date(2008, 8, 19)),
	(datetime.date(2008, 8, 19), datetime.date(2008, 8, 20)),
	(datetime.date(2008, 8, 20), datetime.date(2008, 8, 21)),
]
dateUS4MHS = datetime.date(2008, 8, 19)
prevDateUS4MHS = datetime.date(2008, 8, 18)
dateUS4MHSStr = dateUS4MHS.strftime('%Y%m%d')
prevDateUS4MHSStr = prevDateUS4MHS.strftime('%Y%m%d')

adjRsq = {}
tstats = {}
for prevDate, date in dateList:
    print prevDate, date

    dateStr = date.strftime('%Y%m%d')
    prevDateStr = prevDate.strftime('%Y%m%d')

    if dateStr == dateUS4MHSStr:
        # ####### US4 Excess Returns Test (make sure we can replicate code)

        # read in excess returns
        fname = os.path.join(indir, 'excessReturns-fullhistory-estu-%s.csv' % dateStr)
        rets = pandas.read_csv(fname, index_col=0)
        print rets.isnull().sum().sum() # confirm there are no missing values
        print rets.shape

	# compare with excess returns from US4MH-S code
	fname = os.path.join(indir, 'fromUS4MH-Scode', 'returnsFullHistory-%s.csv' % dateUS4MHSStr)
	rets2 = pandas.read_csv(fname, index_col=0)
	assert set(rets.index).issubset(set(rets2.index))
	fname = os.path.join(indir, 'fromUS4MH-Scode', 'returnsFullHistory-Clipped-%s.csv' % dateUS4MHSStr)
	rets2clipped = pandas.read_csv(fname, index_col=0)
	assetsNE = rets.index[(rets2.reindex(index=rets.index) <> rets).sum(axis=1)>0]
	print 'Number of assets that differ', len(assetsNE)/float(rets.shape[0])
	fname = os.path.join(indir, 'fromUS4MH-Scode', 'estu-%s.csv' % prevDateUS4MHSStr)
	estu = list(pandas.read_csv(fname, names=['Estu'])['Estu'].values)
	newestu = list(set(estu).intersection(set(rets.index)))

	# eigen decomposition
	dates = rets.columns[-250:]
	df = rets2clipped.reindex(index=newestu, columns=dates).fillna(0.0)
	tr = numpy.dot(numpy.transpose(df.values), df.values)
	(d, v) = linalg.eigh(tr)
	d = d / df.shape[0]
	v = numpy.transpose(v)
	order = numpy.argsort(-d)
	d = numpy.take(d, order, axis=0)
	print 'Eigen decomp, US4MH_S, Clipped Excess Returns:', list(d/d.sum())[0:5]

	# svd
	(d, v) = linalg.svd(df.values, full_matrices=False)[1:]
	d = d**2 / df.shape[0]
	order = numpy.argsort(-d)
	d = numpy.take(d, order, axis=0)
	print 'svd decomp, US4MH_S, Clipped Excess Returns:', list(d/d.sum())[0:5]

	#v = numpy.take(v, order, axis=0)[0:self.numFactors,:]
	#factorReturns = v

	# plot results
	plotdf = pandas.DataFrame({'Proportion of Variance Explained': d/d.sum(),
	    'Cumulative Proportion of Variance Explained': numpy.cumsum(d/d.sum())})
	fig = plotdf.iplot(asFigure=True)
	plotly.offline.plot(fig, filename=os.path.join(outdir, 'ScreePlot-US4MH_Scode-ClippedExcessReturns-US4MH_S-%s.html' % dateUS4MHSStr),
		auto_open=False)


	# ####### Compare Excess Returns from US4 and hybrid model code 

	# compare with unclipped rets
	df = rets2.reindex(index=newestu, columns=dates).fillna(0.0)
	(d, v) = linalg.svd(df.values, full_matrices=False)[1:]
	d = d**2 / df.shape[0]
	order = numpy.argsort(-d)
	d = numpy.take(d, order, axis=0)
	print 'svd decomp, US4MH_S, Unclipped Excess Returns:', list(d/d.sum())[0:5]

	df = rets.reindex(index=newestu, columns=dates).fillna(0.0)
	(d, v) = linalg.svd(df.values, full_matrices=False)[1:]
	d = d**2 / df.shape[0]
	order = numpy.argsort(-d)
	d = numpy.take(d, order, axis=0)
	print 'svd decomp, Hybrid code, Unclipped Excess Returns:', list(d/d.sum())[0:5]
	# comment: the differences between the returns from the hybrid
	# model code and the returns from the US4 code are negligble 

	# plot results
	plotdf = pandas.DataFrame({'Proportion of Variance Explained': d/d.sum(),
	    'Cumulative Proportion of Variance Explained': numpy.cumsum(d/d.sum())})
	fig = plotdf.iplot(asFigure=True)
	plotly.offline.plot(fig, filename=os.path.join(outdir, 'ScreePlot-HybridCode-UnclippedExcessReturns-%s.html' % dateStr),
		auto_open=False)

	# if we run the analysis over all 1000 dates, here's what we get
	df = rets.fillna(0.0)
	(d, v) = linalg.svd(df.values, full_matrices=False)[1:]
	d = d**2 / df.shape[0]
	order = numpy.argsort(-d)
	d = numpy.take(d, order, axis=0)
	print 'svd decomp, Hybrid code, unclipped excess returns, 1000 days:', list(d/d.sum())[0:5]


    # ####### US4 Specific Returns 

    # read in specific returns
    fname = os.path.join(indir, 'specificReturns-estu-%s.csv' % dateStr)
    rets = pandas.read_csv(fname, index_col=0)
    dates = rets.columns[-250:]

    # optionally remove assets that are missing more than X returns
    propMissing = df.isnull().sum(axis=1)/df.shape[0]
    newEstu = [a for a in df.index if propMissing[a] <= .9]
    assert (rets.shape[0] - len(newestu))/float(rets.shape[0]) < .5

    # TO DO: optionally trim specific returns (let's think about this)

    # estimate statistical factor returns
    df = rets.reindex(index=newEstu, columns=dates).fillna(0.0)
    (d, v) = linalg.svd(df.values, full_matrices=False)[1:]
    d = d**2 / df.shape[0]
    order = numpy.argsort(-d)
    d = numpy.take(d, order, axis=0)
    print 'svd decomp, hybrid code, Specific Returns', list(d/d.sum())[0:5]

    # plot results
    plotdf = pandas.DataFrame({'Proportion of Variance Explained': d/d.sum(),
	'Cumulative Proportion of Variance Explained': numpy.cumsum(d/d.sum())})
    fig = plotdf.iplot(asFigure=True)
    plotly.offline.plot(fig, filename=os.path.join(outdir, 'ScreePlot-UnclippedSpecificReturns-%s.html' % dateStr),
	    auto_open=False)

    # weighted PCA 
    fname = os.path.join(indir, 'weights-estu-%s.csv' % prevDateStr)
    wgts = pandas.read_csv(fname, names=['w'])['w']
    idx = wgts.index & df.index
    r = df.reindex(index=idx)
    w = wgts.reindex(index=idx)
    scaledRets = r.multiply(w, axis=0)
    (d, v) = linalg.svd(scaledRets.values, full_matrices=False)[1:]
    d = d**2 / scaledRets.shape[0]
    order = numpy.argsort(-d)
    d = numpy.take(d, order, axis=0)
    print 'svd decomp:', list(d/d.sum())[0:5]
    v = numpy.take(v, order, axis=0)[0:10,:]
    # take factor returns to be right singular vectors
    factorReturns = v
    # back out expousres
    statExp = numpy.dot(ma.filled(df.values, 0.0), numpy.transpose(v))
    statExpdf = pandas.DataFrame(statExp, index=df.index, 
	    columns= ['StatFactor-%.0f'%i for i in range(10)])


    # plot results
    plotdf = pandas.DataFrame({'Proportion of Variance Explained': d/d.sum(),
	'Cumulative Proportion of Variance Explained': numpy.cumsum(d/d.sum())})
    fig = plotdf.iplot(asFigure=True)
    plotly.offline.plot(fig, filename=os.path.join(outdir, 'ScreePlot-UnclippedSpecificReturns-WeightedPCA-%s.html' % dateStr),
	    auto_open=False)

    # ###### Run cross-sectional regression with stat factors
    fname = os.path.join(indir, 'excessReturns-fullhistory-estu-%s.csv' % dateStr)
    rets = pandas.read_csv(fname, index_col=0)
    fname = os.path.join(indir, 'fundExposures-estu-%s.csv' % prevDateStr)
    fundExp = pandas.read_csv(fname, index_col=0).fillna(0.0)
    fname = os.path.join(indir, 'weights-estu-%s.csv' % prevDateStr)
    wgts = pandas.read_csv(fname, names=['w'])['w']
    # run regression without market intercept
    fundCols = [c for c in fundExp if c <> 'Market Intercept']
    rets.columns = pandas.to_datetime(rets.columns)
    prets = rets[date]
    idx = prets.index & fundExp.index & wgts.index & set(newEstu)
    y = prets.reindex(index=idx)
    X = fundExp[fundCols].reindex(index=idx)
    w = wgts.reindex(index=idx)
    statX = statExpdf.reindex(index=idx)

    # fund results
    regresult = sm.WLS(y, X, weights=w).fit()

    # hybrid results
    newX = pandas.concat([X, statX[statX.columns[:3]]], axis=1)
    regresult2 = sm.WLS(y, newX, weights=w).fit()
    print 'Rsq:', regresult.rsquared_adj, regresult2.rsquared_adj
    print regresult2.tvalues.loc[statExpdf.columns[:3]]

    adjRsq[date] = {'Fund AdjR2': regresult.rsquared_adj, 
                    'Hybrid AdjR2': regresult2.rsquared_adj}
    tstats[date] = regresult2.tvalues.loc[statExpdf.columns[:3]]

#pandas.DataFrame.from_dict(adjRsq)
tmpdf = pandas.DataFrame.from_dict(adjRsq, orient='index')
tmpdf.to_csv(os.path.join(outdir, 'AdjR2.csv'))
tmpdf = pandas.DataFrame.from_dict(tstats, orient='index')
tmpdf.to_csv(os.path.join(outdir, 'Tstats.csv'))






