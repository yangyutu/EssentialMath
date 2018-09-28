%% FFwebinar - Using MATLAB to Develop Asset Pricing Models
%
%	Robert Taylor, The MathWorks, Inc., 16 November 2006.
%
%	Copyright (C) 2006 by The MathWorks, Inc.
%
%	These scripts require MATLAB, Financial Toolbox, and Statistics Toolbox with
%	version 2006b or higher. See the file readme.txt for instructions.
%
%	Requirements:
%		MATLAB
%		Financial Toolbox
%		Statistics Toolbox (comes bundled with Financial Toolbox)
%
%% Step 1 - Get data for analysis
%	Load in the raw data for the example which is in the form of time series
%	with total return prices. Total return prices are the accumulated value of
%	linked total returns for a given series from an initial value of 1 (since we
%	will be working with returns in the example, it does not matter what the
%	starting price is).
%	
%	The file FFUniverse.mat has the following items -
%	CAPMFactorList - A cell array that contains two codes to identify the CAPM
%		factors. The codes are '_CASH' to identify a cash series and '_XSMRKT'
%		to identify an excess market return series.
%	FactorList - A cell array that contains four codes to identify the Fama &
%		French factors. The codes are '_CASH' to identify a cash series, '_HML'
%		to identify the Fama & French High-minus-Low factor, '_SMB' to identify
%		the Fama & French Small-minus-Big factor, and '_XSMRKT' to identify an
%		excess market return series.
%	Universe - A financial timeseries (fints) object that contains daily total
%		return prices for 14 technology stocks from January 1962 to October
%		2006. Each stock is identified in the fints object by its ticker symbol
%		of October 2006.
%	cash - A matrix with linked returns for a cash instrument with MATLAB serial
%		date numbers in the first column and daily total return prices in the
%		second column.
%	hml - A matrix with linked Fama & French HML factor returns with MATLAB
%		serial date numbers in the first column and daily total return prices in
%		the second column.
%	mrkt - A matrix with linked excess market returns (market minus cash) with
%		MATLAB serial date numbers in the first column and daily total return
%		prices in the second column.
%	smb - A matrix with linked Fama & French HML factor returns with MATLAB
%		serial date numbers in the first column and daily total return prices in
%		the second column.
%	The cash, hml, mrkt, and smb series are courtesy of Kenneth French,
%	Dartmouth University, 2006.

clear all
clc

load FFUniverse

%	Display variables from the mat file and display information about the fints
%	object Universe.

whos
ftsinfo(Universe);

%% Step 2 - Reality check
%	This step shows how to work with the fints object.

clear all

load FFUniverse

%	Convert daily data to monthly data

mUniverse = tomonthly(Universe);

%	Take the log of both daily and monthly total return prices

logUniverse = log(Universe);
logmUniverse = log(mUniverse);

%	Plot all time series in both fints objects with daily data on the upper plot
%	and monthly data on the lower plot. This is always a good step to check your
%	data.

figure(gcf);
subplot(2,1,1);
	plot(logUniverse);
	xlabel('\bfDate');
	ylabel('\bfPrice');
	title('\bfLog Stock Total Return Prices (Daily Data)');
subplot(2,1,2);
	plot(logmUniverse);
	xlabel('\bfDate');
	ylabel('\bfPrice');
	title('\bfLog Stock Total Return Prices (Monthly Data)');

%% Step 3 - Add CAPM factors to universe
%	This step shows how to combine times series and to let MATLAB handle the
%	date math. Specifically, this step sets up the data for the CAPM and saves
%	it in the file FFUniverseCAPM.mat which will be used later. Note that the
%	series names for the added timeseries are obtained from the cell-array
%	CAPMFactorList.

clear all

load FFUniverse

%	Merge the cash and excess return series into a copy of the fints object that
%	already contains stock data.

CAPMUniverse = Universe;
CAPMUniverse = merge(CAPMUniverse, fints(cash(:,1),cash(:,2),CAPMFactorList{1}));
CAPMUniverse = merge(CAPMUniverse, fints(mrkt(:,1),mrkt(:,2),CAPMFactorList{2}));

CAPMUniverse.desc = 'Universe with Factors for CAPM';
CAPMUniverse.freq = 'daily';

ftsinfo(CAPMUniverse);

save FFUniverseCAPM CAPMUniverse CAPMFactorList

%% Step 4 - Estimate CAPM
%	This step runs a script FFestimateCAPM.m which creates a file
%	CAPMResults.mat that contains estimates based on the CAPM.
%
%	If you open the file, you will see that this script computes total returns
%	with the fints object, sets up the SUR regression for the CAPM, and performs
%	a series of regressions with a 5-year estimation (Window = 5) that slides
%	along by 1-year intervals ending at each September month-end (TMonth = 9).
%	The stocks to be included in the estimation for each period must have at
%	least about 1 year worth of data or no more than 4 years of NaN values
%	(MaxNaNs = 4*260). The main loop to do the regressions sets up the SUR
%	regression model and uses multivariate normal regression functions in the
%	Financial Toolbox to estimate parameters, standard errors, and the
%	log-likelihood function.

FFestimateCAPM

%% Step 5 - Examine CAPM estimates
%	Given the results from the estimation, plot the estimates for both Alpha and
%	Beta in the CAPM for the 14 stocks of our example. For each stock, the
%	series of bars represents the value of the parameter for each historical
%	period of the estimation.

clear all

load CAPMResults

figure(gcf);
subplot(2,1,1);
	bar(CAPMAlpha(1:7,:)*sqrt(252));
	set(gca,'XTickLabel',AssetList(1:7));
	set(gca,'XTick',1:7);
	set(gca,'XLim',[0,8]);
	title('\bfEstimated Alphas Based on CAPM ');
	ylabel('\bfAlpha (year)');
    set(gca,'linewidth',2,'fontsize',12,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.02;0.01]);

subplot(2,1,2);
	bar(CAPMBeta(1:7,:));
	set(gca,'XTickLabel',AssetList(1:7));
	set(gca,'XTick',1:7);
	set(gca,'XLim',[0,8]);
	title('\bfEstimated Betas Based on CAPM');
	ylabel('\bfBeta');
set(gca,'linewidth',2,'fontsize',14,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.02;0.01]);
%pbaspect([1 1 1])
%set(gca,'box','on')
%% Step 6 - Add FF factors to universe
%	This step is like Step 3 but for the Fama & French three-factor model. It
%	sets up the data for the model and saves it in the file FFUniverseFF.mat
%	which will be used later. Note that the series names for the added
%	timeseries are obtained from the cell-array FactorList.

clear all

load FFUniverse

Universe = merge(Universe, fints(cash(:,1),cash(:,2),FactorList{1}));
Universe = merge(Universe, fints(hml(:,1),hml(:,2),FactorList{2}));
Universe = merge(Universe, fints(smb(:,1),smb(:,2),FactorList{3}));
Universe = merge(Universe, fints(mrkt(:,1),mrkt(:,2),FactorList{4}));

Universe.desc = 'Universe with Factors for Fama & French Model';
Universe.freq = 'daily';

ftsinfo(Universe);

save FFUniverseFF Universe FactorList

%% Step 7 - Estimate Fama & French three-factor model
%	This script creates a file FFResults.mat that contains estimates based on
%	the Fama & French model. In addition, the script creates a file XResults.mat
%	that computes the Fama & French model with an Alpha = 0 restriction.
%
%	The script parallels the operations of the script FFestimateCAPM but with
%	the additional SMB and HML factors of the Fama & French model. The
%	difference between these two scripts shows how you can add additional
%	factors to a multi-factor model.

FFestimateFF

%% Step 8 - Examine FF model estimates
%	Given the results from the estimation, plot the estimates for Alpha, Beta,
%	HML, and SMB in the Fama & French three-factor model for the 14 stocks of
%	our example. For each stock, the series of bars represents the value of the
%	parameter for each historical period of the estimation.

clear all

load FFResults

figure(gcf);
subplot(4,1,1);
	bar(Alpha(1:7,:)*sqrt(252));
	set(gca,'XTickLabel',AssetList(1:7));
	set(gca,'XTick',1:7);
	set(gca,'XLim',[0,8]);
	title('\bfEstimated Factor Exposures Based on the Fama & French Three-Factor Model');
	ylabel('\bfAlpha (year)');
set(gca,'linewidth',2,'fontsize',14,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.02;0.01]);
    
subplot(4,1,2);
	bar(Beta);
	set(gca,'XTickLabel',AssetList(1:7));
	set(gca,'XTick',1:7);
	set(gca,'XLim',[0,8]);
	ylabel('\bfBeta');
set(gca,'linewidth',2,'fontsize',14,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.02;0.01]);

    subplot(4,1,3);
	bar(HML);
	set(gca,'XTickLabel',AssetList(1:7));
	set(gca,'XTick',1:7);
	set(gca,'XLim',[0,8]);
	ylabel('\bfHML');
set(gca,'linewidth',2,'fontsize',14,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.02;0.01]);
    
subplot(4,1,4);
	bar(SMB);
	set(gca,'XTickLabel',AssetList(1:7));
	set(gca,'XTick',1:7);
	set(gca,'XLim',[0,8]);
	ylabel('\bfSMB');
set(gca,'linewidth',2,'fontsize',14,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.02;0.01]);

%% Step 9 - Compare Alpha estimates for CAPM and FF model
%	Visually compare Alpha estimates between the CAPM and the Fama & French
%	models.

load FFResults
load CAPMResults

figure(gcf);
subplot(2,1,1);
	bar(Alpha);
	set(gca,'XTickLabel',AssetList);
	set(gca,'XTick',1:14);
	set(gca,'XLim',[0,15]);
	title('\bfEstimated Alphas Based on Fama & French');
	ylabel('\bfAlpha');

subplot(2,1,2);
	bar(CAPMAlpha);
	set(gca,'XTickLabel',AssetList);
	set(gca,'XTick',1:14);
	set(gca,'XLim',[0,15]);
	title('\bfEstimated Alphas Based on CAPM');
	ylabel('\bfAlpha');

%% Step 10 - Compare Beta estimates for CAPM and FF model
%	Repeat the preceding step with Beta estimates.

load FFResults
load CAPMResults

figure(gcf);
subplot(2,1,1);
	bar(Beta);
	set(gca,'XTickLabel',AssetList);
	set(gca,'XTick',1:14);
	set(gca,'XLim',[0,15]);
	title('\bfEstimated Betas Based on Fama & French');
	ylabel('\bfBeta');

subplot(2,1,2);
	bar(CAPMBeta);
	set(gca,'XTickLabel',AssetList);
	set(gca,'XTick',1:14);
	set(gca,'XLim',[0,15]);
	title('\bfEstimated Betas Based on CAPM');
	ylabel('\bfBeta');

%% Step 11 - Likelihood ratio test to see if FF model is significant
%	The first statistical test seeks to determine if the additional Fama &
%	French factors HML and SMB are statistically equal to zero (which would
%	imply that the CAPM is sufficient to explain asset returns within this
%	sample). This can be done with a likelihood ratio test. In general, you
%	would do this test to validate factors that you might want to add to a
%	multifactor model.

load FFResults
load CAPMResults

clc

NumPeriods = size(FFLLF,1);

StartYear = 1968;
EndYear = 2006;

%	The test statistic for the likelihood ratio test is LRatio which is the
%	difference in log-likelihood functions between the unrestricted and
%	restricted models (the restriction is to force SMB = HML = 0, which is
%	equivalent to the CAPM model).
%
%	The likelihood ratio is a chi-square random variable and the null hypothesis
%	is rejected if the test statistic is greater than the critical value derived
%	from the chi-square distribution.

LRatio = 2 * (FFLLF - CAPMLLF);
DoF = 2 * FFDoF;

CriticalValue = chi2inv(0.95, DoF);

%	Display a table of test statistics and critical values for each estimation
%	period.

fprintf('H0: Is HML = SMB = 0 in Fama & French Three-Factor Model?\n');
fprintf('H1: At least one non-zero HML or SMB\n');
fprintf('  %4s  %9s  %9s  %9s\n','Year','Decision','  Test','Critical');
fprintf('  %4s  %9s  %9s  %9s\n',' ','(5%% LOS)','Statistic','  Value');
fprintf('  %4s  %9s  %9s  %9s\n','----','---------','---------','---------');
for i = 1:NumPeriods
	fprintf('  %4d  ',StartYear + i - 1);
	if LRatio(i) > CriticalValue(i)
		fprintf('%9s','Reject');
	else
		fprintf('%9s','Accept');
	end
	fprintf('  %9g  %9g\n',LRatio(i),CriticalValue(i));
end

U = CriticalValue - LRatio;
V = U .* (U > 0);
W = U .* (U <= 0);
Y = [ V W ];

%	Plot acceptance or rejection of the null hypothesis, where a positive
%	difference between the critical value and the likelihood ratio is acceptance
%	of the null hypothesis and a negative difference is rejection.

figure(gcf);
clf
bar(StartYear:EndYear,Y,1.5);
set(gca,'YLim',[-600,100]);
title('\bfLikelihood Ratio Test that HML = SMB = 0 in Fama & French Model (5% LOS)');
text(1970,50,'\bfAccept');
text(1970,-400,'\bfReject');
xlabel('\bfYear');
ylabel('\bfDifference between Critical Value and Test Statistic');

%% Step 12 - Likelihood ratio test to see if Alpha = 0
%	The second statistical test examines the null hypothesis that the Alphas
%	from the Fama & French model are zero. It is also a likelihood ratio test
%	that can be refined by estimation of the models with separate restrictions
%	on the Alpha estimates (you would have to modify FFestimateFF.m to set up
%	different hypothesis tests).

load FFResults
load XResults

clc

NumPeriods = size(FFLLF,1);

StartYear = 1968;
EndYear = 2006;

LRatio = 2 * (FFLLF - FFXLLF);
DoF = FFDoF;

CriticalValue = chi2inv(0.95, FFDoF);

fprintf('H0: Is Alpha = 0 in Fama & French Three-Factor Model?\n');
fprintf('H1: At least one non-zero Alpha\n');
fprintf('  %4s  %9s  %9s  %9s\n','Year','Decision','  Test','Critical');
fprintf('  %4s  %9s  %9s  %9s\n',' ','(5%% LOS)','Statistic','  Value');
fprintf('  %4s  %9s  %9s  %9s\n','----','---------','---------','---------');
for i = 1:NumPeriods
	fprintf('  %4d  ',StartYear + i - 1);
	if LRatio(i) > CriticalValue(i)
		fprintf('%9s','Reject');
	else
		fprintf('%9s','Accept');
	end
	fprintf('  %9g  %9g\n',LRatio(i),CriticalValue(i));
end

U = CriticalValue - LRatio;
V = U .* (U > 0);
W = U .* (U <= 0);
Y = [ V W ];

figure(gcf);
clf
bar(StartYear:EndYear,Y,1.5);
title('\bfLikelihood Ratio Test that Alpha = 0 in Fama & French Model (5% LOS)');
text(StartYear,5,'\bfAccept');
text(StartYear,-15,'\bfReject');
xlabel('\bfYear');
ylabel('\bfDifference between Critical Value and Test Statistic');

%% Step 13 - Examine alpha "decay"
%	Perfom visual and statistical tests on each Alpha estimate from the Fama &
%	French model for each individual stock.
%
%	The first part of this test is a display of the Fama & French model
%	estimates for Alpha, Beta, HML, and SMB factor exposures for 2005 and 2006.
%	The t-statistics are also displayed.
%
%	The second part of this test is to plot the Alpha estimates from IPO dates
%	going forward to examine the pattern of decay of Alpha estimates from 1 year
%	after an IPO onward.

load FFResults

[NumAssets, NumPeriods] = size(Alpha);

clc

fprintf('Fama & French Parameter Estimates for 2005\n');

fprintf('  %6s  %7s%8s  %7s%8s  %7s%8s  %7s%8s\n', ...
	'Asset','Alpha','(t)','Beta','(t)','HML','(t)','SMB','(t)');
fprintf('  %6s  %15s  %15s  %15s  %15s\n','------', ...
	'---------------','---------------','---------------','---------------');
for i = 1:NumAssets
	fprintf('  %6s  %6.3f (%6.3f)  %6.3f (%6.3f)  %6.3f (%6.3f)  %6.3f (%6.3f)\n', ...
		AssetList{i},260*Alpha(i,end-1),abs(Alpha(i,end-1)/StdAlpha(i,end-1)), ...
		Beta(i,end-1),abs(Beta(i,end-1)/StdBeta(i,end-1)), ...
		HML(i,end-1),abs(HML(i,end-1)/StdHML(i,end-1)), ...
		SMB(i,end-1),abs(SMB(i,end-1)/StdSMB(i,end-1)));
end

fprintf('\n');
fprintf('Fama & French Parameter Estimates for 2006\n');

fprintf('  %6s  %7s%8s  %7s%8s  %7s%8s  %7s%8s\n', ...
	'Asset','Alpha','(t)','Beta','(t)','HML','(t)','SMB','(t)');
fprintf('  %6s  %15s  %15s  %15s  %15s\n','------', ...
	'---------------','---------------','---------------','---------------');
for i = 1:NumAssets
	fprintf('  %6s  %6.3f (%6.3f)  %6.3f (%6.3f)  %6.3f (%6.3f)  %6.3f (%6.3f)\n', ...
		AssetList{i},260*Alpha(i,end),abs(Alpha(i,end)/StdAlpha(i,end)), ...
		Beta(i,end),abs(Beta(i,end)/StdBeta(i,end)), ...
		HML(i,end),abs(HML(i,end)/StdHML(i,end)), ...
		SMB(i,end),abs(SMB(i,end)/StdSMB(i,end)));
end

AlphaDecay = nan(size(Alpha));
for i = 1:NumAssets
	LastNan = find(isnan(Alpha(i,:)),1,'last');
	if ~isempty(LastNan)
		AlphaDecay(i,1:(NumPeriods - LastNan)) = Alpha(i,(1 + LastNan):NumPeriods);
	end
end
AlphaDecay = AlphaDecay';

figure(gcf);
plot(AlphaDecay(1:20,:));
title('\bfAlpha "Decay" after an IPO');
legend(AssetList);
xlabel('\bfYears after IPO');
ylabel('\bfAlpha (Daily)');


%% my part
clear all

load FFUniverse

Universe = merge(Universe,fints(cash(:,1),cash(:,2),'CASH'));
Universe = merge(Universe, fints(hml(:,1),hml(:,2),'HML'));
Universe = merge(Universe, fints(smb(:,1),smb(:,2),'SMB'));
Universe = merge(Universe, fints(mrkt(:,1),mrkt(:,2),'XSMRKT'));

FactorList = {'XSMRET','SMB','HML','CASH'};
% Step 2 - convert total return prices to total returns

Returns = (Universe - lagts(Universe,1)) ./ lagts(Universe,1);

% Step 3 - trim date range to match date range of factor returns

Returns = Returns(all(isfinite(fts2mat(extfield(Returns, FactorList))),2));


StartDate = datenum(2001,10,eomday(2001,10))+1;
EndDate = datenum(2006,10,eomday(2006,10));


	StartIndex = find(Returns.dates >= StartDate,1,'first');
	EndIndex = find(Returns.dates <= EndDate,1,'last');
    
   AP = fts2mat(Returns.AAPL);
   mk = fts2mat(Returns.XSMRKT);
   smb = fts2mat(Returns.SMB);
   hml = fts2mat(Returns.HML);
   
   h=figure(1)
   
   plot(AP(StartIndex:EndIndex),mk(StartIndex:EndIndex),'O');
   xlabel('AAPL return')
   ylabel('market excess return')
   set(gca,'linewidth',2,'fontsize',15,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.04;0.02]);
set(gca,'box','on')
pbaspect([1 1 1])
saveTightFigure(h,'AAPL_MKT_scatterplot.pdf')

   h=figure(2)
   plot(AP(StartIndex:EndIndex),smb(StartIndex:EndIndex),'O');
   xlabel('AAPL return')
   ylabel('SMB excess return')
   set(gca,'linewidth',2,'fontsize',15,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.04;0.02]);
set(gca,'box','on')
pbaspect([1 1 1])
saveTightFigure(h,'AAPL_SMB_scatterplot.pdf')

   h=figure(3)
  plot(AP(StartIndex:EndIndex),hml(StartIndex:EndIndex),'O');   
  xlabel('AAPL return')
   ylabel('HML excess return')
   set(gca,'linewidth',2,'fontsize',15,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.04;0.02]);
set(gca,'box','on')
pbaspect([1 1 1])
saveTightFigure(h,'AAPL_HML_scatterplot.pdf')
  
  design = [mk(StartIndex:EndIndex) smb(StartIndex:EndIndex) hml(StartIndex:EndIndex)];
     
  md1 = fitlm(design, AP(StartIndex:EndIndex));
  
  md2 = fitlm(mk(StartIndex:EndIndex), AP(StartIndex:EndIndex));
  
  ReturnsMat = fts2mat(Returns);
  techStock = mean(ReturnsMat(:,1:14),2);
  
  mdl3 = fitlm(design,techStock(StartIndex:EndIndex));