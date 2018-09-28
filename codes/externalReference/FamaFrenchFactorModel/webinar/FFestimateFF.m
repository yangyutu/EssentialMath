%FFestimateFF Estimate rolling parameters for Fama & French 3-factor model.
%
%	Fama & French factor returns courtesy of Professor Kenneth French, Dartmouth
%	University.

clear all

% Step 1 - load in raw data and set up list of Fama & French factors

load FFUniverseFF

FactorList = {'_XSMRKT'; '_SMB'; '_HML'; '_CASH' };

% Step 2 - convert total return prices to total returns

Returns = (Universe - lagts(Universe,1)) ./ lagts(Universe,1);

% Step 3 - trim date range to match date range of factor returns

Returns = Returns(all(isfinite(fts2mat(extfield(Returns, FactorList))),2));

% Step 4 - get information about data

SeriesList = fieldnames(Returns, 1);
AssetList = setdiff(SeriesList, FactorList);

NumSeries = numel(SeriesList);
NumAssets = numel(AssetList);

% Step 5 - primary controls

TMonth = 9;				% terminal month for each historical analysis period
Window = 5;				% historical analysis period in years
MaxNaNs = 4*260;		% max number of daily NaNs in an analysis period

% Step 6 - set up date math

StartYear = year(Returns.dates(1));
StartMonth = month(Returns.dates(1));
EndYear = year(Returns.dates(end));
EndMonth = month(Returns.dates(end));

if StartMonth > TMonth
	StartYear = StartYear + 1;
end
if EndMonth < TMonth
	EndYear = EndYear - 1;
end

if (EndYear - StartYear) < Window
	error('Insufficient data to perform analysis.');
end

NumPeriods = EndYear - StartYear - Window + 1;

% Step 7 - final setup

AnalysisPeriod = NaN(NumPeriods,1);

Alpha = NaN(NumAssets, NumPeriods);
Beta = NaN(NumAssets, NumPeriods);
SMB = NaN(NumAssets, NumPeriods);
HML = NaN(NumAssets, NumPeriods);

Sigma = NaN(NumAssets, NumPeriods);

StdAlpha = NaN(NumAssets, NumPeriods);
StdBeta = NaN(NumAssets, NumPeriods);
StdSMB = NaN(NumAssets, NumPeriods);
StdHML = NaN(NumAssets, NumPeriods);

FFLLF = NaN(NumPeriods,1);
FFDoF = NaN(NumPeriods,1);
FFXLLF = NaN(NumPeriods,1);
FFXDoF = NaN(NumPeriods,1);

% Step 8 - main loop

TYear = StartYear + Window;			% initial terminal year

for K = 1:NumPeriods
	
	% Step 8a - get start and end dates for current analysis period
	
	StartDate = datenum(TYear - Window,TMonth,eomday(TYear,TMonth)) + 1;
	EndDate = datenum(TYear,TMonth,eomday(TYear,TMonth));
	
	%fprintf(1,'Period %2d: Target Range [%s - %s]\n',K,datestr(StartDate,1),datestr(EndDate,1));
	
	% Step 8b - locate actual start and end dates in the data
	
	StartIndex = find(Returns.dates >= StartDate,1,'first');
	EndIndex = find(Returns.dates <= EndDate,1,'last');

	AnalysisPeriod(K) = Returns.dates(EndIndex);
	
	% Step 8c - determine active assets for current analysis period

	Active = true(NumAssets,1);
	for i = 1:NumAssets
		TestActive = sum(~isfinite(fts2mat(Returns.(AssetList{i})(StartIndex:EndIndex))));
		if TestActive > MaxNaNs;
			Active(i) = false;
		end
	end
	NumActive = sum(Active);

	fprintf(1,'  Points %5d:%5d: Dates = [%s : %s] Active Assets = %d\n', ...
		StartIndex,EndIndex,datestr(Returns.dates(StartIndex),1), ...
		datestr(Returns.dates(EndIndex),1),NumActive);

	% Step 8d - set up regression with active assets over current date range

	Dates = Returns.dates(StartIndex:EndIndex);
	AssetData = fts2mat(Returns.(AssetList));
	AssetData = AssetData(StartIndex:EndIndex,Active);
	FactorData = fts2mat(Returns.(FactorList));
	FactorData = FactorData(StartIndex:EndIndex,:);

	AssetData = AssetData - repmat(FactorData(:,4),1,size(AssetData,2));

	NumSamples = size(AssetData,1);

	Design = cell(NumSamples,1);
	for t = 1:NumSamples
		Design{t} = repmat([ 1, FactorData(t,1:3)],NumActive,1);
	end

	XDesign = cell(NumSamples,1);
	for t = 1:NumSamples
		XDesign{t} = repmat(FactorData(t,1:3),NumActive,1);
	end

	% Step 8e - set up seemingly-unrelated regression

	FactorDesign = convert2sur(Design,1:NumActive);
	XFactorDesign = convert2sur(XDesign,1:NumActive);
	
	% Step 8f - multivariate normal regression
	
	MaxIter = 1000;
	TolObj = 1.0e-10;
	TolParam = 1.0e-8;
	
	[Param, Covar] = ecmmvnrmle(AssetData, FactorDesign, MaxIter, TolParam, TolObj);
	StdParam = ecmmvnrstd(AssetData, FactorDesign, Covar, 'fisher');

	% estimate model with Alpha = 0 restriction
	[XParam, XCovar] = ecmmvnrmle(AssetData, XFactorDesign, MaxIter, TolParam, TolObj);

	% Step 8g - compute log-likelihood functions

	FFLLF(K) = ecmmvnrobj(AssetData, FactorDesign, Param, Covar);
	FFXLLF(K) = ecmmvnrobj(AssetData, XFactorDesign, XParam, XCovar);
	
	FFDoF(K) = NumActive;
	FFXDoF(K) = NumActive;

	% Step 8h - pack estimates into analysis variables
	
	Param = reshape(Param,4,NumActive)';
	StdParam = reshape(StdParam,4,NumActive)';
	
	Alpha(Active,K) = Param(:,1);
	Beta(Active,K) = Param(:,2);
	SMB(Active,K) = Param(:,3);
	HML(Active,K) = Param(:,4);

	Sigma(Active,K) = sqrt(diag(Covar));
	
	StdAlpha(Active,K) = StdParam(:,1);
	StdBeta(Active,K) = StdParam(:,2);
	StdSMB(Active,K) = StdParam(:,3);
	StdHML(Active,K) = StdParam(:,4);
	
	% Step 8i - increment year by 1

	TYear = TYear + 1;
end

% Step 9 - save results

save FFResults AssetList Alpha Beta HML SMB Sigma StdAlpha StdBeta StdHML StdSMB FFLLF FFDoF
save XResults AssetList FFXLLF FFXDoF
