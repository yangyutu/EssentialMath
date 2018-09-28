%FFestimateCAPM Estimate rolling parameters for CAPM 1-factor model.

clear all

% Step 1 - load in raw data and set up list of Fama & French factors

load FFUniverseCAPM

% Step 2 - convert total return prices to total returns

Returns = (CAPMUniverse - lagts(CAPMUniverse,1)) ./ lagts(CAPMUniverse,1);

% Step 3 - trim date range to match date range of factor returns

Returns = Returns(all(isfinite(fts2mat(extfield(Returns, CAPMFactorList))),2));

% Step 4 - get information about data

SeriesList = fieldnames(Returns, 1);
AssetList = setdiff(SeriesList, CAPMFactorList);

NumSeries = numel(SeriesList);
NumAssets = numel(AssetList);

% Step 5 - primary controls

TMonth = 9;				% terminal month for each historical estimation period
Window = 5;				% historical estimation period in years
MaxNaNs = 4*260;		% max number of daily NaNs in an estimation period

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

CAPMAlpha = NaN(NumAssets, NumPeriods);
CAPMBeta = NaN(NumAssets, NumPeriods);

CAPMSigma = NaN(NumAssets, NumPeriods);

CAPMStdAlpha = NaN(NumAssets, NumPeriods);
CAPMStdBeta = NaN(NumAssets, NumPeriods);

CAPMLLF = NaN(NumPeriods,1);
CAPMDoF = NaN(NumPeriods,1);

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
	FactorData = fts2mat(Returns.(CAPMFactorList));
	FactorData = FactorData(StartIndex:EndIndex,:);

	AssetData = AssetData - repmat(FactorData(:,1),1,size(AssetData,2));

	NumSamples = size(AssetData,1);

	CAPMDesign = cell(NumSamples,1);
	for t = 1:NumSamples
		CAPMDesign{t} = repmat([ 1, FactorData(t,2)], NumActive,1);
	end
	
	% Step 8e - set up seemingly-unrelated regression

	CAPMFactorDesign = convert2sur(CAPMDesign,1:NumActive);
	
	% Step 8f - multivariate normal regression
	
	MaxIter = 1000;
	TolObj = 1.0e-10;
	TolParam = 1.0e-8;
	
	[CAPMParam, CAPMCovar] = ecmmvnrmle(AssetData, CAPMFactorDesign, MaxIter, TolParam, TolObj);
	CAPMStdParam = ecmmvnrstd(AssetData, CAPMFactorDesign, CAPMCovar, 'fisher');

	% Step 8g - compute log-likelihood functions

	CAPMLLF(K) = ecmmvnrobj(AssetData, CAPMFactorDesign, CAPMParam, CAPMCovar);
	CAPMDoF(K) = NumActive;
	
	% Step 8h - pack estimates into analysis variables
	
	CAPMParam = reshape(CAPMParam,2,NumActive)';
	CAPMStdParam = reshape(CAPMStdParam,2,NumActive)';
	
	CAPMAlpha(Active,K) = CAPMParam(:,1);
	CAPMBeta(Active,K) = CAPMParam(:,2);
	
	CAPMSigma(Active,K) = sqrt(diag(CAPMCovar));
	
	CAPMStdAlpha(Active,K) = CAPMStdParam(:,1);
	CAPMStdBeta(Active,K) = CAPMStdParam(:,2);

	% Step 8i - increment year by 1

	TYear = TYear + 1;
end

% Step 9 - save results

save CAPMResults AssetList CAPMAlpha CAPMBeta CAPMSigma CAPMStdAlpha CAPMStdBeta CAPMLLF CAPMDoF
