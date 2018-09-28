clear all
close all

n = 11;

pset = 0:0.01:1;

majorVoteProbSet = [];
for i = 1 : length(pset)
    p = pset(i);
    proj = 0;
    for j = 6 : n
        proj = proj + nchoosek(n,j)*p^j*(1-p)^(n-j);
    end
    majorVoteProbSet = [majorVoteProbSet;proj];
end


h1 = figure(1)
plot(pset, majorVoteProbSet,'linewidth',2);
hold on
plot(pset, pset,'linewidth',1,'linestyle','--')
xlabel('accuracy probability')
ylabel('majority vote accuracy probability')
set(gca,'linewidth',2,'fontsize',15,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.04;0.02]);
pbaspect([1 1 1])
set(gca,'Box','on')
saveTightFigure(h1,'majorVoteAccuracyProbabilityVsSingleVote.pdf')