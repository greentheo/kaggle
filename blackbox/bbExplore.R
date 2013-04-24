#load up the libaries
library(randomForest)
library(bigmemory)
library(biganalytics)
library(parallel)
library(MASS)
library(Rmpi)
source('blackbox/bbFuns.R')
source('../RPackages/commonFuns.R')
libraryRaw('../RPackages/ep/')
libraryRaw('../RPackages/stocksOpt/')

#bbTest = read.big.matrix('blackbox/test.csv',header=F, skip=1,backingfile="bbTest", backingpath="/home/results/")
#bbTrain = read.big.matrix('blackbox/train.csv',header=F, skip=1,backingfile="bbTrain", backingpath="/home/results/")
bbSubmission = read.csv('blackbox/sample_submission.txt')
bbTrain = attach.big.matrix("/home/results/bbTrain.desc")
bbTest = attach.big.matrix("/home/results/bbTest.desc")
extra_data_sample=attach.big.matrix("/home/results/extra_data_sample.desc")

#load('blackbox/extra_data_sample.RData')


# #do a very simple randomForest on the lababled training data, and fit everything else into it
sampleFeats = sample(2:ncol(bbTrain),round(ncol(bbTrain)*.5))
sampleRows = sample(1:nrow(bbTrain), round(1*nrow(bbTrain)))
rmodel = lda(x=bbTrain[sampleRows,sampleFeats],grouping=as.factor(bbTrain[sampleRows,1]) )
sols = predict(rmodel, bbTest[,sampleFeats-1])$class

#newLab = predict(rmodel, bbTrain[-sampleRows, sampleFeats])
#error on the solutions
#errUnlab = errFunUnlab(as.numeric(sols), bbTrain[,1]) 
#errNewLab = errFunlab(as.numeric(newLab), bbTrain[-sampleRows,1])

# #write solutions to output file
# write.table(file="blackbox/submission.csv", x=sprintf("%.1f", as.numeric(sols)), row.names=F, quote=F, col.names=F)


#we have two objective functions:  
#1 labeled data group error (use part of data to train data into groups)
#2 unlabeled data group distribution error (how far are the groups off the labeled data bin rates)
#3 hidden is the unlabeled grouping error

#so iteratively and evolutionarily optimize error types 1 and 2 using a half and half method
#half train, half validate
cl=makeCluster(6)
clusterExport(cl, "prepareSlaves")
clusterCall(cl, prepareSlaves)
#parLapply(cl, c(1:10),function(x)mean(runif(100)))
#sfInit(parallel=T, cpus=4)
#sfExport("prepareSlaves")
#sfClusterCall(prepareSlaves)
params = list(labeled="/home/results/bbTrain.desc", unlabeled="/home/results/extra_data_sample.desc",
                  labLen = nrow(bbTrain), unLabLen=nrow(extra_data_sample),
                  sampIndLab=sample(1:nrow(bbTrain), round(.75*nrow(bbTrain))), 
                  sampIndUnLab=sample(1:nrow(extra_data_sample), round(.1*nrow(extra_data_sample))),
                  parallel=T, cpus=4, weights=c(1,0), cl=cl,returnResults=F,method="ldaUnlab", OVA=F,
                  featSample = sample(1:ncol(extra_data_sample), round(.25*ncol(extra_data_sample))),
                  unLabWidth = ncol(extra_data_sample)
              )
#clusterExport(cl, "params")
pop=60

#start an initial solution for the population
solsRep = rep(sols, pop)
mutsamp = sample(1:length(solsRep), round(.1*length(solsRep)))
solsRep[mutsamp] = round(runif(length(mutsamp), min=1, max=9))

iterations=100
iterativeParams = list(
  objectName="saBBOpt",
  saveon=10,
  iterations = iterations,
  #initialSolution=as.data.frame(matrix(rbinom(pop*ncol(extra_data_sample), 1, prob=1/100),
  #                                     ncol(extra_data_sample),pop)),
  initialSolution=as.data.frame(matrix(solsRep,ncol=pop)),
  
  iterativemethod="gaSimpleInt",#gaSimple",#salitebin",#"randbin",
  iterativemethodparams=list(replacerate=.1, sadist=2, mutrate=.1, gacontdist=1),
  optEval="bbEvalMain",
  optEvalparams=params
)

iterativeObj = list(iterativeParams=iterativeParams)

# #load up a previous run and continue it another N trials
# load("iterativeObjTemp.RData")
# iterativeObj.eval=iterativeObj
# iterativeObj$iterativeParams$iterations = iterativeObj$iterativeParams$iterations+1000

iterativeObj.eval = iterativeOpt(iterativeObj)

#performancePlot(iterativeObj)
pdf('blackbox/plots.pdf')
performancePlot(iterativeObj.eval)
dev.off()
#select best features to use for classifying

#select best modeling types

#use an ep to both at same time.

#write solutions from the best last trial
maxTri=length(iterativeObj.eval$history)
featSet = iterativeObj.eval$history[[maxTri]]$solution[,
                                which.max(iterativeObj.eval$history[[maxTri]]$evaluation$eval)]
params$returnResults=T
params$unlabeled = "/home/results/bbTest.desc"
params$sampIndLab = 1:(nrow(bbTrain)-1)
params$sampIndUnLab = 1:nrow(bbTest)

res = bbEval(featSet)

write.table(file="blackbox/submission.csv", x=sprintf("%.1f", as.numeric(res$preds)), row.names=F, quote=F, col.names=F)

stopCluster(cl)
