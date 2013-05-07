#load up the libaries
library(randomForest)
library(e1071)
library(bigmemory)
library(biganalytics)
library(parallel)
library(MASS)
library(Rmpi)
library(nnet)
library(neuralnet)
library(triangle)
library(amap)
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

bbTestNormSel = attach.big.matrix('/home/results/bbTestNormSel.desc')
bbTrainNormSel = attach.big.matrix('/home/results/bbTrainNormSel.desc')
bbExtraNormSel = attach.big.matrix('/home/results/bbExtraNormSel.desc')
#load('blackbox/extra_data_sample.RData')

#pca (only have to do it once)
colMeansExtra = colMeans(extra_data_sample[,])

bbExtraNorm = (extra_data_sample[,]-matrix(rep(colMeansExtra, nrow(extra_data_sample)), 
                                           nrow=nrow(extra_data_sample), byrow=T))
bbTestNorm = (bbTest[,]-matrix(rep(colMeansExtra, nrow(extra_data_sample)), 
                                           nrow=nrow(extra_data_sample), byrow=T))
bbTrainNorm = bbTrain[,]
bbTrainNorm[,2:ncol(bbTrain)] = (bbTrain[,2:ncol(bbTrain)]-matrix(rep(colMeansExtra, nrow(bbTrain)),
                                                                  nrow=nrow(bbTrain), byrow=T))


p = pca(bbExtraNorm)

#save the PCA's and then use them to only use the first N components (representing 90% of the variance) in the analysis.
k = which(cumsum(p$eig)/sum(p$eig)>.9)[1]
bbExtraNormSel = bbExtraNorm%*%p$loadings[,1:k]
bbTestNormSel = bbTestNorm%*%p$loadings[,1:k]

bbTrainNormSel = cbind(solution=bbTrainNorm[,1], bbTrainNorm[,2:ncol(bbTrain)]%*%p$loadings[,1:k])

as.big.matrix(bbExtraNormSel,backingfile="bbExtraNormSel", backingpath="/home/results/")
as.big.matrix(bbTrainNormSel,backingfile="bbTrainNormSel", backingpath="/home/results/")
as.big.matrix(bbTestNormSel,backingfile="bbTestNormSel", backingpath="/home/results/")

# #do a very simple randomForest on the lababled training data, and fit everything else into it
data = bbTrainNormSel[,]
colnames(data)=gsub(pattern=' ', replacement='', x=colnames(data))
form = paste("solution~", paste(colnames(data)[1:50+1], sep='', collapse='+'),sep='',collapse='')
nn = neuralnet(form, data=data, hidden=c(8), rep=2,threshold=.1)
solsTrain = compute(nn, data[,2:51])
solsTest = compute(nn, bbTestNormSel[,1:50])
solsExtra = compute(nn, bbExtraNormSel[,1:50])

#newLab = predict(rmodel, bbTrain[-sampleRows, sampleFeats])
#error on the solutions
errUnlab = errFunUnlab(round(solsTest$net.result), as.numeric(data[,1])) 
errNewLab = errFunlab(round(solsTrain$net.result), data[,1])

# #write solutions to output file
 write.table(file="blackbox/submission.csv", x=sprintf("%.1f", round(solsTest$net.result)), row.names=F, quote=F, col.names=F)


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
                  sampIndLab=sample(1:nrow(bbTrain), round(.10*nrow(bbTrain))), 
                  sampIndUnLab=sample(1:nrow(extra_data_sample), round(.2*nrow(extra_data_sample))),
                  parallel=T, cpus=4, weights=c(1,0,0), cl=cl,returnResults=F,method="ldaUnlab", OVA=F,
                  featSample = sample(1:ncol(extra_data_sample), round(.25*ncol(extra_data_sample))),
                  unLabWidth = ncol(extra_data_sample)
              )
params = list(labeled="/home/results/bbTrainNormSel.desc", unlabeled="/home/results/bbExtraNormSel.desc",
              labLen = nrow(bbTrainNormSel), unLabLen=nrow(bbTestNormSel),
              sampIndLab=sample(1:nrow(bbTrainNormSel), round(.001*nrow(bbTrainNormSel))), 
              sampIndUnLab=sample(1:nrow(bbTestNormSel), round(1*nrow(bbTestNormSel))),
              parallel=T, cpus=4, weights=c(1,0,1), cl=cl,returnResults=F,method="nnetUnLab", OVA=F,
              featSample = 1:30,#ncol(bbTestNormSel),#sample(1:ncol(bbTestNormSel), round(.25*ncol(bbTestNormSel))),
              unLabWidth = ncol(bbTestNormSel)
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
  saveon=100,
  iterations = iterations,
  #initialSolution=as.data.frame(matrix(rbinom(pop*ncol(extra_data_sample), 1, prob=1/100),
  #                                     ncol(extra_data_sample),pop)),
  initialSolution=as.data.frame(matrix(solsRep,ncol=pop)),
  
  iterativemethod="salitecont",#gaSimpleInt",#gaSimple",#salitebin",#"randbin",
  iterativemethodparams=list(replacerate=.1, sadist=100, mutrate=.05, 
                             gacontdist=1,min=1,max=9),
  optEval="bbEvalMain",
  optEvalparams=params
)

iterativeObj = list(iterativeParams=iterativeParams)

# #load up a previous run and continue it another N trials
# load("iterativeObjTemp.RData")
# iterativeObj.eval=iterativeObj
# iterativeObj$iterativeParams$iterations = iterativeObj$iterativeParams$iterations+200

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
params$unlabeled = "/home/results/bbTestNormSel.desc"
params$sampIndLab = 1:10#(nrow(bbTrain)-1)
params$sampIndUnLab = 1:nrow(bbTest)

res = bbEval(featSet)

#write.table(file="blackbox/submission.csv", x=sprintf("%.1f", as.numeric(res$preds)), row.names=F, quote=F, col.names=F)
write.table(file="blackbox/submission.csv", x=sprintf("%.1f", as.numeric(featSet)), row.names=F, quote=F, col.names=F)

stopCluster(cl)
