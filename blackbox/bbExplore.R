bbTest = read.csv('blackbox/test.csv')
bbTrain = read.csv('blackbox/train.csv')
bbSubmission = read.csv('blackbox/sample_submission.txt')

library(randomForest)
source('blackbox/bbFuns.R')

#do a very simple randomForest on the lababled training data, and fit everything else into it
sampleFeats = sample(2:ncol(bbTrain),10)
sampleRows = sample(1:nrow(bbTrain), round(.5*nrow(bbTrain)))
rmodel = randomForest(x=bbTrain[sampleRows,sampleFeats],y=as.factor(bbTrain[sampleRows,1]) )
sols = predict(rmodel, bbTest[,sampleFeats-1])
newLab = predict(rmodel, bbTrain[-sampleRows, sampleFeats])
#error on the solutions
errUnlab = errFunUnlab(as.numeric(sols), bbTrain[,1]) 
errNewLab = errFunlab(as.numeric(newLab), bbTrain[-sampleRows,1])

#write solutions to output file
write.table(file="blackbox/submission.csv", x=sprintf("%.1f", as.numeric(sols)), row.names=F, quote=F, col.names=F)

#let's get more sophisticated:
#first what are the percentages of each population item?  
table(bbTrain[,1])
hist(bbTrain[,1])

#we have two objective functions:  
#1 labeled data group error (use part of data to train data into groups)
#2 unlabeled data group distribution error (how far are the groups off the labeled data bin rates)
#3 hidden is the unlabeled grouping error

#so iteratively and evolutionarily optimize error types 1 and 2 using a half and half method
#half train, half validate

#select best features to put
#select best modeling types

#use an ep to both at same time.


