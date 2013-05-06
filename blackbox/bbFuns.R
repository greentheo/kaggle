##bbFuns (helper funs for error functions and such)

errFunUnlab = function(sols, labelDat){
  sols = c(sols, as.character(unique(labelDat)))
  err = mean(
      abs(table(sols)/length(sols) - 
        table(labelDat)/length(labelDat))
    )
  return(err)
}

errFunlab = function(newLab, trueLab){
  length(which(newLab==trueLab))/length(trueLab)
}
bbEvalMain = function(solution, params){
  
    
  #generate a new sampleInd
  #params$sampIndLab=sample(1:params$labLen, round(.75*params$labLen)) 
  #params$sampIndUnLab=sample(1:params$unLabLen, round(.25*params$unLabLen))
  #params$featSample = sample(1:params$unLabWidth, round(.25*params$unLabWidth))
  #res = try(sfClusterApplyLB(solution, bbEval, params=params))
  #clusterApplyLB(cl, 1:10, function(x)mean(runif(100)))
 
  clusterExport(cl,"params")
  #res= lapply(solution, bbEval)
  res = parLapplyLB(cl, solution, bbEval)
  
  #transfrom results into a final score (since we have more than one error metric)
  errLab = unlist(lapply(res, function(x)return(x$errLab)))
  errUnLab = unlist(lapply(res, function(x)return(x$errUnLab)))
  #errLab = rep(0, length(errLab))
  errLab[names(errLab)[sort(errLab, decreasing=F, index.return=T)$ix]] = c(1:length(errLab))
  #errUnLab = rep(0, length(errUnLab))
  errUnLab[names(errUnLab)[sort(errUnLab, decreasing=F, index.return=T)$ix]] =c(1:length(errUnLab))
  finalErr = params$weights %*% rbind(errLab, errUnLab)
  
  return(list(eval=finalErr, raw=res))
}
bbEval = function(solution){
  
  sampIndLab=params$sampIndLab
  sampIndUnLab=params$sampIndUnLab
  feats=which(solution==1)
  labeled = attach.big.matrix(params$labeled)
  unlabeled = attach.big.matrix(params$unlabeled)
  #build an OVA classifier
  labels = unique(labeled[,1])
  mods = list()
  resLabProb = list()
  resLab = list()
  resUnlabProb = list()
  resUnlab = list()
  #sampInd = sample(1:nrow(labeled), round(sampSize*nrow(labeled)))
  #OVA models, preds and results
  makeMods = function(OVA=T){
    if(OVA){
      for(lab in labels){
        labeledOVA = deepcopy(labeled)
        #labeledOVA = params$labeled
        
        labInd=which(labeledOVA[,1]==lab)
        labeledOVA[labInd,1] = 1
        labeledOVA[-labInd,1]=0
        labn = as.character(lab)
        #browser()
        if(params$method == "randomForest"){
          mods[[labn]] = randomForest(x=labeledOVA[sampIndLab, feats], y=as.factor(labeledOVA[sampIndLab,1]),ntree=200,)  
          resLabProb[[labn]] = predict(mods[[labn]], newdata=labeledOVA[-sampIndLab, feats], type="prob")
          resUnlabProb[[labn]] = predict(mods[[labn]], newdata=unlabeled[sampIndUnLab, feats-1], type="prob")
        }
        if(params$method == "lda"){
          mods[[labn]] = lda(x=labeledOVA[sampIndLab, feats], grouping=as.factor(labeledOVA[sampIndLab,1]))
          resLabProb[[labn]] = predict(mods[[labn]], labeledOVA[-sampIndLab, feats])$posterior
          resUnlabProb[[labn]] = predict(mods[[labn]], unlabeled[-sampIndUnLab, feats-1])$posterior       
        }
        
      }
     
    }else{
      if(params$method == "randomForest"){
        mods[[1]] = randomForest(x=labeled[sampIndLab, feats], y=as.factor(labeled[sampIndLab,1]),ntree=40)  
        resLabProb[[1]] = predict(mods[[1]], newdata=labeled[-sampIndLab, feats], type="prob")
        resUnlabProb[[1]] = predict(mods[[1]], newdata=unlabeled[sampIndUnLab, feats-1], type="prob")
      }
      if(params$method == "lda"){
        mods[[1]] = lda(x=labeled[sampIndLab, feats], grouping=as.factor(labeled[sampIndLab,1]))
        resLabProb[[1]] = predict(mods[[1]], labeled[-sampIndLab, feats])$class
        resUnlabProb[[1]] = predict(mods[[1]], unlabeled[sampIndUnLab, feats-1])$class       
      }
      
      #browser() 
      if(params$method== "ldaUnlab"){
         
        mods[[1]] = lda(x=unlabeled[sampIndUnLab, params$featSample], 
                        grouping=as.factor(solution[sampIndUnLab]))
        resLabProb[[1]] = predict(mods[[1]], labeled[-sampIndLab, params$featSample+1])$class
        resUnlabProb[[1]] = predict(mods[[1]], unlabeled[sampIndUnLab, params$featSample])$class 
      }
      if(params$method=="nnetUnLab"){
        data = cbind(solution, as.matrix(unlabeled))
        colnames(data)=gsub(pattern=' ', replacement='', x=colnames(data))
        form = paste("solution~", paste(colnames(data)[2:11], sep='', collapse='+'),sep='',collapse='')
        mods[[1]] = nnet(x=as.matrix(unlabeled), y=solution, size=1)
        resLabProb[[1]] = predict(mods[[1]], )
        colnames(unlabeled)
      }
      #browser() 
      
    }    
    #assign classes to data points and get error measures
    
    #put into matrix format
    #classMatLab = matrix(as.numeric(unlist(resLab)), nrow=length(classMatLab[[1]]), ncol=length(classMatLab))-1
    #colnames(classMatLab) = names(resLab)
    extractPred <- function (resProb, predsNames) {
      probMatLab0 = lapply(resProb, function(x)return(x[,"0"]))
      probMatLab1 = lapply(resProb, function(x)return(x[,"1"]))
      probMatLab0 = matrix(as.numeric(unlist(probMatLab0)), nrow=length(probMatLab0[[1]]), ncol=length(probMatLab0))
      probMatLab1 = matrix(as.numeric(unlist(probMatLab1)), nrow=length(probMatLab1[[1]]), ncol=length(probMatLab1))
      preds = predsNames[apply(probMatLab1-probMatLab0, 1, function(x){
        which.max(abs(x))
      })]
      return(preds)
    }
    if(OVA){
      predsLab = extractPred(resLabProb, names(resLabProb))
      errLab = errFunlab(predsLab, labeled[-sampIndLab, 1])
      predsUnLab = extractPred(resUnlabProb, names(resUnlabProb))
      errUnLab = errFunUnlab(predsUnLab, labeled[,1])
    }else{
      
      predsUnLab = resUnlabProb[[1]]
      errLab=errFunlab((resLabProb[[1]]),labeled[-sampIndLab,1])
      errUnLab=errFunUnlab(predsUnLab, labeled[,1])
    }
    if(params$returnResults) preds=predsUnLab
    else preds=NULL
      
    return(list(errLab=errLab, 
                errUnLab=-errUnLab, 
                preds=preds))
  }
  res = try(makeMods(OVA=params$OVA))
  if(class(res)=="try-error") res = list(errLab=0, errUnLab=-1, preds=NULL)
  return(res)
}

performancePlot = function(iterativeObj.eval,...,plotC = 1){
  par(mfrow=c(plotC,plotC))
    perfList = lapply(iterativeObj.eval$history, function(x){
      return(as.data.frame(
        matrix(unlist(x$evaluation$raw), nrow=2))
      )
    })
  
  maxList = unlist(lapply(iterativeObj.eval$history, function(x){
    which.max(x$evaluation$eval)
  }))
  #plot the average of the group, and the max for both metrics
  meanPerf = matrix(unlist(lapply(perfList, function(x){
                    rowMeans(x)
                    })), nrow=2)
  maxPerf = matrix(unlist(lapply(perfList, function(x){
    apply(x,1,max)
  })), nrow=2)
  maxPerfEval = vector("list", length(perfList))
  for(i in 1:length(perfList)){
    maxPerfEval[[i]] = perfList[[i]][,maxList[i]] 
  }
  maxPerfEval = matrix(unlist(maxPerfEval), ncol = length(perfList))  
    
  plot(
    meanPerf[1,], main="Labeling accuracy by Iteration",
    type="l", ylim=c(min(c(meanPerf[1,], maxPerf[1,])), max(c(meanPerf[1,], maxPerf[1,])))
  )
  lines(
    maxPerf[1,],
    lty=2
  )
  lines(maxPerfEval[1, ], lty=3, col="red")
  legend("topright", c("avg", "max","maxPerf"), lty=c(1,2,3), col=c("black", 'black', 'red'))
  
  
  plot(
    meanPerf[2,], main="Unlabeled Data Error by Iteration",
    type="l", ylim=c(min(c(meanPerf[2,], maxPerf[2,])), max(c(meanPerf[2,], maxPerf[2,])))
  )
  lines(maxPerf[2,], lty=2)
  lines(maxPerfEval[2, ], lty=3, col='red')
  legend("topright", c("avg", "max","maxPerf"), lty=c(1,2,3), col=c("black", 'black', 'red'))

  #now plot the number of signals being used
  numFeats = lapply(iterativeObj.eval$history,function(x){
    apply(x$solution, 2, function(y){
      sum(as.numeric(y))
    })
    })
  avgFeat = unlist(lapply(numFeats, mean))
  maxFeat = unlist(lapply(numFeats, max))
  plot(avgFeat, type="l", main="Number of Features Used", 
       ylim=c(min(c(avgFeat, maxFeat)), max(c(avgFeat, maxFeat)))
       )
  lines(maxFeat, lty=2)
  legend("bottomright", c("avg", "max"), lty=c(1,2))
  
  #plot the signals used and the variety in the population
  #plotFeatHistory(iterativeObj.eval)
  par(mfrow=c(1,1))
  
}
prepareSlaves = function(){
  library(randomForest)
  library(bigmemory)
  library(biganalytics)
  library(snowfall)
  library(MASS)
  library(Rmpi)
  source('~/githubrepo/kaggle/blackbox/bbFuns.R')
  source('~/githubrepo/RPackages/commonFuns.R')
  libraryRaw('~/githubrepo/RPackages/ep/')
  libraryRaw('~/githubrepo/RPackages/stocksOpt/')
}

updatePlot = function(filename="~/githubrepo/kaggle/iterativeObjTemp.RData", interval=10){
  while(1==1){
    plotUpdate=function(){
      
      load(filename)
      pdf(paste(filename, ".pdf",sep=''))
      performancePlot(iterativeObj)
      dev.off()
      
    }
    cat("updateing plot... ", filename)
    a=try(plotUpdate)
    if(class(a)!="error")cat(' updated \n')
    Sys.sleep(time=interval)
  }
  
}

