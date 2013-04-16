##bbFuns (helper funs for error functions and such)

errFunUnlab = function(sols, labelDat){
  err = mean(
      abs(table(sols)/length(sols) - 
        table(labelDat)/length(labelDat))
    )
  return(err)
}

errFunlab = function(newLab, trueLab){
  length(which(newLab==trueLab))/length(trueLab)
}