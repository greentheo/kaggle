#shiny monitoring UI
library(shiny)
source('~/githubrepo/kaggle/blackbox/bbFuns.R')
# Define server logic required to generate and plot a random distribution
shinyServer(function(input, output) {
  
  # Expression that generates a plot of the distribution. The expression
  # is wrapped in a call to renderPlot to indicate that:
  #
  #  1) It is "reactive" and therefore should be automatically 
  #     re-executed when inputs change
  #  2) Its output type is a plot 
  #
  output$distPlot <- renderPlot({
    load('~/githubrepo/kaggle/iterativeObjTemp.RData')
    perfList = lapply(iterativeObj.eval$history, function(x){
      return(as.data.frame(
        matrix(unlist(x$evaluation$raw), nrow=2))
      )
    })
    #plot the average of the group, and the max for both metrics
    meanPerf = matrix(unlist(lapply(perfList, function(x){
      rowMeans(x)
    })), nrow=2)
    maxPerf = matrix(unlist(lapply(perfList, function(x){
      apply(x,1,max)
    })), nrow=2)
    
    
    plot(
      meanPerf[1,], main="Labeling accuracy by Iteration",
      type="l", ylim=c(min(c(meanPerf[1,], maxPerf[1,])), max(c(meanPerf[1,], maxPerf[1,])))
    )
    lines(
      maxPerf[1,],
      lty=2
    )
    legend("topright", c("avg", "max"), lty=c(1,2))
    
    # generate an rnorm distribution and plot it
    #dist <- rnorm(input$obs)
    #hist(dist)
  })
})