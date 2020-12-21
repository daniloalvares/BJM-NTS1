rm(list=ls(all=TRUE))

# Required sources
library(mvtnorm)
library(JM)
library(rstan)

# Function to simulate joint models
source("sim-jm.R")

# Prothro data from the JM package
data("prothro")

# Longitudinal variable
prothro$logpro <- log(prothro$pro)

# Fit the joint model to extract estimated parameters
lmeFit <- lme(logpro ~ treat + time, random = ~ time | id, data = prothro)
survFit <- coxph(Surv(Time, death) ~ treat, data = prothros, x = TRUE)
jointFit <- jointModel(lmeFit, survFit, timeVar = "time", method = "weibull-PH-aGH", scaleWB=1)

# "True value" for survival regression coefficients (gamma and alpha)
surv.true.par <- c(jointFit$coef$gammas[2],jointFit$coef$alpha)

# Number of repetition
repetition <- 20

# Variables to store the means, standard deviations and times of each approach
mean.JS <- mean.STS <- mean.NTS <- matrix(NA,nrow=repetition,ncol=2)
SD.JS <- SD.STS <- SD.NTS <- matrix(NA,nrow=repetition,ncol=2)
time.JS <- time.STS <- time.NTS <- rep(NA,repetition)


for(i in 1:repetition){
  
  # Simulate joint data
  # n: number of individuals
  # tmax: maximum observational time
  # minobs: minimum number of longitudinal observations
  print(i)
  set.seed(i)
  jointdata <- simJMdata(jointFit, n=200, tmax=15, minobs=3)

  # Data format 
  l.data <- jointdata$longitudinal
  s.data <- jointdata$survival

  # Number of longitudinal observations per individual
  M <- as.numeric(table(l.data$id))

  ###############################################
  #      Joint specification (JS) approach      #
  ###############################################
  start_time1 <- Sys.time()
  fitJS <- stan(file   = "model-JS.stan", 
                data   = list(N=nrow(l.data), n=nrow(s.data), y=l.data$Y, times=l.data$time, nbetaS=2, nbetaL=3,
                           Time=s.data$Time, death=s.data$status, x=s.data$X, ID=rep(1:nrow(s.data),M)),          
                warmup = 1000,                 
                iter   = 2000,                
                chains = 1,
                cores  = getOption("mc.cores",1))
  end_time1 <- Sys.time()

  # Results
  mean.JS[i,] <- summary(fitJS)$summary[5:6,1]
  SD.JS[i,] <- summary(fitJS)$summary[5:6,3]
  time.JS[i] <- end_time1 - start_time1
  ###############################################


  ###############################################
  #      Standard Two-Stage (STS) approach      #
  ###############################################
  start_time2 <- Sys.time()
  # Stage 1 - Longitudinal submodel
  fitL <- stan(file   = "model-Longitudinal.stan", 
               data   = list(N=nrow(l.data), n=nrow(s.data), y=l.data$Y, times=l.data$time,
                           x=s.data$X, ID=rep(1:nrow(s.data),M), nbetaL=3),          
               warmup = 1000,                 
               iter   = 2000,                
               chains = 1,
               cores  = getOption("mc.cores",1))

  # Extracting random-effects
  bi <- extract(fitL)$bi
  b <- apply(bi,c(2,3),mean)

  # Extracting fixed-effects
  beta1 <- unlist(extract(fitL, par="betaL[1]"))
  beta2 <- unlist(extract(fitL, par="betaL[2]"))
  beta3 <- unlist(extract(fitL, par="betaL[3]"))

  # Mean
  betaLmean <- c(mean(beta1),mean(beta2),mean(beta3))

  # Stage 2 - Survival submodel
  fitSTS <- stan(file  = "model-Survival-STS.stan", 
                data   = list(n=nrow(s.data), bi0=b[,1], bi1=b[,2], betaL=betaLmean, nbetaS=2,
                            Time=s.data$Time, x=s.data$X, death=s.data$status),          
                warmup = 500,                 
                iter   = 1000,                
                chains = 1,
                cores  = getOption("mc.cores",1))
  end_time2 <- Sys.time()

  # Results
  mean.STS[i,] <- summary(fitSTS)$summary[2:3,1]
  SD.STS[i,] <- summary(fitSTS)$summary[2:3,3]
  time.STS[i] <- end_time2 - start_time2
  ###############################################


  ###############################################
  #       Novel Two-Stage (NTS) approach        #
  ###############################################
  start_time3 <- Sys.time()
  # Stage 1 - Longitudinal submodel
  fitL <- stan(file   = "model-Longitudinal.stan", 
               data   = list(N=nrow(l.data), n=nrow(s.data), y=l.data$Y, times=l.data$time,
                           x=s.data$X, ID=rep(1:nrow(s.data),M), nbetaL=3),          
               warmup = 1000,                 
               iter   = 2000,                
               chains = 1,
               cores  = getOption("mc.cores",1))

  # Extracting random-effects
  bi <- extract(fitL)$bi
  b <- apply(bi,c(2,3),mean)

  # Extracting fixed-effects
  beta1 <- unlist(extract(fitL, par="betaL[1]"))
  beta2 <- unlist(extract(fitL, par="betaL[2]"))
  beta3 <- unlist(extract(fitL, par="betaL[3]"))

  # Mean
  betaLmean <- c(mean(beta1),mean(beta2),mean(beta3))

  # Stage 2 - Survival submodel
  fitNTS <- stan(file  = "model-Survival-NTS.stan", 
                data   = list(n=nrow(s.data), bi0=b[,1], bi1=b[,2], betaLmean=betaLmean, nbetaS=2,
                            eta=1.5, Time=s.data$Time, x=s.data$X, death=s.data$status),          
                warmup = 500,                 
                iter   = 1000,                
                chains = 1,
                cores  = getOption("mc.cores",1))
  end_time3 <- Sys.time()

  # Results
  mean.NTS[i,] <- summary(fitNTS)$summary[2:3,1]
  SD.NTS[i,] <- summary(fitNTS)$summary[2:3,3]
  time.NTS[i] <- end_time3 - start_time3
  ###############################################
}


###############################################
#        Comparative Posterior Summary        #
###############################################
# Boxplot
# par(mfrow=c(1,3))
# boxplot(mean.JS, ylim=c(-3,0), main="JS"); abline(h=surv.true.par, col="red")
# boxplot(mean.STS, ylim=c(-3,0), main="STS"); abline(h=surv.true.par, col="red")
# boxplot(mean.NTS, ylim=c(-3,0), main="NTS"); abline(h=surv.true.par, col="red")

# Table
res <- cbind(surv.true.par, apply(mean.JS,2,mean), apply(mean.STS,2,mean), apply(mean.NTS,2,mean))
res <- rbind(res,c(NA,mean(time.JS), mean(time.STS), mean(time.NTS)))
res <- round(res,3)
colnames(res) <- c("True", "JS", "STS", "NTS")
rownames(res) <- c("Posterior Mean (gamma1)", "Posterior Mean (alpha)", "Comp. Time")
print(res)
###############################################
