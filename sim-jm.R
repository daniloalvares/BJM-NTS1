
simJMdata <- function(JMobject, n, tmax, minobs){
  
  # Longitudinal regression parameters
  betas <- JMobject$coefficients$betas
  # Longitudinal error SD
  sd.e <- JMobject$coefficients$sigma
  # Survival regression parameters
  gammas <- JMobject$coefficients$gammas
  # Association paramenter
  alpha <- JMobject$coefficients$alpha
  # Weibull scale parameter
  phi <- JMobject$coefficients$sigma.t
  # Variance-covariance matrix for random effects
  D.b <- matrix(JMobject$coefficients$D[1:4],ncol=2)
  # Random effects
  b <- mvrnorm(n,rep(0,nrow(D.b)),D.b)
  
  # Binary variable
  X <- rbinom(n,1,0.5)
  
  # Simulating survival data
  eta <- as.vector(cbind(1, X)%*%gammas)
  u <- runif(n)
  
  # Inverse method for survival model with Weibull specification
  invS <- function(t,u,i,...){
    h <- function(s,...){
      XX <- cbind(1,X[i],s)
      ZZ <- cbind(1,s)
      fval <- as.vector(XX%*%betas + rowSums(ZZ*b[rep(i,nrow(ZZ)),]))
      return(exp(log(phi)+(phi-1)*log(s)+eta[i]+alpha*fval))
    }
    return(integrate(h,lower=0,upper=t)$value + log(u))
  }
  
  trueTime <- NULL
  for(i in 1:n){
    Up <- 50
    tries <- 5
    Root <- try(uniroot(invS,interval=c(1e-05,Up),u=u[i],i=i)$root,TRUE)
    while(inherits(Root,"try-error") && tries>0){
      tries <- tries - 1 
      Up <- Up + 500
      Root <- try(uniroot(invS,interval=c(1e-05,Up),u=u[i],i=i)$root,TRUE)	
    }
    trueTime[i] <- ifelse(!inherits(Root,"try-error"), Root, NA)	
  }
  
  # True time, censoring, observed time and event indicator
  trueTime <- ifelse(is.na(trueTime),tmax,trueTime)
  Ctimes <- runif(n,0,tmax)
  Time <- pmin(trueTime,Ctimes)
  event <- as.numeric(trueTime<=Ctimes)
  
  # Longitudinal time points
  nmaxobs <- ceiling(tmax) + minobs - 1
  nobs <- ceiling(Time) + minobs
  times <- NULL
  for(i in 1:n){
    times <- c(times,c(seq(0,Time[i],len=nobs[i])[-nobs[i]],rep(NA,nmaxobs-nobs[i]+1)))
  }
  
  # Simulating longitudinal data
  repX <- rep(X,times=rep(nmaxobs,n))
  repb0 <- rep(b[,1],times=rep(nmaxobs,n))
  repb1 <- rep(b[,2],times=rep(nmaxobs,n))
  e <- rnorm(n*nmaxobs,0,sd.e)
  Y <- betas[1] + repb0 + betas[2]*repX + (betas[3]+repb1)*times + e
  id <- rep(1:n,times=rep(nmaxobs,n))
  dta.long <- data.frame(id = id, Y = Y, time = times, X = repX)
  
  # Longitudinal and survival data
  dta.long <- dta.long[which(times <= rep(Time,rep(nmaxobs,n))),]
  dta.surv <- data.frame(id = unique(id), Time = Time, status = event, X = X)
  
  return(list(longitudinal=dta.long,survival=dta.surv))
}
