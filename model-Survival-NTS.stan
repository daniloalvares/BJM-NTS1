data{
  int n;
  int nbetaS;
  vector[n] x;
  vector[n] Time;
  vector[n] bi0;
  vector[n] bi1;
  vector[3] betaLmean;
  vector[n] death;
  real<lower=0> eta;
}

parameters{
  vector[nbetaS] betaS;
  real alpha;
  real<lower=0> w[n];
}

model{
// ------------------------------------------------------
//        LOG-LIKELIHOOD FOR SURVIVAL SUBMODEL                
// ------------------------------------------------------
{
  vector[n] h;
  vector[n] cumHaz;

  for(i in 1:n){
       // Hazard
       h[i] = w[i]*exp( betaS[1] + betaS[2] * x[i] + alpha * (betaLmean[1] + bi0[i] + (betaLmean[2] + bi1[i])*Time[i] + betaLmean[3] * x[i]) );
       // Cumulative hazard H[t] = int_0^t h[u] du
       cumHaz[i] = ( h[i] - w[i]*exp( betaS[1] + betaS[2] * x[i] + alpha * (betaLmean[1] + bi0[i] + betaLmean[3] * x[i] ) ) ) / ( alpha * (betaLmean[2] + bi1[i]) );

       target += gamma_lpdf(w[i] | eta, eta);
       target += death[i]*log(h[i]) - cumHaz[i];
  }
} 
// ------------------------------------------------------
//                       LOG-PRIORS                       
// ------------------------------------------------------
   // Survival fixed effects
   target += normal_lpdf(betaS | 0, 100);  
 
   // Association parameter
   target += normal_lpdf(alpha | 0, 100);

}