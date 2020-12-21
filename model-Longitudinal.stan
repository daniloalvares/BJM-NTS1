functions{
// ------------------------------------------------------
//     LINEAR PREDICTOR FOR THE LONGITUDINAL SUBMODEL                
// ------------------------------------------------------ 
    vector linear_predictor(vector x, vector times, int[] ID, vector betaL, matrix bi){
         int N = num_elements(times);
         vector[N] out;

         out = betaL[1] + bi[ID,1] + betaL[2]*times + rows_dot_product(bi[ID,2],times) + betaL[3]*x[ID];

         return out;
    } 
// ------------------------------------------------------ 
}


data{
  int N; 
  int n;
  vector[N] y;
  vector[n] x;
  vector[N] times;
  int<lower=1,upper=n> ID[N];
  int nbetaL;
}


parameters{
  vector[nbetaL] betaL;
  real<lower=0> sigma2_b[2];
  real<lower=-1, upper=1> rho; 
  real<lower=0> sigma_e;
  matrix[n,2] bi;
}


transformed parameters{
  cov_matrix[2] Sigma;

  Sigma[1,1] = sigma2_b[1];
  Sigma[2,2] = sigma2_b[2];
  Sigma[1,2] = sqrt(sigma2_b[1]*sigma2_b[2])*rho;
  Sigma[2,1] = Sigma[1,2];
}


model{
// ------------------------------------------------------
//        LOG-LIKELIHOOD FOR LONGITUDINAL SUBMODEL                
// ------------------------------------------------------
{
   vector[N] mu; 

   // Linear predictor
   mu = linear_predictor(x, times, ID, betaL, bi);

   // Longitudinal Normal log-likelihood
   target += normal_lpdf(y | mu, sigma_e);
}  
// ------------------------------------------------------
//                       LOG-PRIORS                       
// ------------------------------------------------------
   // Longitudinal fixed effects
   target += normal_lpdf(betaL | 0, 100);

   // Random-effects
   for(i in 1:n){ target += multi_normal_lpdf(bi[i,1:2] | rep_vector(0.0,2), Sigma); }

   // Random-effects variance
   target += inv_gamma_lpdf(sigma2_b | 0.01, 0.01);

   // Random-effects correlation
   target += beta_lpdf((rho+1)/2 | 0.5, 0.5);

   // Residual error standard deviation
   target += cauchy_lpdf(sigma_e | 0, 5);

}