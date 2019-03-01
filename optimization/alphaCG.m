%% Conjugate Gradient Step Length alpha
%% Muhammad Izzatullah 
%% 15 August 2018


function a = alphaCG(m,model,dk,D)

%% Computing epsilon
    dkmax = 0;
    mmax = 0;
    eps1 = 1e-6;
    
    dkmax = max(dkmax,max(dk));
    mmax = max(mmax,max(m));
    
    epsilon = (0.01*mmax)/(dkmax+eps1);
    
%% Preparing data

    mtemp = m + epsilon*dk;
    
    Dsim = F(m,model);
    Dtemp = F(mtemp,model);
    
%% Computing aplha
    res = Dsim - D;
    dpres = D + res;
    dtemp = Dtemp - dpres;
    
    res = sum(res,3);
    dtemp = sum(dtemp,3);
    
    a1 = dot(dtemp,-res);
    a2 = dot(dtemp,dtemp);
    
    a = real((a1*epsilon)/(a2+eps1));
    

end