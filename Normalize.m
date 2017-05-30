function Z = normalize(Data,MU,SIG)

    N=length(MU);
    Z=[];
    
    for i=1:N
        
        vector = (Data(:,i)-MU(i))./SIG(i);
        Z = [Z,vector];
                
    end

end