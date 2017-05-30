function Yesti = KNN(Xval, Xent, Yent, k)

    N=size(Xent,1);
    M=size(Xval,1);
    
    Yesti=zeros(M,1);
    dis=zeros(N,1);
    ind=zeros(k,1);
    
    Yaux=zeros(k,1); 
    
    for j=1:M
                for h=1:N
                   
                    
                    sum = 0;
                    
                    for l=1:18
                       
                         sum = sum + (Xval(j, l) - Xent(h, l)).^2;
                        
                    end
                    
                    dis(h) = sqrt(sum);
                end
                
                dis2=sort(dis);
                
                dis2=dis2';
                l=1;
                while l<=k
                   
                    aux =  find(dis==dis2(l));
                    
                    len = size(aux, 1);
                    
                    for z=1:len
                        if l > k
                           z=len+1; 
                        else 
                            ind(l) = aux(z);
                        
                            l = l + 1;
                        
                        end                   
                       
                    end
                                        
                end
               
                for w=1:k
                   
                    Yaux(w) = Yent(ind(w));
                    
                end
                
                               
                Yesti(j,1) = mode(Yaux);
                
    end
   
end

