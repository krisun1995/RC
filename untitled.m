P=(1-IM)*data(:,J+3);   
                       
						
RB=IM*data(:,J+3);               
PTT=PTT+sum(data(:,J+3))*Fr(J);  

for I=1:SZ   
    if WU(I)+P(I)>=EP(I) 
        EU(I)=EP(I);EL(I)=0;ED(I)=0;
    else if WL(I)>=C*WLM
            EU(I)=WU(I)+P(I);EL(I)=(EP(I)-EU(I))*WL(I)/WLM;ED(I)=0;
        else if WL(I)>=C*(EP(I)-(WU(I)+P(I)))
                EU(I)=WU(I)+P(I);EL(I)=C*(EP(I)-EU(I));ED(I)=0;
            else
                EU(I)=WU(I)+P(I);EL(I)=WL(I);ED(I)=C*(EP(I)-EU(I))-EL(I);
            end
        end
    end
    E(I)=EU(I)+EL(I)+ED(I);
    PE(I)=P(I)-E(I);
    a=WMM*(1-(1-W(I)/WM)^(1/(B+1)));  
    if PE(I)<=0
        R(I)=0;
    else if a+PE(I)<=WMM
        R(I)=PE(I)+W(I)-WM+WM*(1-(PE(I)+a)/WMM)^(B+1);
        else
        R(I)=PE(I)+W(I)-WM;
        end
    end
    WU(I+1)=WU(I)+P(I)-EU(I)-R(I);
    WL(I+1)=WL(I)-EL(I);
    WD(I+1)=WD(I)-ED(I);
    if WU(I+1)>WUM
        WL(I+1)=WL(I)-EL(I)+WU(I+1)-WUM;
        WU(I+1)=WUM;
    end
    if WL(I+1)>WLM
        WD(I+1)=WD(I)-ED(I)+WL(I+1)-WLM;
        WL(I+1)=WLM;
    end
    if WD(I+1)>WDM
        WD(I+1)=WDM;
    end
    W(I+1)=WU(I+1)+WL(I+1)+WD(I+1);
    if PE(I)==0
        FR(I)=0;
    else if R(I)/PE(I)>1
            FR(I)=1;
        else
        FR(I)=R(I)/PE(I);
        end
    end
    if PE(I)<=FC 
        RG2(I)=R(I);RD2(I)=0;
    else
        RG2(I)=FC*FR(I);RD2(I)=(PE(I)-FC)*FR(I);
    end
        if I==1
            AU=SMM*(1-(1-(S1(I)*FR0/FR(I))/SM)^(1/(1+EX)));
            if PE(I)<=0
            RS3(I)=0;RI3(I)=0;RG3(I)=0;S1(I+1)=S1(I)*(1-KI-KG);
            else if PE(I)+AU<SMM
                RS3(I)=FR(I)*(PE(I)+S1(I)*FR0/FR(I)-SM+SM*(1-(PE(I)+AU)/SMM)^(EX+1));
                 else
                RS3(I)=FR(I)*(PE(I)+S1(I)*FR0/FR(I)-SM);
                 end
            S=S1(I)*FR0/FR(I)+(R(I)-RS3(I))/FR(I);
            RI3(I)=KI*S*FR(I);
            RG3(I)=KG*S*FR(I);
            S1(I+1)=S*(1-KI-KG);
            end
        else
            AU=SMM*(1-(1-(S1(I)*FR(I-1)/FR(I))/SM)^(1/(1+EX)));
            if PE(I)<=0
            RS3(I)=0;RI3(I)=0;RG3(I)=0;S1(I+1)=S1(I)*(1-KI-KG);
             else if PE(I)+AU<SMM
                     RS3(I)=FR(I)*(PE(I)+S1(I)*FR(I-1)/FR(I)-SM+SM*(1-(PE(I)+AU)/SMM)^(EX+1));
                  else
                     RS3(I)=FR(I)*(PE(I)+S1(I)*FR(I-1)/FR(I)-SM);
                  end
            S=S1(I)*FR(I-1)/FR(I)+(R(I)-RS3(I))/FR(I);
            RI3(I)=KI*S*FR(I);
            RG3(I)=KG*S*FR(I);  
            S1(I+1)=S*(1-KI-KG);
            end
        end
end