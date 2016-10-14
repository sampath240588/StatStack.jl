m=mocc
m=mdolocc
#m=mpen

custom_vars=[:dolocc_reduction,:occ_reduction,:pen_reduction,:data_b_e_nb,:data_nb_ne_b,:whyo,:iso]
required_vars=vcat([m.y_var,:panid,m.logvar],cfg[:random_demos],cfg[:random_campaigns],cfg[:scoring_vars])
vars=setdiff(m.vars,vcat(required_vars,custom_vars))

#SingleValue
m.removed_SingleLevelVars=FS_singleLevel(dfd,vars)
vars = rmVars(m.removed_SingleLevelVars)



    function rmVars(v::Array{Symbol})
        v=setdiff(v,[:group])
        return setdiff(vars,v)
    end

    function getDSname()
        mn = Dict("occ"=>"finaldata_occ","dolocc"=>"finaldata_dolocc","pen"=>"finaldata_pen")
        return get(mn, m.modelName,NA)
    end

    function rv()
        return convert(Array{Symbol},sort(   map(x->lowercase(x),rcopy("  setdiff(names("*getDSname()*"),c(all_mandatory_vars,scoring_vars,random_campaigns,random_demos,'panid')) "))  ))         
    end
    
    function rchk()
        onlyinJ=setdiff(vars,rv())
        onlyinR=setdiff(rv(),vars)
        lj=length(onlyinJ)
        lr=length(onlyinR)
        lj>lr ? DataFrame(onlyinJ=onlyinJ,onlyinR=vcat(onlyinR,fill(:x, lj-lr))) : DataFrame(DataFrame(onlyinJ=vcat(onlyinJ,fill(:x, lr-lj)),onlyinR=onlyinR))    
    end
    
    function isSame()
        println(getDSname()*" : J vals : ",length(vars),"  --- R vals : ",length(rv()))
        sort(vars)==sort(rv())
    end

    function rlst(v::String)
        sort(map(x->Symbol(lowercase(x)),rcopy(v)))
    end
 

include("/home/rmadmin/g/StatStack/src/RCode.jl")
#dfd=df_data[(df_data[:iso].==false)&(df_data[:buyer_pos_p1].==1),:]



if   m.modelName == "occ"
#OCC
R"""
 y_var <- "Trps_POS_P1"
finaldata_occ <- initial_data[Buyer_Pos_P1=="1",]
# drop variables with only one level
indices <- sapply(finaldata_occ,function(x) length(unique(x))==1)
finaldata_occ <- finaldata_occ[,!indices,with=F]
"""
end
if  m.modelName == "dolocc"
#DOLOCC
R"""
y_var <- "Dol_per_Trip_POS_P1"
finaldata_dolocc <- initial_data[Buyer_Pos_P1=="1",]
# drop variables with only one level
indices <- sapply(finaldata_dolocc,function(x) length(unique(x))==1)
finaldata_dolocc <- finaldata_dolocc[,!indices,with=F]
"""
end

#After Single level reove
isSame()

# GT: Remove cause these will get rmoved in mc test..... so need to QC
if   m.modelName == "occ"
#OCC
R"""
finaldata_occ <- as.data.frame(finaldata_occ)
mc_cols <- c("Fea_or_Dis_Trps_Shr_DPP_P1","Fea_or_Dis_Trps_Shr_DPP_P2","Fea_or_Dis_Trps_Shr_DPP_P3","Fea_or_Dis_Trps_Shr_DPP_P4")
finaldata_occ <- finaldata_occ[, !names(finaldata_occ) %in% mc_cols ] 
"""
end
if   m.modelName == "dolocc"
#DOLOCC
R"""
finaldata_dolocc <- as.data.frame(finaldata_dolocc)
mc_cols <- c("Fea_or_Dis_Trps_Shr_DPP_P1","Fea_or_Dis_Trps_Shr_DPP_P2","Fea_or_Dis_Trps_Shr_DPP_P3","Fea_or_Dis_Trps_Shr_DPP_P4")
finaldata_dolocc <- finaldata_dolocc[, !names(finaldata_dolocc) %in% mc_cols ] 
"""
end


#Copy R Dataset to avoid sampling
dfd=rcopy(getDSname())
lowercase!(dfd)
mc_cols = [:fea_or_dis_trps_shr_dpp_p1,:fea_or_dis_trps_shr_dpp_p2,:fea_or_dis_trps_shr_dpp_p3,:fea_or_dis_trps_shr_dpp_p4]
vars=setdiff(vars,mc_cols)


#  ---------------------------------

if  m.modelName == "occ"
#OCC
R"""
finaldata_occ <- as.data.frame(finaldata_occ)
if(exists('dolocc_outliers_panid_to_remove', envir = environment())){
  finaldata_occ <- finaldata_occ[, !(names(finaldata_occ) %in% c("Dol_per_Trip_POS_P1","Nonbuyer_Pre_P1","Buyer_Pre_P1","Dol_per_Trip_PRE_P1"))]
  finaldata_occ <- finaldata_occ[!finaldata_occ$panid %in% dolocc_outliers_panid_to_remove,]
}else{
  finaldata_occ <- finaldata_occ[, !(names(finaldata_occ) %in% c("Dol_per_Trip_POS_P1","Nonbuyer_Pre_P1","Buyer_Pre_P1","Dol_per_Trip_PRE_P1"))]
}

# set initial model formula
Formula_fixed <- paste0(paste0(y_var,"~ ",paste0(names(finaldata_occ[,!names(finaldata_occ)%in%c(y_var,"Trps_PRE_P1","panid",random_demos,random_campaigns,scoring_vars)]),collapse=" + ")," "),"+ offset(log(Trps_PRE_P1+1))")
options(warn=-1)

# run glm model
model_glm_total <- glm(Formula_fixed,data=finaldata_occ,family=poisson(link="log"))

pval_excluded_vars <- c(rownames(as.data.frame(which(summary(model_glm_total)$coefficient[,4]>0.7))),setdiff(names(model_glm_total$coefficients),rownames(summary(model_glm_total)$coefficients)))
finaldata_occ <- finaldata_occ[, !names(finaldata_occ) %in% c(pval_excluded_vars)] 
"""    
end
if  m.modelName == "dolocc"
#DOLOCC
R"""
#dolocc_mandatory_covariate <- c("Nonbuyer_Pre_P1")
#finaldata_dolocc <- finaldata_dolocc[, !names(finaldata_dolocc) %in% c(dolocc_mandatory_covariate)] 
dolocc_mandatory_covariate <- c()

# set initial model formula
finaldata_dolocc <- as.data.frame(finaldata_dolocc)
finaldata_dolocc <- finaldata_dolocc[, !(names(finaldata_dolocc) %in% c("Trps_POS_P1","Buyer_Pre_P1","Trps_PRE_P1"))]
Formula_fixed <- paste0(paste0(y_var,"~ ",paste0(names(finaldata_dolocc[,!names(finaldata_dolocc)%in%c(y_var,"panid","Dol_per_Trip_PRE_P1",random_demos,random_campaigns,scoring_vars)]),collapse=" + ")," ")," + offset(log(Dol_per_Trip_PRE_P1+1))")
options(warn=-1)

# run glm model
model_glm_total <- glm(Formula_fixed,data=finaldata_dolocc,family=Gamma(link="log"))
summary(model_glm_total)

##### Variable selection - Condition 1 #####
# Remove pvalues greater than 0.7 and rerun the glm model
pval_excluded_vars <- setdiff(c(rownames(as.data.frame(which(summary(model_glm_total)$coefficient[,4]>0.7))),setdiff(names(model_glm_total$coefficients),rownames(summary(model_glm_total)$coefficients))),dolocc_mandatory_covariate)
    
finaldata_dolocc <- finaldata_dolocc[, !names(finaldata_dolocc) %in% c(pval_excluded_vars)] 
"""
end
#filter(x->contains("nonbuyer_pre_p1",string(x)),vars)






#:fea_or_dis_trps_shr_dpp_p1
#:fea_or_dis_trps_shr_dpp_p4
#:fea_or_dis_trps_shr_dpp_p2
#:fea_or_dis_trps_shr_dpp_p3



#cmparr(pval_excluded_vars,j,[:fea_or_dis_trps_shr_dpp_p1,:fea_or_dis_trps_shr_dpp_p2,:fea_or_dis_trps_shr_dpp_p3,:fea_or_dis_trps_shr_dpp_p4])    
    #PVals
    m.glm1_pvals = xGLM(dfd, m.dist, m.y_var, m.logvar, vars  )  
    g1=m.glm1_pvals
    m.glm1_pvals_x=g1.sdf[g1.sdf[:pval].>0.7,:vars]
    vars = rmVars(vcat(m.glm1_pvals_x, g1.xvars ) )
    
isSame()
    

#  --------------------QC -------------------------------------
#sort(m.glm1_pvals_x) == rlst("pval_excluded_vars")
#cmparr(sort(m.glm1_pvals_x), rlst("pval_excluded_vars"))

#Cpn_UN_per_Trip_DPP_P4  -0.460885549 1.221531226  -0.37730149 7.059541e-01

# m.glm1_pvals.sdf[m.glm1_pvals.sdf[:vars].==:cpn_un_per_trip_dpp_p4,:]
#length(m.glm1_pvals.sdf[:vars])
#R"""length(summary(model_glm_total)$coefficients[,4])"""

#m.glm1_pvals.model
#R"""summary(model_glm_total)"""
#prd_1_qty_pre

# -0---------- QC end ---------------------------------------------



#writetable("CDW.csv", dfd, header = true)
if  m.modelName == "occ"
# OCC
R"""

# re-run glm model
Formula_fixed <- paste0(paste0(y_var,"~ ",paste0(names(finaldata_occ[,!names(finaldata_occ)%in%c(y_var,"Trps_PRE_P1","panid",random_demos,random_campaigns,scoring_vars)]),collapse=" + ")," "),"+ offset(log(Trps_PRE_P1+1))")
model_glm_condition1 <- glm(Formula_fixed,data=finaldata_occ,family=poisson(link="log"))
summary(model_glm_condition1)
rm(model_glm_total)

##### Variable selection - Condition 2 #####
# Remove variables that have z value lower than 2 and VIF greater than 15

# get variables with z value lower than 2
z_excluded_vars <- rownames(as.data.frame(which(abs(summary(model_glm_condition1)$coefficient[,3])<1.96)))

# get the variables with VIF greater than 15
multicol <- alias(model_glm_condition1)
multicols_toremove <- rownames(as.data.frame(multicol$Complete))

if (length(multicols_toremove)>0){
  finaldata_occ <- finaldata_occ[ , !names(finaldata_occ) %in% multicols_toremove] 
  Formula_fixed <-  paste0(paste0(y_var,"~ ",paste0(names(finaldata_occ[,!names(finaldata_occ)%in%c(y_var,"panid",random_demos,random_campaigns)]),collapse=" + ")," "),"+ offset(log(Trps_PRE_P1+1)) ")
  model_glm_new <- glm(Formula_fixed,data=finaldata_occ,family=poisson(link="log"))
} else {
  model_glm_new <- model_glm_condition1
}

vif_output <- (vif(model_glm_new))
vif_to_remove <- names(vif_output[vif_output > 15])

# final vars to exclude under condition 2
vars_condition2 <- intersect(z_excluded_vars,vif_to_remove)

# remove the variables under the above conditions
finaldata_occ <- finaldata_occ[ , !names(finaldata_occ) %in% c(vars_condition2,pval_excluded_vars)] 
"""
end
if  m.modelName == "dolocc"
# DOLOCC
R"""
# re-run glm model
Formula_fixed <- paste0(paste0(y_var,"~ ",paste0(names(finaldata_dolocc[,!names(finaldata_dolocc)%in%c(y_var,"panid","Dol_per_Trip_PRE_P1",random_demos,random_campaigns,scoring_vars)]),collapse=" + ")," ")," + offset(log(Dol_per_Trip_PRE_P1+1))")
model_glm_condition1 <- glm(Formula_fixed,data=finaldata_dolocc,family=Gamma(link="log"))
summary(model_glm_condition1)
rm(model_glm_total)

##### Variable selection - Condition 2 #####
# Remove variables that have z value lower than 2 and VIF greater than 15

# get variables with z value lower than 2
z_excluded_vars  <- rownames(as.data.frame(which(abs(summary(model_glm_condition1)$coefficient[,3])<1.96)))

# get the variables with VIF greater than 15
multicol <- alias(model_glm_condition1)
multicols_toremove <- rownames(as.data.frame(multicol$Complete))

if (length(multicols_toremove)>0){
  finaldata_dolocc <- finaldata_dolocc[ , !names(finaldata_dolocc) %in% multicols_toremove]
  Formula_fixed <- paste0(paste0(y_var,"~ ",paste0(names(finaldata_dolocc[,!names(finaldata_dolocc)%in%c(y_var,"panid")]),collapse=" + ")," ")," + offset(log(Dol_per_Trip_PRE_P1+1))")
  model_glm_new <- glm(Formula_fixed,data=finaldata_dolocc,family=Gamma(link="log"))
} else {
  model_glm_new <- model_glm_condition1
}

vif_output <- (vif(model_glm_new))
vif_to_remove <- names(vif_output[vif_output > 15])

# final vars to exclude under condition 2
vars_condition2 <- setdiff(intersect(z_excluded_vars,vif_to_remove),dolocc_mandatory_covariate)
finaldata_dolocc <- finaldata_dolocc[ , !names(finaldata_dolocc) %in% vars_condition2] 
"""
end

    #Z & Vif
    m.glm2_ZnVIF = xGLM(dfd, m.dist, m.y_var, m.logvar, vars  ) 
    g2=m.glm2_ZnVIF
    vif!(g2)
    #z = g2.sdf[g2.sdf[:zval].<1.96,:vars]
    z = g2.sdf[abs(g2.sdf[:zval]).<1.96,:vars]
    #QC: sort(z) == sort(rlst("z_excluded_vars"))       cmparr(convert(Array{Symbol},z),rlst("z_excluded_vars"))
    v = g2.sdf[ !DataArrays.isna(g2.sdf[:vif])&(g2.sdf[:vif].>15),:vars]
    #QC: cmparr(convert(Array{Symbol},v),rlst("vif_to_remove"))
    m.glm2_ZnVIF_x =intersect(z,v)
    #QC: m.glm2_ZnVIF_x == rlst("vars_condition2")
    #QC: cmparr( m.glm2_ZnVIF_x ,rlst("vars_condition2"))
    vars = rmVars(vcat(m.glm2_ZnVIF_x, g2.xvars) )


#sort(m.glm2_ZnVIF_x)==sort(map(x->Symbol(lowercase(x)) ,rcopy("vars_condition2")))

isSame()

# --------QC ------------------
#cmparr(convert(Array{Symbol},m.glm2_ZnVIF_x), rlst("vars_condition2"))
#cmparr(convert(Array{Symbol},z), rlst("z_excluded_vars")
#g2.model
#R"""summary(model_glm_condition1)$coefficient"""
#R"""vif_output"""
#g2.sdf
#---------QC end --------------

if  m.modelName == "occ"
#OCC
R"""
Formula_fixed <- paste0(paste0(y_var,"~ ",paste0(names(finaldata_occ[,!names(finaldata_occ)%in%c(y_var,"panid","Trps_PRE_P1",random_demos,random_campaigns,scoring_vars)]),collapse=" + ")," "),"+ offset(log(Trps_PRE_P1+1))")

# run model
model_glm <- glm(Formula_fixed,data=finaldata_occ,family=poisson(link="log"))
summary(model_glm)
rm(model_glm_condition1)

# p-values and signs checks
neutralvars <- setdiff(names(finaldata_occ),c(positivevars,negativevars))
pvalues <- summary(model_glm)$coefficients[,4][-1]
keep_covariate <- names(pvalues[pvalues<pvalue_lvl])
coeffs <- sign(summary(model_glm)$coefficients[,1])
varstokeep <- c();initialvars <- c()
for (i in negativevars) if (i %in% names(coeffs[coeffs<0])) varstokeep <- c(varstokeep,i)
neg <- varstokeep
for (j in positivevars) if (j %in% names(coeffs[coeffs>0])) varstokeep <- c(varstokeep,j)
pos <- setdiff(varstokeep,neg)
for (l in neutralvars) varstokeep <- c(varstokeep,l)  #append neut to varstokeep

for (k in varstokeep) if (k %in% keep_covariate) initialvars <- c(initialvars,k)

"""
end
if  m.modelName == "dolocc"
#DOLOCC
R"""
# re-run glm model
Formula_fixed <- paste0(paste0(y_var,"~ ",paste0(names(finaldata_dolocc[,!names(finaldata_dolocc)%in%c(y_var,"panid","Dol_per_Trip_PRE_P1",random_demos,random_campaigns,scoring_vars)]),collapse=" + ")," ")," + offset(log(Dol_per_Trip_PRE_P1+1))")
model_glm <- glm(Formula_fixed,data=finaldata_dolocc,family=Gamma(link="log"))
summary(model_glm)
rm(model_glm_condition1)

# p-values and signs checks
neutralvars <- setdiff(names(finaldata_dolocc),c(positivevars,negativevars))
pvalues <- summary(model_glm)$coefficients[,4][-1]
keep_covariate <- names(pvalues[pvalues<pvalue_lvl])
coeffs <- sign(summary(model_glm)$coefficients[,1])
varstokeep <- c();initialvars <- c()
for (i in negativevars) if (i %in% names(coeffs[coeffs<0])) varstokeep <- c(varstokeep,i)
    neg <- varstokeep
for (j in positivevars) if (j %in% names(coeffs[coeffs>0])) varstokeep <- c(varstokeep,j)
    pos <- setdiff(varstokeep,neg)
for (l in neutralvars) varstokeep <- c(varstokeep,l)
for (k in varstokeep) if (k %in% keep_covariate) initialvars <- c(initialvars,k)
#if(dolocc_mandatory_covariate %in% names(finaldata_dolocc) & length(setdiff(dolocc_mandatory_covariate,initialvars))>0){
#  initialvars <- c(dolocc_mandatory_covariate,initialvars)
#}

"""
end

#r=Symbol(lowercase(rcopy("initialvars")))

function chkSigns(m::MModel, vars::Array{Symbol}, dfd::DataFrame, cfg::OrderedDict)  # Pvalue & Signs
    g = xGLM(dfd, m.dist, m.y_var, m.logvar, vars  )  
    neutralvars = setdiff(vars,vcat(cfg[:negativevars],cfg[:positivevars])) 
    #keep_covariate = g.sdf[ (g.sdf[:pval].<cfg[:pvalue_lvl]) & (g.sdf[:vars].!=:intercept),:vars]
    #QC: sort(keep_covariate)==sort(rlst("keep_covariate"))
    neg=intersect(cfg[:negativevars],g.sdf[g.sdf[:coef].<0,:vars])
    #QC: sort(neg)==sort(rlst("neg"))
    #QC: cmparr(neg,rlst("neg"))
    pos=intersect(cfg[:positivevars],g.sdf[g.sdf[:coef].>0,:vars])
    #QC: sort(pos)==sort(rlst("pos"))
    varstokeep = intersect(vcat(neutralvars, pos,neg) ,  g.sdf[ g.sdf[:pval].<cfg[:pvalue_lvl] ,:vars] )
    #m.glm3_PnSigns_x= setdiff(vars, varstokeep )
    #vars = rmVars(vcat(m.glm3_PnSigns_x, g3.xvars) )
    #QC: sort(convert(Array{Symbol},varstokeep)) == sort(rlst("initialvars"))
    return g, varstokeep
end

(m.glm3_PnSigns, initialvars) = chkSigns(m, vars, dfd, cfg)
#sort(initialvars)==rlst("initialvars")


#    # Pvalue & Signs
#    m.glm3_PnSigns = xGLM(dfd, m.dist, m.y_var, m.logvar, vars  )  # Pvalue & Signs
#    g3=m.glm3_PnSigns
#    neutralvars = setdiff(vars,vcat(cfg[:negativevars],cfg[:positivevars])) 
#keep_covariate = g3.sdf[ (g3.sdf[:pval].<cfg[:pvalue_lvl]) & (g3.sdf[:vars].!=:intercept),:vars]
##QC: sort(keep_covariate)==sort(rlst("keep_covariate"))
#    neg=intersect(cfg[:negativevars],g3.sdf[ g3.sdf[:coef].<0,:vars])
##QC: sort(neg)==sort(rlst("neg"))
#    pos=intersect(cfg[:positivevars],g3.sdf[ g3.sdf[:coef].>0,:vars])
##QC: sort(pos)==sort(rlst("pos"))
#    varstokeep = intersect(vcat(neutralvars, pos,neg) ,  g3.sdf[ g3.sdf[:pval].<cfg[:pvalue_lvl] ,:vars] )
#    #m.glm3_PnSigns_x= setdiff(vars, varstokeep )
#    #vars = rmVars(vcat(m.glm3_PnSigns_x, g3.xvars) )
##QC: sort(convert(Array{Symbol},varstokeep)) == sort(rlst("initialvars"))




#"""
## QC Varstokeep
#Symbol(lowercase(rcopy("initialvars"))) == varstokeep[1]
##rvars=rlst("varstokeep")
#sort(cfg[:positivevars]) == sort(rlst("positivevars"))
#sort(cfg[:negativevars]) == sort(rlst("negativevars"))
#cmparr(rvs,vcat(cfg[:random_demos],cfg[:random_campaigns],vars))
#sort(neutralvars) == rlst("neutralvars")
#sort(varstokeep) == rlst("varstokeep")
## -------------
#"""

if  m.modelName == "occ"
#OCC
R"""
#set the formula for the second model after cleaning the first model
Formula_final <- paste0(paste0(y_var,"~ ",paste0(initialvars,collapse=" + ")," "),"+ group+ offset(log(Trps_PRE_P1+1))")

model_glm_2 <-  glm(Formula_final,data=finaldata_occ,family=poisson(link="log"))
summary(model_glm_2)
rm(model_glm)

# final cleaning of second model based on signs and pvalues
pvalues_2 <- summary(model_glm_2)$coefficients[,4][-1]
keep_covariate_2 <- names(pvalues_2[pvalues_2 < pvalue_lvl])
coeffs_2 <- sign(summary(model_glm_2)$coefficients[,1])
varstokeep <- c();vars_2 <- c()
for (i in negativevars) if (i %in% names(coeffs_2[coeffs_2 < 0])) varstokeep <- c(varstokeep,i)
for (j in positivevars) if (j %in% names(coeffs_2[coeffs_2 > 0])) varstokeep <- c(varstokeep,j)
for (l in neutralvars) varstokeep <- c(varstokeep,l)
for (k in varstokeep) if (k %in% keep_covariate_2) vars_2 <- c(vars_2,k)

"""
end
if  m.modelName == "dolocc"
#DOLOCC
R"""
#set the formula for the second model after cleaning the first model
Formula_fixed <- paste0(paste0(y_var,"~ ",paste0(initialvars,collapse=" + ")," ")," + group + offset(log(Dol_per_Trip_PRE_P1+1))")
model_glm_2 <-  glm(Formula_fixed,data=finaldata_dolocc,family=Gamma(link="log"))
summary(model_glm_2)
rm(model_glm)

# final cleaning of second model based on signs and pvalues
pvalues_2 <- summary(model_glm_2)$coefficients[,4][-1]
keep_covariate_2 <- names(pvalues_2[pvalues_2 < pvalue_lvl])
coeffs_2 <- sign(summary(model_glm_2)$coefficients[,1])
varstokeep <- c();vars_2 <- c()
for (i in negativevars) if (i %in% names(coeffs_2[coeffs_2 < 0])) varstokeep <- c(varstokeep,i)
for (j in positivevars) if (j %in% names(coeffs_2[coeffs_2 > 0])) varstokeep <- c(varstokeep,j)
for (l in neutralvars) varstokeep <- c(varstokeep,l)
for (k in varstokeep) if (k %in% keep_covariate_2) vars_2 <- c(vars_2,k)
"""
end
#r=Symbol(lowercase(rcopy("vars_2")))
(m.glm4_PnSignsClean, vars_2) = chkSigns(m, convert(Array{Symbol},vcat(initialvars,[:group])) , dfd, cfg)


R"""
# final correlation check (find pair vars with correlation greater then 0.8 and drop the one with bigger pvalue) #### Maybe 0.6!
if(length(vars_2)>1){
  finaldata_occ_num <- as.data.table(finaldata_occ[,names(finaldata_occ) %in% vars_2])
  t1 <- as.data.frame(finaldata_occ_num[,sapply(finaldata_occ_num,is.numeric)])
  finaldata_occ_num <- as.data.frame(finaldata_occ_num)
  finaldata_occ_num <- finaldata_occ_num[,c(which(t1==T))]
  temp_cor <- as.matrix(cor(finaldata_occ_num))
  temp_cor[lower.tri(temp_cor, diag = T)] <- NA
  temp_corIndex <- which(temp_cor > 0.8 |temp_cor < (-0.8), arr.ind = TRUE)
  cordata <- data.table(col1=colnames(temp_cor)[temp_corIndex[,1]],col2=colnames(temp_cor)[temp_corIndex[,2]])
  
  if(nrow(cordata)>0) {
    drop_cor_vars <- c()
    for (i in 1:nrow(cordata)) {
      if(cordata[i][,1,with=F]=="group"|cordata[i][,2,with=F]=="group"){
        if(cordata[i][,1,with=F]=="group") drop_cor_vars <- c(drop_cor_vars,as.character(cordata[i][,2,with=F]))
        if(cordata[i][,2,with=F]=="group") drop_cor_vars <- c(drop_cor_vars,as.character(cordata[i][,1,with=F]))
      } else {
        if (summary(model_glm_2)$coefficients[which(rownames(summary(model_glm_2)$coefficients)==as.character(cordata[i][,1,with=F])),4] > summary(model_glm_2)$coefficients[which(rownames(summary(model_glm_2)$coefficients)==as.character(cordata[i][,2,with=F])),4]) {
          drop_cor_vars <- c(drop_cor_vars,as.character(cordata[i][,1,with=F]))
        } else {
          drop_cor_vars <- c(drop_cor_vars,as.character(cordata[i][,2,with=F]))
        }
      }
    }
    vars_2 <- setdiff(vars_2,drop_cor_vars)
  }
}


"""



#end

#modelme(mocc, df_data[(df_data[:iso].==false)&(df_data[:buyer_pos_p1].==1),:] )   





"""
function xcorr(df::DataFrame,vars::Array{Symbol})
    vx=Symbol[]
    vy=Symbol[]
    vC=Float64[]
    function f(x::Symbol,y::Symbol)
        c=cor(df[x],df[y])
        push!(vx,x)
        push!(vy,y)
        push!(vC,c)
    end
    map(x-> map(y->f(x,y) ,vars)  ,vars)
    
    for i in 1:length(vC)
        if (vC[i] > 0.8) & (vx[i]!=vy[i]) 
            println(vx[i]," ~ ",vy[i]," : ",vC[i])  
        end
    end
end
xcorr(df_data,m.vars)

"""