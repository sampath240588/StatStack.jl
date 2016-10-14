######################################
#--------------OCCASION--------------#
######################################

occ.start.time <- Sys.time()

# set response variable
y_var <- "Trps_POS_P1"

# keep only the buyers
finaldata_occ <- initial_data[Buyer_Pos_P1=="1",]

# drop variables with only one level
indices <- sapply(finaldata_occ,function(x) length(unique(x))==1)
finaldata_occ <- finaldata_occ[,!indices,with=F]

#######################################
#-------- Variable Selection ---------#
#######################################

# remove dolocc outliers
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
summary(model_glm_total)

##### Variable selection - Condition 1 #####
# Remove pvalues greater than 0.7 and rerun the glm model
pval_excluded_vars <- c(rownames(as.data.frame(which(summary(model_glm_total)$coefficient[,4]>0.7))),setdiff(names(model_glm_total$coefficients),rownames(summary(model_glm_total)$coefficients)))
finaldata_occ <- finaldata_occ[, !names(finaldata_occ) %in% c(pval_excluded_vars)] 

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
for (j in positivevars) if (j %in% names(coeffs[coeffs>0])) varstokeep <- c(varstokeep,j)
for (l in neutralvars) varstokeep <- c(varstokeep,l)
for (k in varstokeep) if (k %in% keep_covariate) initialvars <- c(initialvars,k)

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

#######################################
#------------ Final Model ------------#
#######################################

#set the formula for the final model
Formula_final <-paste0( paste0(y_var,"~ ",paste0(vars_2,collapse=" + ")," "),"+ group + offset(log(Trps_PRE_P1+1))")
occ_final_model <-  glm(Formula_final,data=finaldata_occ,family=poisson(link="log"))
summary(occ_final_model)
rm(model_glm_2)

# final model
# final cleaning of model based on signs
pvalues_2 <- summary(occ_final_model)$coefficients[,4][-1]
keep_covariate_2 <- names(pvalues_2[pvalues_2 < pvalue_lvl])
coeffs <- sign(summary(occ_final_model)$coefficients[,1])
finalvars <- c();finalvars1 <- c()
for (i in negativevars) if (i %in% names(coeffs[coeffs<0])) finalvars1 <- c(finalvars1,i)
for (j in positivevars) if (j %in% names(coeffs[coeffs>0])) finalvars1 <- c(finalvars1,j)
for (l in neutralvars) finalvars1 <- c(finalvars1,l)
for (k in finalvars1) if (k %in% keep_covariate_2) finalvars <- c(finalvars,k)

#set final model
Formula_final <- paste0(paste0(y_var,"~ ",paste0(finalvars,collapse=" + ")," "),"+ group + offset(log(Trps_PRE_P1+1))")
occ_final_model <-  glm(Formula_final,data=finaldata_occ,family=poisson(link="log"))
summary(occ_final_model)

options(warn=0)

finaldata_occ <- droplevels(finaldata_occ)
occ_random_campaigns <- intersect(random_campaigns,names(finaldata_occ))

if(length(occ_random_campaigns)>0){
  occ_random_campaigns_factors <- paste0("+(0+group|",occ_random_campaigns,")")
} else {
  occ_random_campaigns_factors <- NULL
}
random_demos_factors <- paste0( "+(0+group|",random_demos,")" )

##################################################################################################################

# run glmer model
occ.mixed.start.time <- Sys.time()
Formula_final <- paste0(c(paste0(paste0(y_var,"~ ",paste0(finalvars,collapse=" + ")," "),"+ offset(log(Trps_PRE_P1+1)) + group"),random_demos_factors,occ_random_campaigns_factors),collapse='')
occ_final_model <- glmer(Formula_final,data=finaldata_occ,family=poisson(link="log"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
occ.mixed.model.execution.time <- Sys.time()-occ.mixed.start.time
out_resid <- order(residuals(occ_final_model),decreasing = T)
finaldata_occ <- as.data.table(finaldata_occ)
occ_total_outliers_panid <- finaldata_occ[out_resid,panid]
segment_panid <- data_NB_NE_B[,panid]
occ_outliers_panid_segm <- intersect(occ_total_outliers_panid,segment_panid)

# create a txt file with the summary of the first glmer model 
logsavefname <- "Occasion_First_Run_Results.txt"
sink(paste0(initial_Occ_modelling_output,logsavefname))
print(Sys.time()-occ.start.time)
print(summary(occ_final_model)$coef)
print(ranef(occ_final_model))
sink()

# remove outliers if needed (negative estimation for exposure)
occ_outliers_panid_to_remove <- c()
occ_outliers_panid_segm_new <- c()
occ_outliers_panid_to_remove_threshold <- c()

flag <- c()
coef_flag <- c()
for(i in setdiff(levels(finaldata_occ$group),"0")) flag <- append(flag,paste0('group',i))

for(j in flag){
  coef_flag <- append(coef_flag,coef(summary(occ_final_model))[j,"Estimate"])
}

if(length(occ_random_campaigns)>0){
  min_random_coeff <- min(unlist(lapply(ranef(occ_final_model)[occ_random_campaigns],FUN=min)))
  max_random_coeff <- max(unlist(lapply(ranef(occ_final_model)[occ_random_campaigns],FUN=max)))
} else {
  min_random_coeff <- 0
}

if(min_random_coeff < -0.05 & min(coef_flag) + min_random_coeff < 0){
  out_resid_positive <- order(residuals(occ_final_model),decreasing = T)
  out_resid_negative <- order(residuals(occ_final_model),decreasing = F)
  occ_outliers_panid_positive_extreme_panid <- finaldata_occ[out_resid_positive,panid]
  occ_outliers_panid_negative_extreme_panid <- finaldata_occ[out_resid_negative,panid]
  for(i in 1:length(occ_random_campaigns)){
    min_random_coeff <- min(unlist(lapply(ranef(occ_final_model)[occ_random_campaigns][i],FUN=min)))
    if(min_random_coeff < -0.05 & min(coef_flag) + min_random_coeff < 0){
      max_random_coeff <- max(unlist(lapply(ranef(occ_final_model)[occ_random_campaigns][i],FUN=max)))
      random_frame <- as.data.frame(ranef(occ_final_model)[occ_random_campaigns][i])
      random_table <- as.data.table(random_frame)
      random_table[,level:=row.names(random_frame)]
      negative_extreme_levels <- random_table[eval(parse(text=paste0(names(ranef(occ_final_model)[occ_random_campaigns][i]),'.group1')))==min_random_coeff,]$level
      positive_extreme_levels <- random_table[eval(parse(text=paste0(names(ranef(occ_final_model)[occ_random_campaigns][i]),'.group1')))== max_random_coeff]$level
      negative_extreme_panid <- finaldata_occ[eval(parse(text=paste0(names(ranef(occ_final_model)[occ_random_campaigns][i])))) %in% negative_extreme_levels,panid] # -13%
      positive_extreme_panid  <- finaldata_occ[eval(parse(text=paste0(names(ranef(occ_final_model)[occ_random_campaigns][i])))) %in% positive_extreme_levels,panid] # +12%
      occ_outliers_panid_segm1 <- intersect(occ_outliers_panid_negative_extreme_panid,negative_extreme_panid)
      occ_outliers_panid_segm1 <- occ_outliers_panid_segm1[1:round(0.02*(length(negative_extreme_panid)))]
      if(max_random_coeff > 0.05) {
        occ_outliers_panid_segm2 <- intersect(occ_outliers_panid_positive_extreme_panid,positive_extreme_panid)
        occ_outliers_panid_segm2 <- occ_outliers_panid_segm2[1:round(0.02*(length(positive_extreme_panid)))]
      } else {
        occ_outliers_panid_segm2 <- c()
      }
      occ_outliers_panid_segm_new <- c(occ_outliers_panid_segm1,occ_outliers_panid_segm2)
      finaldata_occ <- finaldata_occ[!finaldata_occ$panid %in% occ_outliers_panid_segm_new,]
    }
  }
  occ_final_model <- glmer(Formula_final,data=finaldata_occ,family=poisson(link="log"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
  coef_flag <- c()
  for(j in flag){
    coef_flag <- append(coef_flag,coef(summary(occ_final_model))[j,"Estimate"])
  }
  min_random_coeff <- min(unlist(lapply(ranef(occ_final_model)[occ_random_campaigns],FUN=min)))
  if(min_random_coeff < -0.05 & min(coef_flag) + min_random_coeff < 0){
    for(i in 1:length(occ_random_campaigns)){
      min_random_coeff <- min(unlist(lapply(ranef(occ_final_model)[occ_random_campaigns][i],FUN=min)))
      if(min_random_coeff < -0.05 & min(coef_flag) + min_random_coeff < 0){
        max_random_coeff <- max(unlist(lapply(ranef(occ_final_model)[occ_random_campaigns][i],FUN=max)))
        random_frame <- as.data.frame(ranef(occ_final_model)[occ_random_campaigns][i])
        random_table <- as.data.table(random_frame)
        random_table[,level:=row.names(random_frame)]
        negative_extreme_levels <- random_table[eval(parse(text=paste0(names(ranef(occ_final_model)[occ_random_campaigns][i]),'.group1')))==min_random_coeff,]$level
        positive_extreme_levels <- random_table[eval(parse(text=paste0(names(ranef(occ_final_model)[occ_random_campaigns][i]),'.group1')))== max_random_coeff]$level
        negative_extreme_panid <- finaldata_occ[eval(parse(text=paste0(names(ranef(occ_final_model)[occ_random_campaigns][i])))) %in% negative_extreme_levels,panid] # -13%
        positive_extreme_panid  <- finaldata_occ[eval(parse(text=paste0(names(ranef(occ_final_model)[occ_random_campaigns][i])))) %in% positive_extreme_levels,panid] # +12%
        occ_outliers_panid_segm1 <- intersect(occ_outliers_panid_negative_extreme_panid,negative_extreme_panid)
        occ_outliers_panid_segm1 <- occ_outliers_panid_segm1[1:round(0.05*(length(negative_extreme_panid)))]
        if(max_random_coeff > 0.05) {
          occ_outliers_panid_segm2 <- intersect(occ_outliers_panid_positive_extreme_panid,positive_extreme_panid)
          occ_outliers_panid_segm2 <- occ_outliers_panid_segm2[1:round(0.05*(length(positive_extreme_panid)))]
        } else {
          occ_outliers_panid_segm2 <- c()
        }
        occ_outliers_panid_segm_new <- c(occ_outliers_panid_segm1,occ_outliers_panid_segm2)
        finaldata_occ <- finaldata_occ[!finaldata_occ$panid %in% occ_outliers_panid_segm_new,]
      }
    }
    occ_final_model <- glmer(Formula_final,data=finaldata_occ,family=poisson(link="log"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
    coef_flag <- c()
    for(j in flag){
      coef_flag <- append(coef_flag,coef(summary(occ_final_model))[j,"Estimate"])
    }
    min_random_coeff <- min(unlist(lapply(ranef(occ_final_model)[occ_random_campaigns],FUN=min)))
  }
  if(min_random_coeff < -0.05 & min(coef_flag) + min_random_coeff < 0){
    for(i in 1:length(occ_random_campaigns)){
      min_random_coeff <- min(unlist(lapply(ranef(occ_final_model)[occ_random_campaigns][i],FUN=min)))
      if(min_random_coeff < -0.05 & min(coef_flag) + min_random_coeff < 0){
        max_random_coeff <- max(unlist(lapply(ranef(occ_final_model)[occ_random_campaigns][i],FUN=max)))
        random_frame <- as.data.frame(ranef(occ_final_model)[occ_random_campaigns][i])
        random_table <- as.data.table(random_frame)
        random_table[,level:=row.names(random_frame)]
        negative_extreme_levels <- random_table[eval(parse(text=paste0(names(ranef(occ_final_model)[occ_random_campaigns][i]),'.group1')))==min_random_coeff,]$level
        positive_extreme_levels <- random_table[eval(parse(text=paste0(names(ranef(occ_final_model)[occ_random_campaigns][i]),'.group1')))== max_random_coeff]$level
        negative_extreme_panid <- finaldata_occ[eval(parse(text=paste0(names(ranef(occ_final_model)[occ_random_campaigns][i])))) %in% negative_extreme_levels,panid] # -13%
        positive_extreme_panid  <- finaldata_occ[eval(parse(text=paste0(names(ranef(occ_final_model)[occ_random_campaigns][i])))) %in% positive_extreme_levels,panid] # +12%
        occ_outliers_panid_segm1 <- intersect(occ_outliers_panid_negative_extreme_panid,negative_extreme_panid)
        occ_outliers_panid_segm1 <- occ_outliers_panid_segm1[1:round(0.10*(length(negative_extreme_panid)))]
        if(max_random_coeff > 0.05) {
          occ_outliers_panid_segm2 <- intersect(occ_outliers_panid_positive_extreme_panid,positive_extreme_panid)
          occ_outliers_panid_segm2 <- occ_outliers_panid_segm2[1:round(0.10*(length(positive_extreme_panid)))]
        } else {
          occ_outliers_panid_segm2 <- c()
        }
        occ_outliers_panid_segm_new <- c(occ_outliers_panid_segm1,occ_outliers_panid_segm2)
        finaldata_occ <- finaldata_occ[!finaldata_occ$panid %in% occ_outliers_panid_segm_new,]
      }
    }
    occ_final_model <- glmer(Formula_final,data=finaldata_occ,family=Gamma(link="log"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
    coef_flag <- c()
    for(j in flag){
      coef_flag <- append(coef_flag,coef(summary(occ_final_model))[j,"Estimate"])
    }
    min_random_coeff <- min(unlist(lapply(ranef(occ_final_model)[occ_random_campaigns],FUN=min)))
  }
}

coef_flag <- c()
for(j in flag){
  coef_flag <- append(coef_flag,coef(summary(occ_final_model))[j,"Estimate"])
}

if(length(occ_random_campaigns)>0){
  min_random_coeff <- min(unlist(lapply(ranef(occ_final_model)[occ_random_campaigns],FUN=min)))
  max_random_coeff <- max(unlist(lapply(ranef(occ_final_model)[occ_random_campaigns],FUN=max)))
} else {
  min_random_coeff <- 0
}

# drop outliers in case of high exposure estimation (in case we want a lower group1 coef to meet business requirements)
# segment_panid <- occ_reduction[,panid]
# outliers_panid <- intersect(occ_total_outliers_panid,segment_panid)
# occ_outliers_panid_to_remove_threshold <- outliers_panid[c(1:round(0.10*(length(outliers_panid))))]
# finaldata_occ <- finaldata_occ[!finaldata_occ$panid %in% occ_outliers_panid_to_remove_threshold,]
# occ_final_model <- glmer(Formula_final,data=finaldata_occ,family=poisson(link="log"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))

if(min(coef_flag) + min_random_coeff < 0){
  cat("removing outliers ...\nEstimated Iteration Execution ");print(occ.mixed.model.execution.time)
  cat("\nIteration 1 ... Start Time: ");print(Sys.time())
  occ_outliers_panid_to_remove <- occ_outliers_panid_segm[1:round(0.025*(nrow(data_NB_NE_B)))]
  finaldata_occ <- finaldata_occ[!finaldata_occ$panid %in% occ_outliers_panid_to_remove,]
  occ_final_model <- glmer(Formula_final,data=finaldata_occ,family=poisson(link="log"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
  coef_flag <- c()
  for(j in flag){
    coef_flag <- append(coef_flag,coef(summary(occ_final_model))[j,"Estimate"])
  }
  min_random_coeff <- min(unlist(lapply(ranef(occ_final_model)[occ_random_campaigns],FUN=min)))
  if(min(coef_flag) + min_random_coeff < 0){
    occ_outliers_panid_to_remove <- occ_outliers_panid_segm[1:round(0.05*(nrow(data_NB_NE_B)))]  
    finaldata_occ <- finaldata_occ[!finaldata_occ$panid %in% occ_outliers_panid_to_remove,]
    cat("\nIteration 2 ... Start Time: ");print(Sys.time())
    occ_final_model <- glmer(Formula_final,data=finaldata_occ,family=poisson(link="log"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
    coef_flag <- c()
    for(j in flag){
      coef_flag <- append(coef_flag,coef(summary(occ_final_model))[j,"Estimate"])
    }
    min_random_coeff <- min(unlist(lapply(ranef(occ_final_model)[occ_random_campaigns],FUN=min)))
    if(min(coef_flag) + min_random_coeff < 0){
      occ_outliers_panid_to_remove <- occ_outliers_panid_segm[1:round(0.10*(nrow(data_NB_NE_B)))]  
      finaldata_occ <- finaldata_occ[!finaldata_occ$panid %in% occ_outliers_panid_to_remove,]
      cat("\nIteration 3 ... Start Time: ");print(Sys.time())
      occ_final_model <- glmer(Formula_final,data=finaldata_occ,family=poisson(link="log"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
      coef_flag <- c()
      for(j in flag){
        coef_flag <- append(coef_flag,coef(summary(occ_final_model))[j,"Estimate"])
      }
      min_random_coeff <- min(unlist(lapply(ranef(occ_final_model)[occ_random_campaigns],FUN=min)))
      if(min(coef_flag) + min_random_coeff < 0){
        occ_outliers_panid_to_remove <- occ_outliers_panid_segm[1:round(0.15*(nrow(data_NB_NE_B)))]  
        finaldata_occ <- finaldata_occ[!finaldata_occ$panid %in% occ_outliers_panid_to_remove,]
        cat("\nFinal Iteration ... Start Time: ");print(Sys.time())
        occ_final_model <- glmer(Formula_final,data=finaldata_occ,family=poisson(link="log"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
      }
    }
  }
}

occ_dolocc_outliers_panid_to_remove <- unique(c(dolocc_outliers_panid_to_remove,occ_outliers_panid_to_remove,occ_outliers_panid_segm_new,occ_outliers_panid_to_remove_threshold))

write.csv(as.data.frame(summary(occ_final_model)$coefficients),file=paste0(initial_Occ_modelling_output,"Occ_fixed_effects.csv"))

# create output for random effects
randoms <- as.data.frame(augment.ranef.mer(ranef(occ_final_model,condVar=TRUE)))
random.ef_temp.report <- randoms[,!names(randoms) %in% c("qq")]
write.csv(random.ef_temp.report,file=paste0(initial_Occ_modelling_output,"Occ_random_effects_pvalue.csv"),row.names=FALSE)

for(i in random_demos){
  random.ef_temp <- as.data.frame(ranef(occ_final_model)[i])
  random.coef_temp <- as.data.table(random.ef_temp,keep.rownames=TRUE)
  random.coef <- random.coef_temp[,class:=i]
  names(random.coef) <- c("","group0","group1","class")
  options(warn=-1);write.table(random.coef,file=paste0(initial_Occ_modelling_output,'Occ_random_demographics_effects.csv'), quote = FALSE,append    = i > 1, 
                               sep= ",", row.names = FALSE,col.names=ifelse(i %in% random_demos[1], TRUE, FALSE));options(warn=0)
}

for(i in occ_random_campaigns){
  random.ef_temp <- as.data.frame(ranef(occ_final_model)[i])
  random.coef_temp <- as.data.table(random.ef_temp[2],keep.rownames=TRUE)
  random.coef <- random.coef_temp[,label:=names(random.coef_temp)[2]]
  write.table(random.coef,file=paste0(initial_Occ_modelling_output,'Occ_random_effects_coeffs.csv'), quote = FALSE,append    = i > 1, 
              sep= ",", row.names = FALSE,col.names=FALSE)
}

for(i in random_demos){
  random.ef_temp <- as.data.frame(se.ranef(occ_final_model)[i])
  options(warn=-1);write.table(random.ef_temp,file=paste0(initial_Occ_modelling_output,'Occ_SE_random_demographics_effects.csv'), quote = FALSE,append    = i > 1, 
                               sep= ",", row.names = FALSE);options(warn=0)
}

for(i in occ_random_campaigns){
  random.ef_temp <- as.data.frame(se.ranef(occ_final_model)[i])
  colnames(random.ef_temp)[2] <- paste0(i,' S.E.')
  options(warn=-1);write.table(random.ef_temp[2],file=paste0(initial_Occ_modelling_output,'Occ_SE_random_campaigns_effects.csv'), quote = FALSE,append    = i > 1, 
                               sep= ",", row.names = FALSE);options(warn=0)
}

random_groupe <- c(random_demos,occ_random_campaigns)

if(length(random_demos)>0){
  random_group_demos <- data.frame(random_demos)
  names.random_group_demos <- c("class")
  setnames(random_group_demos,names.random_group_demos)
  random_group_demos$exposed <- 0
} else {
  random_group_demos <- c()
}

random_group_campaigns <- data.frame(occ_random_campaigns)
names.random_group_campaigns <- c("class")
setnames(random_group_campaigns,names.random_group_campaigns)
random_group_campaigns$exposed <- 1

random_group_exposed <- rbind( random_group_campaigns,random_group_demos)

random_group_data <- data.frame()
for(i in random_groupe){
  random.ef_temp <- as.data.frame(ranef(occ_final_model)[i])
  random.coef_temp <- as.data.table(random.ef_temp,keep.rownames=TRUE)
  random.coef <- random.coef_temp[,class:=i]
  names.random.coef <- c("level","group0","group1","class")
  setnames(random.coef ,names.random.coef)
  random.coef$row <- row.names(random.coef)
  random_group_data <-  rbind( random.coef,random_group_data)
}

random_group_SEdata <- data.frame()
for(i in random_groupe){
  random.sef_temp <- as.data.frame(se.ranef(occ_final_model)[i])
  random.secoef_temp <- as.data.table(random.sef_temp,keep.rownames=TRUE)
  random.secoef <- random.secoef_temp[,class:=i]
  names.random.secoef <- c("levelSE","SE_group0","SE_group1","class")
  setnames(random.secoef ,names.random.secoef)
  random.secoef$row <- row.names(random.secoef)
  random_group_SEdata <-  rbind( random.secoef,random_group_SEdata)
}

final_random <- merge(random_group_data,random_group_SEdata, by=c("class","row"))
final_random <- merge(final_random,random_group_exposed, by=c("class"))
View(final_random)
write.csv(final_random,file=paste0(initial_Occ_modelling_output,'Occ_random_effects.csv'), row.names = FALSE)

# create a txt file with the summary of the glmer model 
logsavefname <- "Occasion_Initial_Results.txt"
sink(paste0(initial_Occ_modelling_output,logsavefname))
print(Sys.time()-occ.start.time)
print(summary(occ_final_model))
print(ranef(occ_final_model))
sink()

for(i in c(occ_random_campaigns,random_demos)){
  pdf(paste0(initial_Occ_modelling_output,"Occ_Caterpillar_Plots ",i,".pdf"),width=7,height=5)
  print(ggCaterpillar(ranef(occ_final_model, condVar=TRUE)[i]))
  dev.off()
}

# save Occasion final dataset and final model
Occ_Random_Formula_final <- Formula_final
rm(finaldata_occ_num,temp_cor,cordata)

# save data in .Rdata file
save(finaldata_occ,Occ_Random_Formula_final,occ_final_model,occ_outliers_panid_to_remove,file=paste0(output,"saved_data/","Occasion_initial_model_data.Rdata"))

Occasion.Execution.Time <- Sys.time()-occ.start.time
cat(paste0("Occasion.Execution.Time: ",Occasion.Execution.Time))
