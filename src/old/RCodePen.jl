#######################################
#-------------PENETRATION-------------#
#######################################

pen.start.time <- Sys.time()

#set response variable
y_var <- "Buyer_Pos_P1"

# drop variables with only one level
finaldata_pen <- as.data.table(initial_data)
indices <- sapply(finaldata_pen,function(x) length(unique(x))==1)
finaldata_pen <- finaldata_pen[,!indices,with=F]

#finaldata_pen <- finaldata_pen[sample(nrow(finaldata_pen),0.50*nrow(finaldata_pen)),]

# set mandatory variables
pen_mandatory_covariate <- c("Buyer_Pre_P1")

# final data for the modeling process
finaldata_pen <- as.data.frame(finaldata_pen)
finaldata_pen <- finaldata_pen[, !(names(finaldata_pen) %in% c("Trps_POS_P1","Dol_per_Trip_POS_P1","Trps_PRE_P1","Dol_per_Trip_PRE_P1","Nonbuyer_Pre_P1"))]
finaldata_pen <- finaldata_pen[!finaldata_pen$panid %in% occ_dolocc_outliers_panid_to_remove,]

# set initial model formula
Formula_fixed <- paste0(paste0(y_var,"~ ",paste0(names(finaldata_pen[,!names(finaldata_pen) %in% c(y_var,"panid",random_demos,random_campaigns,"Buyer_Pre_P1",scoring_vars)]),collapse=" + ")," "),"+ offset(log(Buyer_Pre_P1+1))")
options(warn=-1)

# run glm model
model_glm_total <- glm(Formula_fixed,data=finaldata_pen,family=binomial(link="logit"))
summary(model_glm_total)

##### Variable selection - Condition 1 #####
# Remove pvalues greater than 0.7 and rerun the glm model
pval_excluded_vars <- setdiff(c(rownames(as.data.frame(which(summary(model_glm_total)$coefficient[,4]>0.7))),setdiff(names(model_glm_total$coefficients),rownames(summary(model_glm_total)$coefficients))),pen_mandatory_covariate)
finaldata_pen <- finaldata_pen[, !names(finaldata_pen) %in% c(pval_excluded_vars)] 

# re-run glm model
Formula_fixed <- paste0(paste0(y_var,"~ ",paste0(names(finaldata_pen[,!names(finaldata_pen)%in%c(y_var,"panid",random_demos,random_campaigns,"Buyer_Pre_P1",scoring_vars)]),collapse=" + ")," "),"+ offset(log(Buyer_Pre_P1+1))")
model_glm_condition1 <- glm(Formula_fixed,data=finaldata_pen,family=binomial(link="logit"))
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
  finaldata_pen <- finaldata_pen[ , !names(finaldata_pen) %in% multicols_toremove] 
  Formula_fixed <- paste0(paste0(y_var,"~ ",paste0(names(finaldata_pen[,!names(finaldata_pen)%in%c(y_var,"panid",random_demos,random_campaigns,"Buyer_Pre_P1")]),collapse=" + ")," "),"+ offset(log(Buyer_Pre_P1+1))")
  model_glm_new <- glm(Formula_fixed,data=finaldata_pen,family=binomial(link="logit"))  
} else {
  model_glm_new <- model_glm_condition1
}

vif_output <- as.data.frame(vif(model_glm_new))
vif_to_remove <- rownames(vif_output)[(vif(model_glm_new)) > 15]

# final vars to exclude under condition 2
vars_condition2 <- setdiff(intersect(z_excluded_vars,vif_to_remove),pen_mandatory_covariate)

# re-run glm model
finaldata_pen <- finaldata_pen[ , !names(finaldata_pen) %in% c(vars_condition2,pval_excluded_vars)] 
Formula_fixed <- paste0(paste0(y_var,"~ ",paste0(names(finaldata_pen[,!names(finaldata_pen)%in%c(y_var,"panid",random_demos,random_campaigns,"Buyer_Pre_P1",scoring_vars)]),collapse=" + ")," ")," + offset(log(Buyer_Pre_P1+1))")
model_glm <- glm(Formula_fixed,data=finaldata_pen,family=binomial(link="logit"))
summary(model_glm)
rm(model_glm_condition1)

# p-values and signs checks
pvalues <- summary(model_glm)$coefficients[,4][-1]
keep_covariate <- names(pvalues[pvalues<pvalue_lvl])

# final cleaning of first model based on signs and pvalues
neutralvars <- setdiff(names(finaldata_pen),c(positivevars,negativevars))
coeffs <- sign(summary(model_glm)$coefficients[,1])
varstokeep <- c();initialvars <- c()
for (i in negativevars) if (i %in% names(coeffs[coeffs<0])) varstokeep <- c(varstokeep,i)
for (j in positivevars) if (j %in% names(coeffs[coeffs>0])) varstokeep <- c(varstokeep,j)
for (l in neutralvars) varstokeep <- c(varstokeep,l)
for (k in varstokeep) if (k %in% keep_covariate) initialvars <- c(initialvars,k)

#set the formula for the second model after cleaning the first model
Formula_final <- paste0(paste0(y_var,"~ ",paste0(initialvars,collapse=" + ")," "),"+ group + offset(log(Buyer_Pre_P1+1))")
model_glm_2 <- glm(Formula_final,data=finaldata_pen,family=binomial(link="logit"))
summary(model_glm_2)
rm(model_glm)

# cleaning of second model based on signs and pvalues
pvalues_2 <- summary(model_glm_2)$coefficients[,4][-1]
keep_covariate_2 <- names(pvalues_2[pvalues_2<pvalue_lvl])
coeffs_2 <- sign(summary(model_glm_2)$coefficients[,1])
varstokeep_2 <- c();vars_2 <- c()
for (i in negativevars) if (i %in% names(coeffs_2[coeffs_2<0])) varstokeep_2 <- c(varstokeep_2,i)
for (j in positivevars) if (j %in% names(coeffs_2[coeffs_2>0])) varstokeep_2 <- c(varstokeep_2,j)
for (l in neutralvars) varstokeep_2 <- c(varstokeep_2,l)
for (k in varstokeep_2) if (k %in% keep_covariate_2) vars_2 <- c(vars_2,k)

#######################################
#------------ Final Model ------------#
#######################################

#set the formula for the final model
Formula_final <- paste0(paste0(y_var,"~ ",paste0(vars_2,collapse=" + ")," "),"+ group + offset(log(Buyer_Pre_P1+1))")
pen_final_model <- glm(Formula_final,data=finaldata_pen,family=binomial(link="logit"))
summary(pen_final_model)
rm(model_glm_2)

# final model
# final cleaning of model based on signs
pvalues_2 <- summary(pen_final_model)$coefficients[,4][-1]
keep_covariate_2 <- names(pvalues_2[pvalues_2<pvalue_lvl])
coeffs <- sign(summary(pen_final_model)$coefficients[,1])
finalvars1 <- c();finalvars <- c()
for (i in negativevars) if (i %in% names(coeffs[coeffs<0])) finalvars1 <- c(finalvars1,i)
for (j in positivevars) if (j %in% names(coeffs[coeffs>0])) finalvars1 <- c(finalvars1,j)
for (l in neutralvars) if (l %in% keep_covariate_2) finalvars1 <- c(finalvars1,l)
for (k in finalvars1) if (k %in% keep_covariate_2) finalvars <- c(finalvars,k)

# final correlation check (find pair vars with correlation greater then 0.8 and drop the one with bigger pvalue)
finaldata_pen_num <- as.data.table(finaldata_pen[,names(finaldata_pen) %in% finalvars])
t1 <- as.data.frame(finaldata_pen_num[,sapply(finaldata_pen_num,is.numeric)])
finaldata_pen_num <- as.data.frame(finaldata_pen_num)
finaldata_pen_num <- finaldata_pen_num[,c(which(t1==T))]
temp_cor <- as.matrix(cor(finaldata_pen_num))
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
      if (summary(pen_final_model)$coefficients[which(rownames(summary(pen_final_model)$coefficients)==as.character(cordata[i][,1,with=F])),4] > summary(pen_final_model)$coefficients[which(rownames(summary(pen_final_model)$coefficients)==as.character(cordata[i][,2,with=F])),4]) {
        drop_cor_vars <- c(drop_cor_vars,as.character(cordata[i][,1,with=F]))
      } else {
        drop_cor_vars <- c(drop_cor_vars,as.character(cordata[i][,2,with=F]))
      }
    }
  }
  finalvars <- setdiff(finalvars,drop_cor_vars)
}

#set the final formula
Formula_final <- paste0(paste0(y_var,"~ ",paste0(finalvars,collapse=" + ")," ")," + group  + offset(log(Buyer_Pre_P1+1))")
pen_final_model <- glm(Formula_final,data=finaldata_pen,family=binomial(link="logit"))
summary(pen_final_model)

options(warn=0)

# # kruskal wallis test for pair comparison of publisher's levels only for penetration
# # Pen model
# finaldata_pen <- as.data.table(finaldata_pen)
# finaldata_pen[,Prediction:=predict(pen_final_model, type="response")]
# finaldata_pen <- droplevels(finaldata_pen)
# 
# kruskal_var <- 'Prediction'
# kruskal_randoms <- c()
# for(i in random_campaigns){
#   kt <- kruskal.test(eval(parse(text=kruskal_var))~eval(parse(text=i)),data=finaldata_pen)
#   print(i);print(kt)
#   if(kt$p.value<0.10){
#     kruskal_randoms <- append(kruskal_randoms,i)
#   }
# }
#
# warning('In case of aggregation based on Kruskal-Wallis test, make sure that the other level of each sub-campaign is named as Other_XXX')
# for(i in kruskal_randoms){
#   lev <- setdiff(levels(finaldata_pen[,eval(parse(text=i))]),'none')
#   if(length(lev)>2){
#     aggr <- c()
#     for(j in 1:(length(lev)-1)){
#       for( k in (j+1):(length(lev))){       
#         df <- subset(finaldata_pen, eval(parse(text=i))  %in% c(lev[j],lev[k]))
#         l1 <- list(c(lev[j],lev[k]),P_Value=try(round(kruskal.test(df[,eval(parse(text=kruskal_var))]~df[,eval(parse(text=i))])$p.value,5),silent=TRUE))
#         if(l1[2]=="NaN") next
#         vectorElements <- unlist(l1)
#         listnames <- names(vectorElements) <- c(paste0(i,':'),paste0(i,':'),'P_value:')
#         listout <- paste(listnames,vectorElements)
#         write.table(listout,file=paste0(final_Occ_modelling_output,'Pen_Kruskal.csv'), row.names = FALSE, col.names = FALSE, quote = FALSE,append=TRUE)  
#         if(l1[['P_Value']]<0.05){
#           aggr <- append(aggr,c(sapply(l1[1],`[`,1),sapply(l1[1],`[`,2)))
#           aggr <- setdiff(aggr,c('none',paste0("Other_",i)))
#         }
#       }  
#     }
#     if(length(unique(aggr))>0){
#       finaldata_pen[!eval(parse(text=i)) %in% c(aggr,'none'),eval(parse(text=i)):=paste0("Other_",i)]
#     } else {
#       finaldata_pen[,eval(parse(text=i)):=NULL]
#     }
#   }
# }

finaldata_pen <- as.data.table(finaldata_pen)
finaldata_pen <- droplevels(finaldata_pen)
pen_random_campaigns <- intersect(random_campaigns,names(finaldata_pen))

if(length(pen_random_campaigns)>0){
  pen_random_campaigns_factors <- paste0("+(0+group|",pen_random_campaigns,")")
} else {
  pen_random_campaigns_factors <- NULL
}
random_demos_factors <- paste0( "+(0+group|",random_demos,")" )

##################################################################################################################

# run glmer model
pen.mixed.start.time <- Sys.time()
Formula_final <- paste0(c(paste0(paste0(y_var,"~ ",paste0(finalvars,collapse=" + ")," "),"+ offset(log(Buyer_Pre_P1+1)) + group"),pen_random_campaigns_factors),collapse='')
pen_final_model <- glmer(Formula_final,data=finaldata_pen,family=binomial(link="logit"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
pen.mixed.model.execution.time <- Sys.time()-pen.mixed.start.time
out_resid <- order(residuals(pen_final_model),decreasing = T)
finaldata_pen <- as.data.table(finaldata_pen)
pen_total_outliers_panid <- finaldata_pen[out_resid,panid]
segment1_panid <- data_NB_NE_B[,panid]
#segment2_panid <- data_B_E_NB[,panid]
outliers1_panid <- intersect(pen_total_outliers_panid,segment1_panid)
outliers2_panid <- c()
#outliers2_panid <- intersect(pen_total_outliers_panid,segment2_panid)
pen_outliers_panid_segm <- c(outliers1_panid,outliers2_panid)

# create a txt file with the summary of the first glmer model 
logsavefname <- "Pen_First_Run_Results.txt"
sink(paste0(initial_Pen_modelling_output,logsavefname))
print(Sys.time()-pen.start.time)
print(summary(pen_final_model)$coef)
print(ranef(pen_final_model))
sink()

# remove outliers if needed (negative estimation for exposure)
pen_outliers_panid_to_remove <- c()
pen_outliers_panid_segm_new <- c()
pen_outliers_panid_to_remove_threshold <- c()

flag <- c()
coef_flag <- c()
for(i in setdiff(levels(finaldata_pen$group),"0")) flag <- append(flag,paste0('group',i))

for(j in flag){
  coef_flag <- append(coef_flag,coef(summary(pen_final_model))[j,"Estimate"])
}

if(length(pen_random_campaigns)>0){
  min_random_coeff <- min(unlist(lapply(ranef(pen_final_model)[pen_random_campaigns],FUN=min)))
  max_random_coeff <- max(unlist(lapply(ranef(pen_final_model)[pen_random_campaigns],FUN=max)))
} else {
  min_random_coeff <- 0
}

if(min_random_coeff < -0.05 & min(coef_flag) + min_random_coeff < 0){
  out_resid_positive <- order(residuals(pen_final_model),decreasing = T)
  out_resid_negative <- order(residuals(pen_final_model),decreasing = F)
  pen_outliers_panid_positive_extreme_panid <- finaldata_pen[out_resid_positive,panid]
  pen_outliers_panid_negative_extreme_panid <- finaldata_pen[out_resid_negative,panid]
  for(i in 1:length(pen_random_campaigns)){
    min_random_coeff <- min(unlist(lapply(ranef(pen_final_model)[pen_random_campaigns][i],FUN=min)))
    if(min_random_coeff < -0.05 & min(coef_flag) + min_random_coeff < 0){
      max_random_coeff <- max(unlist(lapply(ranef(pen_final_model)[pen_random_campaigns][i],FUN=max)))
      random_frame <- as.data.frame(ranef(pen_final_model)[pen_random_campaigns][i])
      random_table <- as.data.table(random_frame)
      random_table[,level:=row.names(random_frame)]
      negative_extreme_levels <- random_table[eval(parse(text=paste0(names(ranef(pen_final_model)[pen_random_campaigns][i]),'.group1')))==min_random_coeff,]$level
      positive_extreme_levels <- random_table[eval(parse(text=paste0(names(ranef(pen_final_model)[pen_random_campaigns][i]),'.group1')))== max_random_coeff]$level
      negative_extreme_panid <- finaldata_pen[eval(parse(text=paste0(names(ranef(pen_final_model)[pen_random_campaigns][i])))) %in% negative_extreme_levels,panid] # -13%
      positive_extreme_panid  <- finaldata_pen[eval(parse(text=paste0(names(ranef(pen_final_model)[pen_random_campaigns][i])))) %in% positive_extreme_levels,panid] # +12%
      pen_outliers_panid_segm1 <- intersect(pen_outliers_panid_negative_extreme_panid,negative_extreme_panid)
      pen_outliers_panid_segm1 <- pen_outliers_panid_segm1[1:round(0.02*(length(negative_extreme_panid)))]
      if(max_random_coeff > 0.05) {
        pen_outliers_panid_segm2 <- intersect(pen_outliers_panid_positive_extreme_panid,positive_extreme_panid)
        pen_outliers_panid_segm2 <- pen_outliers_panid_segm2[1:round(0.02*(length(positive_extreme_panid)))]
      } else {
        pen_outliers_panid_segm2 <- c()
      }
      pen_outliers_panid_segm_new <- c(pen_outliers_panid_segm1,pen_outliers_panid_segm2)
      finaldata_pen <- finaldata_pen[!finaldata_pen$panid %in% pen_outliers_panid_segm_new,]
    }
  }
  pen_final_model <- glmer(Formula_final,data=finaldata_pen,family=binomial(link="logit"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
  coef_flag <- c()
  for(j in flag){
    coef_flag <- append(coef_flag,coef(summary(pen_final_model))[j,"Estimate"])
  }
  min_random_coeff <- min(unlist(lapply(ranef(pen_final_model)[pen_random_campaigns],FUN=min)))
  if(min_random_coeff < -0.05 & min(coef_flag) + min_random_coeff < 0){
    for(i in 1:length(pen_random_campaigns)){
      min_random_coeff <- min(unlist(lapply(ranef(pen_final_model)[pen_random_campaigns][i],FUN=min)))
      if(min_random_coeff < -0.05 & min(coef_flag) + min_random_coeff < 0){
        max_random_coeff <- max(unlist(lapply(ranef(pen_final_model)[pen_random_campaigns][i],FUN=max)))
        random_frame <- as.data.frame(ranef(pen_final_model)[pen_random_campaigns][i])
        random_table <- as.data.table(random_frame)
        random_table[,level:=row.names(random_frame)]
        negative_extreme_levels <- random_table[eval(parse(text=paste0(names(ranef(pen_final_model)[pen_random_campaigns][i]),'.group1')))==min_random_coeff,]$level
        positive_extreme_levels <- random_table[eval(parse(text=paste0(names(ranef(pen_final_model)[pen_random_campaigns][i]),'.group1')))== max_random_coeff]$level
        negative_extreme_panid <- finaldata_pen[eval(parse(text=paste0(names(ranef(pen_final_model)[pen_random_campaigns][i])))) %in% negative_extreme_levels,panid] # -13%
        positive_extreme_panid  <- finaldata_pen[eval(parse(text=paste0(names(ranef(pen_final_model)[pen_random_campaigns][i])))) %in% positive_extreme_levels,panid] # +12%
        pen_outliers_panid_segm1 <- intersect(pen_outliers_panid_negative_extreme_panid,negative_extreme_panid)
        pen_outliers_panid_segm1 <- pen_outliers_panid_segm1[1:round(0.05*(length(negative_extreme_panid)))]
        if(max_random_coeff > 0.05) {
          pen_outliers_panid_segm2 <- intersect(pen_outliers_panid_positive_extreme_panid,positive_extreme_panid)
          pen_outliers_panid_segm2 <- pen_outliers_panid_segm2[1:round(0.05*(length(positive_extreme_panid)))]
        } else {
          pen_outliers_panid_segm2 <- c()
        }
        pen_outliers_panid_segm_new <- c(pen_outliers_panid_segm1,pen_outliers_panid_segm2)
        finaldata_pen <- finaldata_pen[!finaldata_pen$panid %in% pen_outliers_panid_segm_new,]
      }
    }
    pen_final_model <- glmer(Formula_final,data=finaldata_pen,family=binomial(link="logit"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
    coef_flag <- c()
    for(j in flag){
      coef_flag <- append(coef_flag,coef(summary(pen_final_model))[j,"Estimate"])
    }
    min_random_coeff <- min(unlist(lapply(ranef(pen_final_model)[pen_random_campaigns],FUN=min)))
  }
  if(min_random_coeff < -0.05 & min(coef_flag) + min_random_coeff < 0){
    for(i in 1:length(pen_random_campaigns)){
      min_random_coeff <- min(unlist(lapply(ranef(pen_final_model)[pen_random_campaigns][i],FUN=min)))
      if(min_random_coeff < -0.05 & min(coef_flag) + min_random_coeff < 0){
        max_random_coeff <- max(unlist(lapply(ranef(pen_final_model)[pen_random_campaigns][i],FUN=max)))
        random_frame <- as.data.frame(ranef(pen_final_model[pen_random_campaigns])[i])
        random_table <- as.data.table(random_frame)
        random_table[,level:=row.names(random_frame)]
        negative_extreme_levels <- random_table[eval(parse(text=paste0(names(ranef(pen_final_model)[pen_random_campaigns][i]),'.group1')))==min_random_coeff,]$level
        positive_extreme_levels <- random_table[eval(parse(text=paste0(names(ranef(pen_final_model)[pen_random_campaigns][i]),'.group1')))== max_random_coeff]$level
        negative_extreme_panid <- finaldata_pen[eval(parse(text=paste0(names(ranef(pen_final_model)[pen_random_campaigns][i])))) %in% negative_extreme_levels,panid] # -13%
        positive_extreme_panid  <- finaldata_pen[eval(parse(text=paste0(names(ranef(pen_final_model)[pen_random_campaigns][i])))) %in% positive_extreme_levels,panid] # +12%
        pen_outliers_panid_segm1 <- intersect(pen_outliers_panid_negative_extreme_panid,negative_extreme_panid)
        pen_outliers_panid_segm1 <- pen_outliers_panid_segm1[1:round(0.10*(length(negative_extreme_panid)))]
        if(max_random_coeff > 0.05) {
          pen_outliers_panid_segm2 <- intersect(pen_outliers_panid_positive_extreme_panid,positive_extreme_panid)
          pen_outliers_panid_segm2 <- pen_outliers_panid_segm2[1:round(0.10*(length(positive_extreme_panid)))]
        } else {
          pen_outliers_panid_segm2 <- c()
        }
        pen_outliers_panid_segm_new <- c(pen_outliers_panid_segm1,pen_outliers_panid_segm2)
        finaldata_pen <- finaldata_pen[!finaldata_pen$panid %in% pen_outliers_panid_segm_new,]
      }
    }
    pen_final_model <- glmer(Formula_final,data=finaldata_pen,family=binomial(link="logit"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
    coef_flag <- c()
    for(j in flag){
      coef_flag <- append(coef_flag,coef(summary(pen_final_model))[j,"Estimate"])
    }
    min_random_coeff <- min(unlist(lapply(ranef(pen_final_model)[pen_random_campaigns],FUN=min)))
  }
}

pvalue_flag <- c()
for(j in flag){
  pvalue_flag <- append(pvalue_flag,coef(summary(pen_final_model))[j,"Pr(>|z|)"])
}

if(length(pen_random_campaigns)>0){
  min_random_coeff <- min(unlist(lapply(ranef(pen_final_model)[pen_random_campaigns],FUN=min)))
  max_random_coeff <- max(unlist(lapply(ranef(pen_final_model)[pen_random_campaigns],FUN=max)))
} else {
  min_random_coeff <- 0
}

# drop outliers in case of high exposure estimation (in case we want a lower group1 coef to meet business requirements)
# segment_panid <- pen_reduction[,panid]
# outliers_panid <- intersect(pen_total_outliers_panid,segment_panid)
# pen_outliers_panid_to_remove_threshold <- outliers_panid[c(1:round(0.10*(length(outliers_panid))))]
# finaldata_pen <- finaldata_pen[!finaldata_pen$panid %in% pen_outliers_panid_to_remove_threshold,]
# pen_final_model <- glmer(Formula_final,data=finaldata_pen,family=binomial(link="logit"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))

if(min(coef_flag) + min_random_coeff < 0 | max(pvalue_flag)>0.40){
  cat('\n')
  cat("removing outliers ...\nEstimated Iteration Execution ");print(pen.mixed.model.execution.time)
  cat("\nIteration 1 ... Start Time: ");print(Sys.time())
  pen_outliers_panid_to_remove <- pen_outliers_panid_segm[c(1:round(0.05*(nrow(data_NB_NE_B))))]
  #pen_outliers_panid_to_remove <- pen_outliers_panid_segm[c(1:round(0.05*(nrow(data_NB_NE_B)+nrow(data_B_E_NB))))]
  finaldata_pen <- finaldata_pen[!finaldata_pen$panid %in% pen_outliers_panid_to_remove,]
  pen_final_model <- glmer(Formula_final,data=finaldata_pen,family=binomial(link="logit"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
  coef_flag <- c()
  pvalue_flag <- c()
  for(j in flag){
    coef_flag <- append(coef_flag,coef(summary(pen_final_model))[j,"Estimate"])
    pvalue_flag <- append(pvalue_flag,coef(summary(pen_final_model))[j,"Pr(>|z|)"])
  }
  min_random_coeff <- min(unlist(lapply(ranef(pen_final_model)[pen_random_campaigns],FUN=min)))
  if(min(coef_flag) + min_random_coeff < 0 | max(pvalue_flag)>0.40){
    pen_outliers_panid_to_remove <- pen_outliers_panid_segm[c(1:round(0.10*(nrow(data_NB_NE_B))))]
    #pen_outliers_panid_to_remove <- pen_outliers_panid_segm[c(1:round(0.10*(nrow(data_NB_NE_B)+nrow(data_B_E_NB))))]  
    finaldata_pen <- finaldata_pen[!finaldata_pen$panid %in% pen_outliers_panid_to_remove,]
    cat("\nIteration 2 ... Start Time: ");print(Sys.time())
    pen_final_model <- glmer(Formula_final,data=finaldata_pen,family=binomial(link="logit"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
    coef_flag <- c()
    pvalue_flag <- c()
    for(j in flag){
      coef_flag <- append(coef_flag,coef(summary(pen_final_model))[j,"Estimate"])
      pvalue_flag <- append(pvalue_flag,coef(summary(pen_final_model))[j,"Pr(>|z|)"])
    }
    min_random_coeff <- min(unlist(lapply(ranef(pen_final_model)[pen_random_campaigns],FUN=min)))
    if(min(coef_flag) + min_random_coeff < 0 | max(pvalue_flag)>0.40){
      pen_outliers_panid_to_remove <- pen_outliers_panid_segm[c(1:round(0.15*(nrow(data_NB_NE_B))))]
      #pen_outliers_panid_to_remove <- pen_outliers_panid_segm[c(1:round(0.15*(nrow(data_NB_NE_B)+nrow(data_B_E_NB))))]  
      finaldata_pen <- finaldata_pen[!finaldata_pen$panid %in% pen_outliers_panid_to_remove,]
      cat("\nIteration 3 ... Start Time: ");print(Sys.time())
      pen_final_model <- glmer(Formula_final,data=finaldata_pen,family=binomial(link="logit"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
      coef_flag <- c()
      pvalue_flag <- c()
      for(j in flag){
        coef_flag <- append(coef_flag,coef(summary(pen_final_model))[j,"Estimate"])
        pvalue_flag <- append(pvalue_flag,coef(summary(pen_final_model))[j,"Pr(>|z|)"])
      }
      min_random_coeff <- min(unlist(lapply(ranef(pen_final_model)[pen_random_campaigns],FUN=min)))
      if(min(coef_flag) + min_random_coeff < 0 | max(pvalue_flag)>0.40){
        pen_outliers_panid_to_remove <- pen_outliers_panid_segm[c(1:round(0.20*(nrow(data_NB_NE_B))))]
        #pen_outliers_panid_to_remove <- pen_outliers_panid_segm[c(1:round(0.20*(nrow(data_NB_NE_B)+nrow(data_B_E_NB))))]  
        finaldata_pen <- finaldata_pen[!finaldata_pen$panid %in% pen_outliers_panid_to_remove,]
        cat("\nFinal Iteration ... Start Time: ");print(Sys.time())
        pen_final_model <- glmer(Formula_final,data=finaldata_pen,family=binomial(link="logit"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
      }
    }
  }
}

write.csv(as.data.frame(summary(pen_final_model)$coefficients),file=paste0(final_Pen_modelling_output,"Pen_fixed_effects.csv"))
write.csv(as.data.frame(summary(pen_final_model)$coefficients),file=paste0(scoring_Pen_output,"Pen_fixed_effects.csv"))

# create output for random effects
randoms <- as.data.frame(augment.ranef.mer(ranef(pen_final_model,condVar=TRUE)))
random.ef_temp.report <- randoms[,!names(randoms) %in% c("qq")]
write.csv(random.ef_temp.report,file=paste0(final_Pen_modelling_output,"Pen_random_effects_pvalue.csv"),row.names=FALSE)

for(i in pen_random_campaigns){
  random.ef_temp <- as.data.frame(ranef(pen_final_model)[i])
  random.coef_temp <- as.data.table(random.ef_temp[2],keep.rownames=TRUE)
  random.coef <- random.coef_temp[,label:=names(random.coef_temp)[2]]
  write.table(random.coef,file=paste0(final_Pen_modelling_output,'Pen_random_effects_coeffs.csv'), quote = FALSE,append    = i > 1, 
              sep= ",", row.names = FALSE,col.names=FALSE)
}

for(i in pen_random_campaigns){
  random.ef_temp <- as.data.frame(se.ranef(pen_final_model)[i])
  colnames(random.ef_temp)[2] <- paste0(i,' S.E.')
  options(warn=-1);write.table(random.ef_temp[2],file=paste0(final_Pen_modelling_output,'Pen_SE_random_effects.csv'), quote = FALSE,append    = i > 1, 
                               sep= ",", row.names = FALSE);options(warn=0)
}

pen_random_demos <- c()
random_groupe <- c(pen_random_demos,pen_random_campaigns)

if(length(pen_random_demos)>0){
  random_group_demos <- data.frame(random_demos)
  names.random_group_demos <- c("class")
  setnames(random_group_demos,names.random_group_demos)
  random_group_demos$exposed <- 0
} else {
  random_group_demos <- c()
}

random_group_campaigns <- data.frame(pen_random_campaigns)
names.random_group_campaigns <- c("class")
setnames(random_group_campaigns,names.random_group_campaigns)
random_group_campaigns$exposed <- 1

random_group_exposed <- rbind( random_group_campaigns,random_group_demos)

random_group_data <- data.frame()
for(i in random_groupe){
  random.ef_temp <- as.data.frame(ranef(pen_final_model)[i])
  random.coef_temp <- as.data.table(random.ef_temp,keep.rownames=TRUE)
  random.coef <- random.coef_temp[,class:=i]
  names.random.coef <- c("level","group0","group1","class")
  setnames(random.coef ,names.random.coef)
  random.coef$row <- row.names(random.coef)
  random_group_data <-  rbind( random.coef,random_group_data)
}

random_group_SEdata <- data.frame()
for(i in random_groupe){
  random.sef_temp <- as.data.frame(se.ranef(pen_final_model)[i])
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
write.csv(final_random,file=paste0(scoring_Pen_output,'Pen_random_effects.csv'), row.names = FALSE)

# create a txt file with the summary of the glmer model 
logsavefname <- "Pen_Final_Results.txt"
sink(paste0(final_Pen_modelling_output,logsavefname))
print(Sys.time()-pen.start.time)
print(summary(pen_final_model))
print(ranef(pen_final_model))
sink()

for(i in pen_random_campaigns){
  pdf(paste0(final_Pen_modelling_output,"Pen_Caterpillar_Plots ",i,".pdf"),width=7,height=5)
  print(ggCaterpillar(ranef(pen_final_model, condVar=TRUE)[i]))
  dev.off()
}

# create descriptive plots for the Pen breaks based on  the predicted values
finaldata_pen <- as.data.table(finaldata_pen)
finaldata_pen[,Prediction_Pen:=predict(pen_final_model, type="response")]
finaldata_pen <- droplevels(finaldata_pen)

response_var <- c("Prediction_Pen")
for(i in random_campaigns){
  data <- as.data.table(copy(finaldata_pen))
  data <- data[eval(parse(text=i))!="none",]
  data <- droplevels(data)
  data <- as.data.frame(data)
  rr <- c(response_var,i)
  df <- data[,rr]
  res <- ldply(levels(df[,i]),function(x) round(mean(df[df[,i]==x,]$Prediction_Pen),2))
  df <-cbind(res,levels(df[,i]))
  colnames(df) <- c('Buyer_Pos_Prob','Levels')
  p <- ggplot(df,aes(x=Levels,y=Buyer_Pos_Prob))+geom_bar(stat='identity')+ theme(axis.text.x = element_text(angle=90, vjust=0.5))+ylab('Buyer_Pos_Prob')+
    theme(legend.position="none")+theme(panel.background = element_rect(fill = 'white'))+ #geom_hline(yintercept=mean(finaldata_pen$Prediction_Pen),color="red",size=0.5)+
    geom_text(stat='identity',aes(label=Buyer_Pos_Prob,group=Levels), position = position_dodge(0.9), vjust = 0)  
  ggsave(filename=paste0(output_checks,i,'_Buyer_Pos_Prob.pdf'), width=4, height=5,p)
}

# save Penetration final dataset and final model
Pen_Formula_final <- Formula_final
all_pen_outliers_panid_to_remove <- c(pen_outliers_panid_to_remove,pen_outliers_panid_segm_new,pen_outliers_panid_to_remove_threshold)
rm(finaldata_pen_num,temp_cor,cordata)

# save data in .Rdata file
save(finaldata_pen,Pen_Formula_final,pen_final_model,all_pen_outliers_panid_to_remove,file=paste0(output,"saved_data/","pen_final_model_data.Rdata"))






# --------------------------------------------------------------------------------------------------------------------------------------------------
# ---- END OF PENNETRATION --------
# --------------------------------------------------------------------------------------------------------------------------------------------------




# Check and Rerun Occasion and DolOcc models due to outliers removal if needed
if(length(c(pen_outliers_panid_segm_new,pen_outliers_panid_to_remove))>0){
  #load("Occasion_initial_model_data.Rdata")
  cat('\n')
  cat("Re-Running Occasion Model ...")
  finaldata_occ <- finaldata_occ[!finaldata_occ$panid %in% c(pen_outliers_panid_segm_new,pen_outliers_panid_to_remove),]
  
  # run glmer model
  occ_final_model <- glmer(Occ_Random_Formula_final,data=finaldata_occ,family=poisson(link="log"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
}

# save data in .Rdata file
save(finaldata_occ,occ_final_model,Occ_Random_Formula_final,file=paste0(output,"saved_data/","Occasion_final_model_data.Rdata"))

write.csv(as.data.frame(summary(occ_final_model)$coefficients),file=paste0(final_Occ_modelling_output,"Occ_fixed_effects.csv"))
write.csv(as.data.frame(summary(occ_final_model)$coefficients),file=paste0(scoring_Occ_output,"Occ_fixed_effects.csv"))

# create output for random effects
randoms <- as.data.frame(augment.ranef.mer(ranef(occ_final_model,condVar=TRUE)))
random.ef_temp.report <- randoms[,!names(randoms) %in% c("qq")]
write.csv(random.ef_temp.report,file=paste0(final_Occ_modelling_output,"Occ_random_effects_pvalue.csv"),row.names=FALSE)

for(i in random_demos){
  random.ef_temp <- as.data.frame(ranef(occ_final_model)[i])
  random.coef_temp <- as.data.table(random.ef_temp,keep.rownames=TRUE)
  random.coef <- random.coef_temp[,class:=i]
  names(random.coef) <- c("","group0","group1","class")
  options(warn=-1);write.table(random.coef,file=paste0(final_Occ_modelling_output,'Occ_random_demographics_effects.csv'), quote = FALSE,append    = i > 1, 
                               sep= ",", row.names = FALSE,col.names=ifelse(i %in% random_demos[1], TRUE, FALSE));options(warn=0)
}

for(i in occ_random_campaigns){
  random.ef_temp <- as.data.frame(ranef(occ_final_model)[i])
  random.coef_temp <- as.data.table(random.ef_temp[2],keep.rownames=TRUE)
  random.coef <- random.coef_temp[,label:=names(random.coef_temp)[2]]
  write.table(random.coef,file=paste0(final_Occ_modelling_output,'Occ_random_effects_coeffs.csv'), quote = FALSE,append    = i > 1, 
              sep= ",", row.names = FALSE,col.names=FALSE)
}

for(i in random_demos){
  random.ef_temp <- as.data.frame(se.ranef(occ_final_model)[i])
  options(warn=-1);write.table(random.ef_temp,file=paste0(final_Occ_modelling_output,'Occ_SE_random_demographics_effects.csv'), quote = FALSE,append    = i > 1, 
                               sep= ",", row.names = FALSE);options(warn=0)
}

for(i in occ_random_campaigns){
  random.ef_temp <- as.data.frame(se.ranef(occ_final_model)[i])
  colnames(random.ef_temp)[2] <- paste0(i,' S.E.')
  options(warn=-1);write.table(random.ef_temp[2],file=paste0(final_Occ_modelling_output,'Occ_SE_random_campaigns_effects.csv'), quote = FALSE,append    = i > 1, 
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
write.csv(final_random,file=paste0(scoring_Occ_output,'Occ_random_effects.csv'), row.names = FALSE)

# create a txt file with the summary of the glmer model 
logsavefname <- "Occasion_Final_Results.txt"
sink(paste0(final_Occ_modelling_output,logsavefname))
print(Sys.time()-occ.start.time)
print(summary(occ_final_model))
print(ranef(occ_final_model))
sink()

for(i in c(occ_random_campaigns,random_demos)){
  pdf(paste0(final_Occ_modelling_output,"Occ_Caterpillar_Plots ",i,".pdf"),width=7,height=5)
  print(ggCaterpillar(ranef(occ_final_model, condVar=TRUE)[i]))
  dev.off()
}

if(length(unique(c(occ_outliers_panid_to_remove,occ_outliers_panid_to_remove_threshold,occ_outliers_panid_segm_new,pen_outliers_panid_segm_new,pen_outliers_panid_to_remove)))>0){
  #load("DolOcc_initial_model_data.Rdata")
  cat('\n')
  cat("Re-Running DolOcc Model ...")
  finaldata_dolocc <- finaldata_dolocc[!finaldata_dolocc$panid %in% unique(c(occ_outliers_panid_to_remove,occ_outliers_panid_to_remove_threshold,occ_outliers_panid_segm_new,pen_outliers_panid_segm_new,pen_outliers_panid_to_remove)),]
  
  # run glmer model
  dolocc_final_model <- glmer(dolocc_Formula_final,data=finaldata_dolocc,family=Gamma(link="log"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
}

# save data in .Rdata file
save(finaldata_dolocc,dolocc_final_model,dolocc_Formula_final,final_DollOcc_modelling_output,file=paste0(output,"saved_data/","DolOcc_final_model_data.Rdata"))

write.csv(as.data.frame(summary(dolocc_final_model)$coefficients),file=paste0(final_DollOcc_modelling_output,"DolOcc_fixed_effects.csv"))
write.csv(as.data.frame(summary(dolocc_final_model)$coefficients),file=paste0(scoring_DolOcc_output,"DolOcc_fixed_effects.csv"))

# create output for random effects
randoms <- as.data.frame(augment.ranef.mer(ranef(dolocc_final_model,condVar=TRUE)))
random.ef_temp.report <- randoms[,!names(randoms) %in% c("qq")]
write.csv(random.ef_temp.report,file=paste0(final_DollOcc_modelling_output,"DolOcc_random_effects_pvalue.csv"),row.names=FALSE)

for(i in dolocc_random_campaigns){
  random.ef_temp <- as.data.frame(ranef(dolocc_final_model)[i])
  random.coef_temp <- as.data.table(random.ef_temp[2],keep.rownames=TRUE)
  random.coef <- random.coef_temp[,label:=names(random.coef_temp)[2]]
  write.table(random.coef,file=paste0(final_DollOcc_modelling_output,'DolOcc_random_effects_coeffs.csv'), quote = FALSE,append    = i > 1, 
              sep= ",", row.names = FALSE,col.names=FALSE)
}

for(i in dolocc_random_campaigns){
  random.ef_temp <- as.data.frame(se.ranef(dolocc_final_model)[i])
  colnames(random.ef_temp)[2] <- paste0(i,' S.E.')
  options(warn=-1);write.table(random.ef_temp[2],file=paste0(final_DollOcc_modelling_output,'DolOcc_SE_random_effects.csv'), quote = FALSE,append    = i > 1, 
                               sep= ",", row.names = FALSE);options(warn=0)
}

dolocc_random_demos <- c()
random_groupe <- c(dolocc_random_demos,dolocc_random_campaigns)

if(length(dolocc_random_demos)>0){
  random_group_demos <- data.frame(random_demos)
  names.random_group_demos <- c("class")
  setnames(random_group_demos,names.random_group_demos)
  random_group_demos$exposed <- 0
} else {
  random_group_demos <- c()
}

random_group_campaigns <- data.frame(dolocc_random_campaigns)
names.random_group_campaigns <- c("class")
setnames(random_group_campaigns,names.random_group_campaigns)
random_group_campaigns$exposed <- 1

random_group_exposed <- rbind( random_group_campaigns,random_group_demos)

random_group_data <- data.frame()
for(i in random_groupe){
  random.ef_temp <- as.data.frame(ranef(dolocc_final_model)[i])
  random.coef_temp <- as.data.table(random.ef_temp,keep.rownames=TRUE)
  random.coef <- random.coef_temp[,class:=i]
  names.random.coef <- c("level","group0","group1","class")
  setnames(random.coef ,names.random.coef)
  random.coef$row <- row.names(random.coef)
  random_group_data <-  rbind( random.coef,random_group_data)
}

random_group_SEdata <- data.frame()
for(i in random_groupe){
  random.sef_temp <- as.data.frame(se.ranef(dolocc_final_model)[i])
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
write.csv(final_random,file=paste0(scoring_DolOcc_output,'DolOcc_random_effects.csv'), row.names = FALSE)

# create a txt file with the summary of the glmer model 
logsavefname <- "DolOcc_Final_Results.txt"
sink(paste0(final_DollOcc_modelling_output,logsavefname))
print(Sys.time()-dolocc.start.time)
print(summary(dolocc_final_model))
print(ranef(dolocc_final_model))
sink()

for(i in dolocc_random_campaigns){
  pdf(paste0(final_DollOcc_modelling_output,"DolOcc_Caterpillar_Plots ",i,".pdf"),width=7,height=5)
  print(ggCaterpillar(ranef(dolocc_final_model, condVar=TRUE)[i]))
  dev.off()
}

total_outliers_panid_to_remove <- unique(c(panid_out_all,occ_outliers_panid_to_remove,dolocc_outliers_panid_to_remove,all_pen_outliers_panid_to_remove))

# save all modeling data in .Rdata file
save(finaldata_dolocc,finaldata_pen,finaldata_occ,total_outliers_panid_to_remove,file=paste0(output,"saved_data/","Total_Modeling_data.Rdata"))

initial_data <- initial_data[!initial_data$panid %in% total_outliers_panid_to_remove,]
write.csv(initial_data,file=paste0(scoring_output,"final_data.csv"),row.names=F)

# create a summary for the outliers
logsavefname <- "Outliers Summary.txt"
sink(paste0(output,logsavefname))
print(paste0("Modeling Execution in: ",Sys.time()-Start.Time))
print(paste0("Number of outliers HH: ",length(unique(na.omit(total_outliers_panid_to_remove)))," from ",nrow(initial_data)))
print(cat("List of removed Experian IDs: "))
print(cat(na.omit(total_outliers_panid_to_remove)))
sink()

# calculate an estimation of total lift
total_lift_estimation <- ((1+coef(summary(pen_final_model))["group1","Estimate"])*(1+coef(summary(dolocc_final_model))["group1","Estimate"])*(1+coef(summary(occ_final_model))["group1","Estimate"])-1)
cat(paste0("last lift estimation: ",100*total_lift_estimation,"%"))
