#####################################
#--------------DOL_OCC--------------#
#####################################

dolocc.start.time <- Sys.time()

#set response variable
y_var <- "Dol_per_Trip_POS_P1"

# keep only the buyers
finaldata_dolocc <- initial_data[Buyer_Pos_P1=="1",]

# drop variables with only one level
indices <- sapply(finaldata_dolocc,function(x) length(unique(x))==1)
finaldata_dolocc <- finaldata_dolocc[,!indices,with=F]

#######################################
#-------- Variable Selection ---------#
#######################################

# #kruskal wallis test for pair comparison of publisher's levels for DolOcc model only
# #DolOcc model
# # kruskal wallis test for pair comparison of publisher's levels
# kruskal_var <- 'Dol_per_Trip_POS_P1'
# kruskal_randoms <- c()
# for(i in random_campaigns){
#   kt <- kruskal.test(eval(parse(text=kruskal_var))~eval(parse(text=i)),data=finaldata_dolocc)
#   cat(i);print(kt)
#   if(kt$p.value<0.10){
#     kruskal_randoms <- append(kruskal_randoms,i)
#   } else {
#     finaldata_dolocc[,eval(parse(text=i)):=NULL]
#   }
# }
#
# warning('In case of aggregation based on Kruskal-Wallis test, make sure that the other level of each sub-campaign is named as Other_XXX')
# for(i in kruskal_randoms){
#   lev <- setdiff(levels(finaldata_dolocc[,eval(parse(text=i))]),'none')
#   if(length(lev)>2){
#     aggr <- c()
#     for(j in 1:(length(lev)-1)){
#       for( k in (j+1):(length(lev))){       
#         df <- subset(finaldata_dolocc, eval(parse(text=i))  %in% c(lev[j],lev[k]))
#         l1 <- list(c(lev[j],lev[k]),P_Value=try(round(kruskal.test(df[,eval(parse(text=kruskal_var))]~df[,eval(parse(text=i))])$p.value,5),silent=TRUE))
#         if(l1[2]=="NaN") next
#         vectorElements <- unlist(l1)
#         listnames <- names(vectorElements) <- c(paste0(i,':'),paste0(i,':'),'P_value:')
#         listout <- paste(listnames,vectorElements)
#         write.table(listout,file=paste0(output,'DolOcc_Kruskal.csv'), row.names = FALSE, col.names = FALSE, quote = FALSE,append=TRUE)  
#         if(l1[['P_Value']]<0.05){
#           aggr <- append(aggr,c(sapply(l1[1],`[`,1),sapply(l1[1],`[`,2)))
#           aggr <- setdiff(aggr,c('none',paste0("Other_",i)))
#         }
#       }  
#     }
#     if(length(unique(aggr))>0){
#       finaldata_dolocc[!eval(parse(text=i)) %in% c(aggr,'none'),eval(parse(text=i)):=paste0("Other_",i)]
#     } else {
#       finaldata_dolocc[,eval(parse(text=i)):=NULL]
#     }
#   }
# }

# finaldata_dolocc <- droplevels(finaldata_dolocc)
# dolocc_random_campaigns <- intersect(random_campaigns,names(finaldata_dolocc))
# 
# if(length(dolocc_random_campaigns)>0){
#   dolocc_random_campaigns_factors <- paste0("+(0+group|",dolocc_random_campaigns,")")
# } else {
#   dolocc_random_campaigns_factors <- NULL
# }
# random_demos_factors <- paste0( "+(0+group|",random_demos,")" )

# Create Descriptives for Campaign breaks in case of running Kruskal in DolOcc above
#dir.create(paste0(initial_DollOcc_modelling_output,"Descriptives/"))
#output_checks <- paste0(initial_DollOcc_modelling_output,"Descriptives/")
#randoms <- c(dolocc_random_campaigns)
# # create descriptive plots for the DolOcc breaks
# for(i in dolocc_random_campaigns){
#   finaldata_dolocc <- finaldata_dolocc[eval(parse(text=i))!="none",]
#   finaldata_dolocc <- droplevels(finaldata_dolocc)
#   finaldata_dolocc <- as.data.frame(finaldata_dolocc)
#   res <- ldply(levels(finaldata_dolocc[,i]),function(x) round(mean(finaldata_dolocc[finaldata_dolocc[,i]==x,]$Dol_per_Trip_POS_P1),2))
#   df <-cbind(res,levels(finaldata_dolocc[,i]))
#   colnames(df) <- c('Average_Spent_during_the_campaign_period','Levels')
#   
#   p <- ggplot(df,aes(x=Levels,y=Average_Spent_during_the_campaign_period))+geom_bar(stat='identity')+ theme(axis.text.x = element_text(angle=90, vjust=0.5))+ylab('Average Spent for the target brand during the campaign period')+
#     theme(legend.position="none")+theme(panel.background = element_rect(fill = 'white'))+
#     geom_text(stat='identity',aes(label=Average_Spent_during_the_campaign_period,group=Levels), position = position_dodge(0.9), vjust = 0)  
#   ggsave(filename=paste0(output_checks,i,'_Mean_Spent.pdf'),p,height = 8,width = 15)
#   
#   p <- ggplot(finaldata_dolocc,aes_string(i,'Dol_per_Trip_POS_P1'))+geom_boxplot()+theme(axis.text.x = element_text(angle = 90, vjust = 0.5))+
#     ylab('Spent per Trip for the target bramd during the campaign period')
#   ggsave(filename=paste0(output_checks,i,'_Spent_BoxPlot.pdf'),p,height = 8,width = 15)
# } 

dolocc_mandatory_covariate <- c("Nonbuyer_Pre_P1")

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
for (j in positivevars) if (j %in% names(coeffs[coeffs>0])) varstokeep <- c(varstokeep,j)
for (l in neutralvars) varstokeep <- c(varstokeep,l)
for (k in varstokeep) if (k %in% keep_covariate) initialvars <- c(initialvars,k)
if(dolocc_mandatory_covariate %in% names(finaldata_dolocc) & length(setdiff(dolocc_mandatory_covariate,initialvars))>0){
  initialvars <- c(dolocc_mandatory_covariate,initialvars)
}

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

# final correlation check (find pair vars with correlation greater then 0.8 and drop the one with bigger pvalue)
if(length(vars_2)>1){
  finaldata_dolocc_num <- as.data.table(finaldata_dolocc[,names(finaldata_dolocc) %in% vars_2])
  t1 <- as.data.frame(finaldata_dolocc_num[,sapply(finaldata_dolocc_num,is.numeric)])
  finaldata_dolocc_num <- as.data.frame(finaldata_dolocc_num)
  finaldata_dolocc_num <- finaldata_dolocc_num[,c(which(t1==T))]
  temp_cor <- as.matrix(cor(finaldata_dolocc_num))
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

# set the formula for the final model
Formula_final <- paste0(paste0(y_var,"~ ",paste0(vars_2,collapse=" + ")," ")," + group + offset(log(Dol_per_Trip_PRE_P1+1))")

# run model
dolocc_final_model <-  glm(Formula_final,data=finaldata_dolocc,family=Gamma(link="log"))
summary(dolocc_final_model)
rm(model_glm_2)

# final model
# final cleaning of model based on signs
pvalues_2 <- summary(dolocc_final_model)$coefficients[,4][-1]
keep_covariate_2 <- names(pvalues_2[pvalues_2 < pvalue_lvl])
coeffs <- sign(summary(dolocc_final_model)$coefficients[,1])
finalvars <- c();finalvars1 <- c()
for (i in negativevars) if (i %in% names(coeffs[coeffs<0])) finalvars1 <- c(finalvars1,i)
for (j in positivevars) if (j %in% names(coeffs[coeffs>0])) finalvars1 <- c(finalvars1,j)
for (l in neutralvars) finalvars1 <- c(finalvars1,l)
for (k in finalvars1) if (k %in% keep_covariate_2) finalvars <- c(finalvars,k)

#set the formula for the final model
Formula_final <-paste0(paste0(y_var,"~ ",paste0(finalvars,collapse=" + ")," ")," + group + offset(log(Dol_per_Trip_PRE_P1+1))")
dolocc_final_model <-  glm(Formula_final,data=finaldata_dolocc,family=Gamma(link="log"))
summary(dolocc_final_model)

options(warn=0)

dolocc_random_campaigns <- intersect(random_campaigns,names(finaldata_dolocc))

if(length(dolocc_random_campaigns)>0){
  dolocc_random_campaigns_factors <- paste0("+(0+group|",dolocc_random_campaigns,")")
} else {
  dolocc_random_campaigns_factors <- NULL
}
random_demos_factors <- paste0( "+(0+group|",random_demos,")" )

##################################################################################################################

# run glmer model
dolocc.mixed.start.time <- Sys.time()
Formula_final <- paste0(c(paste0(paste0(y_var,"~ ",paste0(finalvars,collapse=" + ")," "),"+ offset(log(Dol_per_Trip_PRE_P1+1)) + group"),dolocc_random_campaigns_factors),collapse='')
dolocc_final_model <- glmer(Formula_final,data=finaldata_dolocc,family=Gamma(link="log"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
dolocc.mixed.model.execution.time <- Sys.time()-dolocc.mixed.start.time
out_resid <- order(residuals(dolocc_final_model),decreasing = T)
finaldata_dolocc <- as.data.table(finaldata_dolocc)
dolocc_total_outliers_panid <- finaldata_dolocc[out_resid,panid]
segment_panid <- data_NB_NE_B[,panid]
dolocc_outliers_panid_segm <- intersect(dolocc_total_outliers_panid,segment_panid)

# create a txt file with the summary of the first glmer model 
logsavefname <- "DolOcc_First_Run_Results.txt"
sink(paste0(initial_DollOcc_modelling_output,logsavefname))
print(Sys.time()-dolocc.start.time)
print(summary(dolocc_final_model)$coef)
print(ranef(dolocc_final_model))
sink()

# remove outliers if needed (negative estimation for exposure)
dolocc_outliers_panid_to_remove <- c()
dolocc_outliers_panid_segm_new <- c()
dolocc_outliers_panid_to_remove_threshold <- c()

flag <- c()
coef_flag <- c()
for(i in setdiff(levels(finaldata_dolocc$group),"0")) flag <- append(flag,paste0('group',i))

for(j in flag){
  coef_flag <- append(coef_flag,coef(summary(dolocc_final_model))[j,"Estimate"])
}

if(length(dolocc_random_campaigns)>0){
  min_random_coeff <- min(unlist(lapply(ranef(dolocc_final_model)[dolocc_random_campaigns],FUN=min)))
  max_random_coeff <- max(unlist(lapply(ranef(dolocc_final_model)[dolocc_random_campaigns],FUN=max)))
} else {
  min_random_coeff <- 0
}

if(min_random_coeff < -0.05 & min(coef_flag) + min_random_coeff < 0){
  out_resid_positive <- order(residuals(dolocc_final_model),decreasing = T)
  out_resid_negative <- order(residuals(dolocc_final_model),decreasing = F)
  dolocc_outliers_panid_positive_extreme_panid <- finaldata_dolocc[out_resid_positive,panid]
  dolocc_outliers_panid_negative_extreme_panid <- finaldata_dolocc[out_resid_negative,panid]
  for(i in 1:length(dolocc_random_campaigns)){
    min_random_coeff <- min(unlist(lapply(ranef(dolocc_final_model)[dolocc_random_campaigns][i],FUN=min)))
    if(min_random_coeff < -0.05 & min(coef_flag) + min_random_coeff < 0){
      max_random_coeff <- max(unlist(lapply(ranef(dolocc_final_model)[dolocc_random_campaigns][i],FUN=max)))
      random_frame <- as.data.frame(ranef(dolocc_final_model)[dolocc_random_campaigns][i])
      random_table <- as.data.table(random_frame)
      random_table[,level:=row.names(random_frame)]
      negative_extreme_levels <- random_table[eval(parse(text=paste0(names(ranef(dolocc_final_model)[dolocc_random_campaigns][i]),'.group1')))==min_random_coeff,]$level
      positive_extreme_levels <- random_table[eval(parse(text=paste0(names(ranef(dolocc_final_model)[dolocc_random_campaigns][i]),'.group1')))== max_random_coeff]$level
      negative_extreme_panid <- finaldata_dolocc[eval(parse(text=paste0(names(ranef(dolocc_final_model)[dolocc_random_campaigns][i])))) %in% negative_extreme_levels,panid] # -13%
      positive_extreme_panid  <- finaldata_dolocc[eval(parse(text=paste0(names(ranef(dolocc_final_model)[dolocc_random_campaigns][i])))) %in% positive_extreme_levels,panid] # +12%
      dolocc_outliers_panid_segm1 <- intersect(dolocc_outliers_panid_negative_extreme_panid,negative_extreme_panid)
      dolocc_outliers_panid_segm1 <- dolocc_outliers_panid_segm1[1:round(0.02*(length(negative_extreme_panid)))]
      if(max_random_coeff > 0.05) {
        dolocc_outliers_panid_segm2 <- intersect(dolocc_outliers_panid_positive_extreme_panid,positive_extreme_panid)
        dolocc_outliers_panid_segm2 <- dolocc_outliers_panid_segm2[1:round(0.02*(length(positive_extreme_panid)))]
      } else {
        dolocc_outliers_panid_segm2 <- c()
      }
      dolocc_outliers_panid_segm_new <- c(dolocc_outliers_panid_segm1,dolocc_outliers_panid_segm2)
      finaldata_dolocc <- finaldata_dolocc[!finaldata_dolocc$panid %in% dolocc_outliers_panid_segm_new,]
    }
  }
  dolocc_final_model <- glmer(Formula_final,data=finaldata_dolocc,family=Gamma(link="log"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
  coef_flag <- c()
  for(j in flag){
    coef_flag <- append(coef_flag,coef(summary(dolocc_final_model))[j,"Estimate"])
  }
  min_random_coeff <- min(unlist(lapply(ranef(dolocc_final_model)[dolocc_random_campaigns],FUN=min)))
  if(min_random_coeff < -0.05 & min(coef_flag) + min_random_coeff < 0){
    for(i in 1:length(dolocc_random_campaigns)){
      min_random_coeff <- min(unlist(lapply(ranef(dolocc_final_model)[dolocc_random_campaigns][i],FUN=min)))
      if(min_random_coeff < -0.05 & min(coef_flag) + min_random_coeff < 0){
        max_random_coeff <- max(unlist(lapply(ranef(dolocc_final_model)[dolocc_random_campaigns][i],FUN=max)))
        random_frame <- as.data.frame(ranef(dolocc_final_model)[dolocc_random_campaigns][i])
        random_table <- as.data.table(random_frame)
        random_table[,level:=row.names(random_frame)]
        negative_extreme_levels <- random_table[eval(parse(text=paste0(names(ranef(dolocc_final_model)[dolocc_random_campaigns][i]),'.group1')))==min_random_coeff,]$level
        positive_extreme_levels <- random_table[eval(parse(text=paste0(names(ranef(dolocc_final_model)[dolocc_random_campaigns][i]),'.group1')))== max_random_coeff]$level
        negative_extreme_panid <- finaldata_dolocc[eval(parse(text=paste0(names(ranef(dolocc_final_model)[dolocc_random_campaigns][i])))) %in% negative_extreme_levels,panid] # -13%
        positive_extreme_panid  <- finaldata_dolocc[eval(parse(text=paste0(names(ranef(dolocc_final_model)[dolocc_random_campaigns][i])))) %in% positive_extreme_levels,panid] # +12%
        dolocc_outliers_panid_segm1 <- intersect(dolocc_outliers_panid_negative_extreme_panid,negative_extreme_panid)
        dolocc_outliers_panid_segm1 <- dolocc_outliers_panid_segm1[1:round(0.05*(length(negative_extreme_panid)))]
        if(max_random_coeff > 0.05) {
          dolocc_outliers_panid_segm2 <- intersect(dolocc_outliers_panid_positive_extreme_panid,positive_extreme_panid)
          dolocc_outliers_panid_segm2 <- dolocc_outliers_panid_segm2[1:round(0.05*(length(positive_extreme_panid)))]
        } else {
          dolocc_outliers_panid_segm2 <- c()
        }
        dolocc_outliers_panid_segm_new <- c(dolocc_outliers_panid_segm1,dolocc_outliers_panid_segm2)
        finaldata_dolocc <- finaldata_dolocc[!finaldata_dolocc$panid %in% dolocc_outliers_panid_segm_new,]
      }
    }
    dolocc_final_model <- glmer(Formula_final,data=finaldata_dolocc,family=Gamma(link="log"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
    coef_flag <- c()
    for(j in flag){
      coef_flag <- append(coef_flag,coef(summary(dolocc_final_model))[j,"Estimate"])
    }
    min_random_coeff <- min(unlist(lapply(ranef(dolocc_final_model)[dolocc_random_campaigns],FUN=min)))
  }
  if(min_random_coeff < -0.05 & min(coef_flag) + min_random_coeff < 0){
    for(i in 1:length(dolocc_random_campaigns)){
      min_random_coeff <- min(unlist(lapply(ranef(dolocc_final_model)[dolocc_random_campaigns][i],FUN=min)))
      if(min_random_coeff < -0.05 & min(coef_flag) + min_random_coeff < 0){
        max_random_coeff <- max(unlist(lapply(ranef(dolocc_final_model)[dolocc_random_campaigns][i],FUN=max)))
        random_frame <- as.data.frame(ranef(dolocc_final_model)[dolocc_random_campaigns][i])
        random_table <- as.data.table(random_frame)
        random_table[,level:=row.names(random_frame)]
        negative_extreme_levels <- random_table[eval(parse(text=paste0(names(ranef(dolocc_final_model)[dolocc_random_campaigns][i]),'.group1')))==min_random_coeff,]$level
        positive_extreme_levels <- random_table[eval(parse(text=paste0(names(ranef(dolocc_final_model)[dolocc_random_campaigns][i]),'.group1')))== max_random_coeff]$level
        negative_extreme_panid <- finaldata_dolocc[eval(parse(text=paste0(names(ranef(dolocc_final_model)[dolocc_random_campaigns][i])))) %in% negative_extreme_levels,panid] # -13%
        positive_extreme_panid  <- finaldata_dolocc[eval(parse(text=paste0(names(ranef(dolocc_final_model)[dolocc_random_campaigns][i])))) %in% positive_extreme_levels,panid] # +12%        
        dolocc_outliers_panid_segm1 <- intersect(dolocc_outliers_panid_negative_extreme_panid,negative_extreme_panid)
        dolocc_outliers_panid_segm1 <- dolocc_outliers_panid_segm1[1:round(0.10*(length(negative_extreme_panid)))]
        if(max_random_coeff > 0.05) {
          dolocc_outliers_panid_segm2 <- intersect(dolocc_outliers_panid_positive_extreme_panid,positive_extreme_panid)
          dolocc_outliers_panid_segm2 <- dolocc_outliers_panid_segm2[1:round(0.10*(length(positive_extreme_panid)))]
        } else {
          dolocc_outliers_panid_segm2 <- c()
        }
        dolocc_outliers_panid_segm_new <- c(dolocc_outliers_panid_segm1,dolocc_outliers_panid_segm2)
        finaldata_dolocc <- finaldata_dolocc[!finaldata_dolocc$panid %in% dolocc_outliers_panid_segm_new,]
      }
    }
    dolocc_final_model <- glmer(Formula_final,data=finaldata_dolocc,family=Gamma(link="log"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
    coef_flag <- c()
    for(j in flag){
      coef_flag <- append(coef_flag,coef(summary(dolocc_final_model))[j,"Estimate"])
    }
    min_random_coeff <- min(unlist(lapply(ranef(dolocc_final_model)[dolocc_random_campaigns],FUN=min)))
  }
}

if(length(dolocc_random_campaigns)>0){
  min_random_coeff <- min(unlist(lapply(ranef(dolocc_final_model)[dolocc_random_campaigns],FUN=min)))
  max_random_coeff <- max(unlist(lapply(ranef(dolocc_final_model)[dolocc_random_campaigns],FUN=max)))
} else {
  min_random_coeff <- 0
}

coef_flag <- c()
pvalue_flag <- c()
for(j in flag){
  coef_flag <- append(coef_flag,coef(summary(dolocc_final_model))[j,"Estimate"])
  pvalue_flag <- append(pvalue_flag,coef(summary(dolocc_final_model))[j,"Pr(>|z|)"])
}

# drop outliers in case of high exposure estimation (in case we want a lower group1 coef to meet business requirements)
# segment_panid <- dol_reduction[,panid]
# outliers_panid <- intersect(dolocc_total_outliers_panid,segment_panid)
# dolocc_outliers_panid_to_remove_threshold <- outliers_panid[c(1:round(0.10*(length(outliers_panid))))]
# finaldata_dolocc <- finaldata_dolocc[!finaldata_dolocc$panid %in% dolocc_outliers_panid_to_remove_threshold,]
# dolocc_final_model <- glmer(Formula_final,data=finaldata_dolocc,family=Gamma(link="log"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))

if(min(coef_flag) + min_random_coeff  < 0| max(pvalue_flag)>0.20){
  cat('\n')
  cat("removing outliers ...\nEstimated Iteration Execution ");print(dolocc.mixed.model.execution.time)
  cat("\nIteration 1 ... Start Time: ");print(Sys.time())
  dolocc_outliers_panid_to_remove <- dolocc_outliers_panid_segm[1:round(0.025*(nrow(data_NB_NE_B)))]
  finaldata_dolocc <- finaldata_dolocc[!finaldata_dolocc$panid %in% dolocc_outliers_panid_to_remove,]
  dolocc_final_model <- glmer(Formula_final,data=finaldata_dolocc,family=Gamma(link="log"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
  coef_flag <- c()
  pvalue_flag <- c()
  for(j in flag){
    coef_flag <- append(coef_flag,coef(summary(dolocc_final_model))[j,"Estimate"])
    pvalue_flag <- append(pvalue_flag,coef(summary(dolocc_final_model))[j,"Pr(>|z|)"])
  }
  min_random_coeff <- min(unlist(lapply(ranef(dolocc_final_model)[dolocc_random_campaigns],FUN=min)))
  if(min(coef_flag) + min_random_coeff  < 0| max(pvalue_flag)>0.20){
    dolocc_outliers_panid_to_remove <- dolocc_outliers_panid_segm[1:round(0.05*(nrow(data_NB_NE_B)))]  
    finaldata_dolocc <- finaldata_dolocc[!finaldata_dolocc$panid %in% dolocc_outliers_panid_to_remove,]
    cat("\nIteration 2 ... Start Time: ");print(Sys.time())
    dolocc_final_model <- glmer(Formula_final,data=finaldata_dolocc,family=Gamma(link="log"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
    coef_flag <- c()
    pvalue_flag <- c()
    for(j in flag){
      coef_flag <- append(coef_flag,coef(summary(dolocc_final_model))[j,"Estimate"])
      pvalue_flag <- append(pvalue_flag,coef(summary(dolocc_final_model))[j,"Pr(>|z|)"])
    }
    min_random_coeff <- min(unlist(lapply(ranef(dolocc_final_model)[dolocc_random_campaigns],FUN=min)))
    if(min(coef_flag) + min_random_coeff < 0| max(pvalue_flag)>0.20 ){
      dolocc_outliers_panid_to_remove <- dolocc_outliers_panid_segm[1:round(0.10*(nrow(data_NB_NE_B)))]  
      finaldata_dolocc <- finaldata_dolocc[!finaldata_dolocc$panid %in% dolocc_outliers_panid_to_remove,]
      cat("\nIteration 3 ... Start Time: ");print(Sys.time())
      dolocc_final_model <- glmer(Formula_final,data=finaldata_dolocc,family=Gamma(link="log"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
      coef_flag <- c()
      pvalue_flag <- c()
      for(j in flag){
        coef_flag <- append(coef_flag,coef(summary(dolocc_final_model))[j,"Estimate"])
        pvalue_flag <- append(pvalue_flag,coef(summary(dolocc_final_model))[j,"Pr(>|z|)"])
      }
      min_random_coeff <- min(unlist(lapply(ranef(dolocc_final_model)[dolocc_random_campaigns],FUN=min)))
      if(min(coef_flag) + min_random_coeff  < 0| max(pvalue_flag)>0.20){
        dolocc_outliers_panid_to_remove <- dolocc_outliers_panid_segm[1:round(0.15*(nrow(data_NB_NE_B)))]  
        finaldata_dolocc <- finaldata_dolocc[!finaldata_dolocc$panid %in% dolocc_outliers_panid_to_remove,]
        cat("\nFinal Iteration ... Start Time: ");print(Sys.time())
        dolocc_final_model <- glmer(Formula_final,data=finaldata_dolocc,family=Gamma(link="log"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
      }
    }
  }
}

write.csv(as.data.frame(summary(dolocc_final_model)$coefficients),file=paste0(initial_DollOcc_modelling_output,"DolOcc_fixed_effects.csv"))

# create output for random effects
randoms <- as.data.frame(augment.ranef.mer(ranef(dolocc_final_model,condVar=TRUE)))
random.ef_temp.report <- randoms[,!names(randoms) %in% c("qq")]
write.csv(random.ef_temp.report,file=paste0(initial_DollOcc_modelling_output,"DolOcc_random_effects_pvalue.csv"),row.names=FALSE)

for(i in dolocc_random_campaigns){
  random.ef_temp <- as.data.frame(ranef(dolocc_final_model)[i])
  random.coef_temp <- as.data.table(random.ef_temp[2],keep.rownames=TRUE)
  random.coef <- random.coef_temp[,label:=names(random.coef_temp)[2]]
  write.table(random.coef,file=paste0(initial_DollOcc_modelling_output,'DolOcc_random_effects_coeffs.csv'), quote = FALSE,append    = i > 1, 
                               sep= ",", row.names = FALSE,col.names=FALSE)
}

for(i in dolocc_random_campaigns){
  random.ef_temp <- as.data.frame(se.ranef(dolocc_final_model)[i])
  colnames(random.ef_temp)[2] <- paste0(i,' S.E.')
  options(warn=-1);write.table(random.ef_temp[2],file=paste0(initial_DollOcc_modelling_output,'DolOcc_SE_random_effects.csv'), quote = FALSE,append    = i > 1, 
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
write.csv(final_random,file=paste0(initial_DollOcc_modelling_output,'DolOcc_random_effects.csv'), row.names = FALSE)

# create a txt file with the summary of the glmer model 
logsavefname <- "DolOcc_Initial_Results.txt"
sink(paste0(initial_DollOcc_modelling_output,logsavefname))
print(Sys.time()-dolocc.start.time)
print(summary(dolocc_final_model))
print(ranef(dolocc_final_model))
sink()

for(i in dolocc_random_campaigns){
  pdf(paste0(initial_DollOcc_modelling_output,"DolOcc_Caterpillar_Plots ",i,".pdf"),width=7,height=5)
  print(ggCaterpillar(ranef(dolocc_final_model, condVar=TRUE)[i]))
  dev.off()
}

# save DolOcc final dataset and final model
dolocc_Formula_final <- Formula_final
dolocc_outliers_panid_to_remove <- c(dolocc_outliers_panid_to_remove,dolocc_outliers_panid_segm_new,dolocc_outliers_panid_to_remove_threshold)
rm(finaldata_dolocc_num,temp_cor,cordata)

# save data in .Rdata file
save(finaldata_dolocc,dolocc_Formula_final,dolocc_final_model,dolocc_outliers_panid_to_remove,file=paste0(output,"saved_data/","DolOcc_initial_model_data.Rdata"))

DolOcc.Execution.Time <- Sys.time()-dolocc.start.time
cat(paste0("DolOcc.Execution.Time: ", DolOcc.Execution.Time))
