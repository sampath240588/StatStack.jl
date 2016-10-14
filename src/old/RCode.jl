R"""
rm(list=ls())
Start.Time <- Sys.time()

#load libraries
list.of.packages <- c("data.table","bit64","optimx","lme4",'languageR','lmerTest','glmnet','lsmeans','car','RDS','DMwR','stringr','nloptr','ggplot2','reshape2','plyr','viridis','gmodels')
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages,require,character.only=TRUE)

#######################################
#------------- Load Data -------------#
#######################################

# set working directory
setwd("/media/u01/analytics/scoring/CDW5_792")

# set path of data and headers location
input <- c("/media/u01/analytics/scoring/CDW5_792/")

# set output file path
output <- c("/media/u01/analytics/scoring/CDW5_792/Output/")

# set name of initial data file
dataname <- "csv_final_cdw5_792"

# set name of headers file
headers_data <- "Headers_cdw5.csv"

# read initial data and attach headers
finaldata <- fread(paste0(input,dataname),colClasses = "as.character",header=F)
datanames <- fread(paste0(input,headers_data),header=F)
setnames(finaldata,c(datanames$V1))

#check if response and mandatory variables exist in the data
#Put also the sub campaigns as mandatory
all_mandatory_vars <- c("Buyer_Pos_P1","Buyer_Pre_P1","Trps_POS_P1","Trps_PRE_P1","Dol_per_Trip_POS_P1","Dol_per_Trip_PRE_P1","Nonbuyer_Pre_P1","hh_age","estimated_hh_income","number_of_children_in_living_Un","person_1_gender","experian_id")
print(all_mandatory_vars %in% names(finaldata))

# set exposed flag variable
exposed_flag_var <- "exposed_flag_new"

# Put the sub campaigns and the demographic variables that will be used as random effects in the models
random_demos <- c("estimated_hh_income","hh_age","number_of_children_in_living_Un","person_1_gender")
random_campaigns <- c("Publisher_Fct1","Targeting_Fct1")




# set extra exposed flag variable to be excluded (in case you have more than one in dataset) and publisher variables that are not used in the mixed models
dropvars <- c("exposed_flag")

# set True if P2 category products are competitor or False otherwise (need for sign check in the model)
P2_Competitor <- TRUE #or FALSE

# pvalue threshold
pvalue_lvl <- 0.20

# Create folders for outputs
dir.create(paste0(output,"Descriptive Statistics for Publishers and Demographics"))
dir.create(paste0(output,"Final_Modelling_Results"))
dir.create(paste0(output,"Initial_Modelling_Results"))
dir.create(paste0(output,"QCs"))
dir.create(paste0(output,"saved_data"))
dir.create(paste0(output,"Scoring"))
final_modelling_output <- paste0(output,"Final_Modelling_Results/")
initial_modelling_output <- paste0(output,"Initial_Modelling_Results/")
scoring_output <- paste0(output,"Scoring/")
dir.create(paste0(scoring_output,"Occasion/"))
dir.create(paste0(scoring_output,"DollarsOccasion/"))
dir.create(paste0(scoring_output,"Penetration/"))
dir.create(paste0(final_modelling_output,"Occasion/"))
dir.create(paste0(final_modelling_output,"DollarsOccasion/"))
dir.create(paste0(final_modelling_output,"Penetration/"))
dir.create(paste0(initial_modelling_output,"Occasion/"))
dir.create(paste0(initial_modelling_output,"DollarsOccasion/"))
dir.create(paste0(initial_modelling_output,"Penetration/"))

scoring_Occ_output <- paste0(scoring_output,"Occasion/")
scoring_DolOcc_output <- paste0(scoring_output,"DollarsOccasion/")
scoring_Pen_output <- paste0(scoring_output,"Penetration/")
final_Occ_modelling_output <- paste0(final_modelling_output,"Occasion/")
final_DollOcc_modelling_output <- paste0(final_modelling_output,"DollarsOccasion/")
final_Pen_modelling_output <- paste0(final_modelling_output,"Penetration/")
initial_Occ_modelling_output <- paste0(initial_modelling_output,"Occasion/")
initial_DollOcc_modelling_output <- paste0(initial_modelling_output,"DollarsOccasion/")
initial_Pen_modelling_output <- paste0(initial_modelling_output,"Penetration/")

#######################################
#--------- Data Manipulation ---------#
#######################################

# create group, panid columns
finaldata[,`:=`(group=eval(parse(text=exposed_flag_var)),panid=experian_id)]

# remove not needed variables
if(length(dropvars)>0){
  for(i in noquote(dropvars)){
    finaldata <- finaldata[,noquote(i):=NULL]
  }
}

# mandatory variables for scoring
scoring_vars <- c("Prd_1_Net_Pr_PRE","Prd_1_Net_Pr_POS","Buyer_Pos_P0","Buyer_Pre_P0")


# Set the ProScore variable in the dataset
ProScore <- grep("MODEL",colnames(finaldata),value = TRUE) 

# aggregate # of children for 4+
finaldata[number_of_children_in_living_Un>=4,number_of_children_in_living_Un:="4+" ]

# create factor variables
allrandoms <- c(random_demos,random_campaigns)
finaldata  <- as.data.frame(finaldata)
for(i in noquote(allrandoms)){
  finaldata[i] <- lapply(finaldata[i],function(x) as.factor(x))
}
finaldata  <- as.data.table(finaldata)

#     for(i in random_campaigns)
#     {   
#       finaldata <- as.data.frame(finaldata)
#       finaldata[,i] <- as.factor(finaldata[,i])                                                  
#       
#       # re-group random intercept levels 
#       finaldata[,i] <- str_sub(finaldata[,i], start=1, end=2)
#       finaldata[,i] <- gsub("_","",finaldata[,i])
#       finaldata[,i] <- as.factor(finaldata[,i])
#       
#       # create a csv file with the counts of each level of the publisher variable
#       #write.csv(table(finaldata$group,finaldata[,eval(parse(text=random_campaigns[i]))]),file="ExposureFreq5w.csv")
#       
#       levelstokeep <- c("14","15","41","3","8","2","6","10","24")
#       all_levels <- unique(finaldata[,i])
#       
#       other_levels <- setdiff(all_levels,c(levelstokeep,"\\N"))
#       
#       finaldata <- as.data.table(finaldata)
#       for(j in other_levels){
#         finaldata <- finaldata[i==j,i:="Other"]
#       }
#       
#     }

# GMT Check

# set variables as.numeric and replace NA's with zero
finaldata <- as.data.frame(finaldata)
indx <- sapply(finaldata, is.character)
options(warn=-1)
finaldata[indx] <- lapply(finaldata[indx],function(x) as.numeric(as.character(x)))
finaldata[,exposed_flag_var] <- NULL
options(warn=0)

# replace NAs with zero for numeric variables 
finaldata <- as.data.frame(finaldata)
finaldata[sapply(finaldata, is.numeric)][is.na(finaldata[sapply(finaldata, is.numeric)])] <- 0
finaldata <- as.data.table(finaldata)

# set as zero the //N value of the group variable
finaldata <- finaldata[is.na(group),group:="0"]
finaldata <- finaldata[group=="\\N",group:="0"]
finaldata$group <- factor(finaldata$group)
finaldata$group <- relevel(finaldata$group,"0")

########## QCs ##########

# Create a summary of the dataset
t1 <- as.data.frame(finaldata[,sapply(finaldata,is.numeric)])
finaldata_num <- as.data.frame(finaldata)
finaldata_num <- finaldata_num[,c(which(t1==T))]
finaldata_num <- as.data.table(finaldata_num)
finaldata_num <- finaldata_num[,c(3:(which(names(finaldata_num)=="state")-1),((which(names(finaldata_num)=="mail_responder"))+1):length(names(finaldata_num))),with=F]
finaldata_num <- as.data.frame(finaldata_num)

summary_num <- sapply(finaldata_num[,!names(finaldata_num) %in% c("exposed_flag","experian_id","panid")],function(x) c(min(x,na.rm=T),quantile(x,c(0.25),na.rm=T),mean(x,na.rm=T),median(x,na.rm=T), quantile(x,c(0.75),na.rm=T),max(x,na.rm=T),sum(x,na.rm=T)))
rownames(summary_num) <- c("min","Q1","mean","median","Q3","max","sum")

na_count <-sapply(finaldata_num[,!names(finaldata_num) %in% c("exposed_flag","experian_id","panid")], function(y) sum(length(which(is.na(y)))))
NAs  <- as.vector(na_count)
N <- nrow(finaldata)-na_count

#write csv
write.csv(t(rbind(N,NAs,summary_num)),paste0(output,"QCs/","Summary.csv"),na='')

# Create Crosstables for campaign and pre campaign period
sink(paste0(output,"QCs/","PreCampaign_Summary.txt")) 
CrossTable(finaldata$group,finaldata$Buyer_Pre_P1, expected = FALSE)
sink() 
sink(paste0(output,"QCs/","Campaign_Summary.txt")) 
CrossTable(finaldata$group,finaldata$Buyer_Pos_P1,expected = FALSE)
sink() 

# Create check list for factors and Buyers
buy.pre <-  unique(finaldata$Buyer_Pre_P1)
buy.pos <- unique(finaldata$Buyer_Pos_P1)
non.pre <- unique(finaldata$Nonbuyer_Pre_P1)
gr <- unique(finaldata$group)
mos <-  unique(finaldata$Mosaic)
ban <- unique(finaldata$banner)
exp.uniq <- length(unique(finaldata$experian_id))

# check freq for demographic variables
sink(paste0(output,"QCs/","Demographics_Summary.txt")) 
xtabs(~estimated_hh_income,finaldata)
cat("\n")
xtabs(~person_1_gender,finaldata)
cat("\n")
xtabs(~number_of_children_in_living_Un,finaldata)
cat("\n")
xtabs(~hh_age,finaldata)

cat("\n")
ftable(xtabs(~group+estimated_hh_income,finaldata))
cat("\n")
ftable(xtabs(~group+person_1_gender,finaldata))
cat("\n")
ftable(xtabs(~group+number_of_children_in_living_Un,finaldata))
cat("\n")
ftable(xtabs(~group+hh_age,finaldata))
sink()

finaldata_buyers <- finaldata[Buyer_Pos_P1=="1",]
sink(paste0(output,"QCs/","Demographics_Summary_Buyers_Campaign_Period.txt")) 
cat("\n")
xtabs(~estimated_hh_income,finaldata_buyers)
cat("\n")
xtabs(~person_1_gender,finaldata_buyers)
cat("\n")
xtabs(~number_of_children_in_living_Un,finaldata_buyers)
cat("\n")
xtabs(~hh_age,finaldata_buyers)

cat("\n")
ftable(xtabs(~group+estimated_hh_income,finaldata_buyers))
cat("\n")
ftable(xtabs(~group+person_1_gender,finaldata_buyers))
cat("\n")
ftable(xtabs(~group+number_of_children_in_living_Un,finaldata_buyers))
cat("\n")
ftable(xtabs(~group+hh_age,finaldata_buyers))
sink()

sink(paste0(output,"QCs/","Buyers Data Metrics.txt")) 
list(
  Mean_Dollars_Spent_per_Trip_POST_P1_Non_Exposed=mean(finaldata[group=="0" & Buyer_Pos_P1=="1",]$Dol_per_Trip_POS_P1),
  Mean_Dollars_Spent_per_Trip_PRE_P1_Non_Exposed=mean(finaldata[group=="0" & Buyer_Pos_P1=="1",]$Dol_per_Trip_PRE_P1),
  Mean_Dollars_Spent_per_Trip_POST_P1_Exposed=mean(finaldata[group!="0" & Buyer_Pos_P1=="1",]$Dol_per_Trip_POS_P1),
  Mean_Dollars_Spent_per_Trip_PRE_P1_Exposed=mean(finaldata[group!="0" & Buyer_Pos_P1=="1",]$Dol_per_Trip_PRE_P1),
  Mean_Trips_POST_P1_Non_Exposed=mean(finaldata[group=="0" & Buyer_Pos_P1=="1",]$Trps_POS_P1),
  Mean_Trips_PRE_P1_Non_Exposed=mean(finaldata[group=="0" & Buyer_Pos_P1=="1",]$Trps_PRE_P1),
  Mean_Trips_POST_P1_Exposed=mean(finaldata[group!="0" & Buyer_Pos_P1=="1",]$Trps_POS_P1),
  Mean_Trips_PRE_P1_Exposed=mean(finaldata[group!="0" & Buyer_Pos_P1=="1",]$Trps_PRE_P1),
  
  SD_Dol_per_Trip_POST_P1_Non_Exposed=sd(finaldata[group=="0" & Buyer_Pos_P1=="1",]$Dol_per_Trip_POS_P1),
  SD_Dol_per_Trip_PRE_P1_Non_Exposed=sd(finaldata[group=="0" & Buyer_Pos_P1=="1",]$Dol_per_Trip_PRE_P1),
  SD_Dol_per_Trip_POST_P1_Exposed=sd(finaldata[group!="0" & Buyer_Pos_P1=="1",]$Dol_per_Trip_POS_P1),
  SD_Dol_per_Trip_PRE_P1_Exposed=sd(finaldata[group!="0" & Buyer_Pos_P1=="1",]$Dol_per_Trip_PRE_P1),
  SD_Trips_POST_P1_Non_Exposed=sd(finaldata[group=="0" & Buyer_Pos_P1=="1",]$Trps_POS_P1),
  SD_Trips_PRE_P1_Non_Exposed=sd(finaldata[group=="0" & Buyer_Pos_P1=="1",]$Trps_PRE_P1),
  SD_Trips_POST_P1_Exposed=sd(finaldata[group!="0" & Buyer_Pos_P1=="1",]$Trps_POS_P1),
  SD_Trips_PRE_P1_Exposed=sd(finaldata[group!="0" & Buyer_Pos_P1=="1",]$Trps_PRE_P1),
  
  Min_Dol_per_Trip_POST_P1_Non_Exposed=min(finaldata[group=="0" & Buyer_Pos_P1=="1",]$Dol_per_Trip_POS_P1),
  Min_Dol_per_Trip_PRE_P1_Non_Exposed=min(finaldata[group=="0" & Buyer_Pos_P1=="1",]$Dol_per_Trip_PRE_P1),
  Min_Dol_per_Trip_POST_P1_Exposed=min(finaldata[group!="0" & Buyer_Pos_P1=="1",]$Dol_per_Trip_POS_P1),
  Min_Dol_per_Trip_PRE_P1_Exposed=min(finaldata[group!="0" & Buyer_Pos_P1=="1",]$Dol_per_Trip_PRE_P1),
  Min_Trips_POST_P1_Non_Exposed=min(finaldata[group=="0" & Buyer_Pos_P1=="1",]$Trps_POS_P1),
  Min_Trips_PRE_P1_Non_Exposed=min(finaldata[group=="0" & Buyer_Pos_P1=="1",]$Trps_PRE_P1),
  Min_Trips_POST_P1_Exposed=min(finaldata[group!="0" & Buyer_Pos_P1=="1",]$Trps_POS_P1),
  Min_Trips_PRE_P1_Exposed=min(finaldata[group!="0" & Buyer_Pos_P1=="1",]$Trps_PRE_P1),
  
  Max_Dol_per_Trip_POST_P1_Non_Exposed=max(finaldata[group=="0" & Buyer_Pos_P1=="1",]$Dol_per_Trip_POS_P1),
  Max_Dol_per_Trip_PRE_P1_Non_Exposed=max(finaldata[group=="0" & Buyer_Pos_P1=="1",]$Dol_per_Trip_PRE_P1),
  Max_Dol_per_Trip_POST_P1_Exposed=max(finaldata[group!="0" & Buyer_Pos_P1=="1",]$Dol_per_Trip_POS_P1),
  Max_Dol_per_Trip_PRE_P1_Exposed=max(finaldata[group!="0" & Buyer_Pos_P1=="1",]$Dol_per_Trip_PRE_P1),
  Max_Trips_POST_P1_Non_Exposed=max(finaldata[group=="0" & Buyer_Pos_P1=="1",]$Trps_POS_P1),
  Max_Trips_PRE_P1_Non_Exposed=max(finaldata[group=="0" & Buyer_Pos_P1=="1",]$Trps_PRE_P1),
  Max_Trips_POST_P1_Exposed=max(finaldata[group!="0" & Buyer_Pos_P1=="1",]$Trps_POS_P1),
  Max_Trips_PRE_P1_Exposed=max(finaldata[group!="0" & Buyer_Pos_P1=="1",]$Trps_PRE_P1)
)
sink()

cat("Please check summary.csv and proceed with the modeling process")

# remove HHs with no gender info
finaldata <- finaldata[!person_1_gender=="U",]

# aggregate U and L levels of hh income
finaldata <- finaldata[estimated_hh_income=="U",estimated_hh_income:="L"]

# check and drop exposed HHs with no publisher info or non-exposed HHs with publisher info
panid_out_all <- c()
if(length(random_campaigns)>0){
  for(k in random_campaigns){
    finaldata <- finaldata[eval(parse(text=k))=="\\N"|eval(parse(text=k))=="NULL"|eval(parse(text=k))=="0"|eval(parse(text=k))=="NONE",eval(parse(text=k)):="none"]
    cat(paste0(nrow(finaldata[(group=="0" & eval(parse(text=k))!="none")])," non exposed HHs with publisher info "))
    panid_out <- finaldata[(group=="0" & eval(parse(text=k))!="none"),panid]
    finaldata <- finaldata[!(group=="0" & eval(parse(text=k))!="none"),]  
    cat('\n')
    # check and remove HHs that were exposed and we had no publisher info
    cat(paste0(nrow(finaldata[(group!="0" & eval(parse(text=k))=="none")])," exposed HHs with no publisher info "))
    finaldata <- finaldata[!(group!="0" & eval(parse(text=k))=="none")]
    panid_out2 <- finaldata[(group!="0" & eval(parse(text=k))=="none"),panid]
    panid_out_all <- unique(c(panid_out_all,panid_out,panid_out2))
  }
}

# apply matching rules for control group vs exposed (default is TRUE)
Match_control_exposed_groups <- TRUE  #FALSE or TRUE

if(Match_control_exposed_groups==T & length(ProScore)==0){
  finaldata <- as.data.table(finaldata)
  data_exposed <- finaldata[group!="0" ,]
  data_nonexposed1 <- finaldata[group=="0" & Buyer_Pre_P1=="1",]
  data_nonexposed0 <- finaldata[group=="0" & Buyer_Pre_P1=="0",]
  
  if(nrow(finaldata[group=="0",])>2000000){
    new_data_dim <- 0.3*nrow(finaldata[group=="0",])  
  } else {
    if(nrow(finaldata[group=="0",])>1000000){
      new_data_dim <- 0.4*nrow(finaldata[group=="0",])
    } else {
      if(nrow(finaldata[group=="0",])>750000){
        new_data_dim <- 0.6*nrow(finaldata[group=="0",])
      } else {
        new_data_dim <- 0.7*nrow(finaldata[group=="0",])
      }
    }
  }
  
  dim_sample0 <- round(((new_data_dim-nrow(data_exposed))/(1+(nrow(data_exposed[Buyer_Pre_P1=="1",])/(nrow(data_exposed[Buyer_Pre_P1=="0",]))))))
  dim_sample1 <- new_data_dim-nrow(data_exposed)-dim_sample0
  
  new_data_nonexposed_11 <- as.data.frame(data_nonexposed1[sample(nrow(data_nonexposed1), dim_sample1),])
  new_data_nonexposed_01 <- as.data.frame(data_nonexposed0[sample(nrow(data_nonexposed0), dim_sample0),])
  data_exposed <- as.data.frame(data_exposed)
  finaldata_sample <- rbind(data_exposed,new_data_nonexposed_11,new_data_nonexposed_01)
  
  finaldata <- as.data.table(finaldata_sample)
}

if(Match_control_exposed_groups==T & length(ProScore)>0){
  exposed_proportion <- prop.table(xtabs(~eval(parse(text=ProScore)),finaldata[group!="0",]))
  proscore_levels <- names(exposed_proportion)
  
  if(nrow(finaldata[group=="0",])>2000000){
    new_data_dim <- 0.3*nrow(finaldata[group=="0",])  
  } else {
    if(nrow(finaldata[group=="0",])>1000000){
      new_data_dim <- 0.4*nrow(finaldata[group=="0",])
    } else {
      if(nrow(finaldata[group=="0",])>800000){
        new_data_dim <- 0.6*nrow(finaldata[group=="0",])
      } else {
        new_data_dim <- 0.7*nrow(finaldata[group=="0",])
      }
    }
  }
  
  sample_control_data <- c()
  
  for (i in proscore_levels){
    sample_i_dim <- round(new_data_dim*exposed_proportion[names(exposed_proportion)==i])
    temp_data <- finaldata[eval(parse(text=ProScore))==i & group=="0",]
    data_i <- temp_data[sample(nrow(temp_data), sample_i_dim),]
    sample_control_data <- rbind(sample_control_data,data_i)
    #    cat(i)
    #    cat("\n")
    #    cat(length(sample_control_data$panid))
    #    cat("\n")
  }
  
  sample_data <- rbind(sample_control_data,finaldata[group!="0",])
  
  sample_data <- as.data.table(sample_data)
  data_exposed <- as.data.table(sample_data[group!="0",])
  
  data_nonexposed1 <- sample_data[group=="0" & Buyer_Pre_P1=="1",]
  data_nonexposed0 <- sample_data[group=="0" & Buyer_Pre_P1=="0",]
  
  if(nrow(data_nonexposed1)/nrow(data_nonexposed0) > nrow(data_exposed[Buyer_Pre_P1=="1",])/nrow(data_exposed[Buyer_Pre_P1=="0",])){
    new_data_nonexposed_11 <- as.data.frame(data_nonexposed1[sample(nrow(data_nonexposed1), (nrow(data_exposed[Buyer_Pre_P1=="1",])/nrow(data_exposed[Buyer_Pre_P1=="0",]))*nrow(data_nonexposed0)),])
    data_exposed <- as.data.frame(data_exposed)
    finaldata_sample <- rbind(data_exposed,new_data_nonexposed_11,data_nonexposed0)
  } else {
    new_data_nonexposed_01 <- as.data.frame(data_nonexposed0[sample(nrow(data_nonexposed0), (nrow(data_exposed[Buyer_Pre_P1=="0",])/nrow(data_exposed[Buyer_Pre_P1=="1",]))*nrow(data_nonexposed1)),])
    data_exposed <- as.data.frame(data_exposed)
    finaldata_sample <- rbind(data_exposed,new_data_nonexposed_01,data_nonexposed1)
  }
  length(finaldata_sample$panid)
  
  finaldata <- as.data.table(finaldata_sample)
}

# check proportions of the sample dataset
if(Match_control_exposed_groups==T ){
  finaldata_sample <- as.data.table(finaldata_sample)
  sink(paste0(output,"Matching_data_summary.txt"))
  if(length(ProScore)>0){
    cat("\n")
    xt <- xtabs(~eval(parse(text=ProScore)),finaldata_sample[group!="0",])
    names(dimnames(xt)) <- c("Proportions of ProScore in the exposed group")
    print(prop.table(xt))
    cat("\n\n")
    xt <- xtabs(~eval(parse(text=ProScore)),finaldata_sample[group=="0",])
    names(dimnames(xt)) <- c("Proportions of ProScore in the control group")
    print(prop.table(xt))
  }
  cat("\n\n")
  xt <- xtabs(~Buyer_Pre_P1,finaldata_sample[group!="0",])
  names(dimnames(xt)) <- c("Percentages of target product buyers during the Pre-Campaign Period in the exposed group")
  print(prop.table(xt))
  cat("\n\n")
  xt <- xtabs(~Buyer_Pre_P1,finaldata_sample[group=="0",])
  names(dimnames(xt)) <- c("Percentages of target product buyers during the Pre-Campaign Period in the control group")
  print(prop.table(xt))
  cat("\n\n")
  cat('Average_Trips_PRE_P1_Non_Exposed_matched_data')
  cat("\n")
  cat(mean(finaldata_sample[Buyer_Pre_P1=="1" & group=="0",]$Trps_PRE_P1))
  cat("\n\n")
  cat('Average_Trips_PRE_P1__Exposed_matched_data')
  cat("\n")
  cat(mean(finaldata_sample[Buyer_Pre_P1=="1" & group!="0",]$Trps_PRE_P1))
  cat("\n\n")
  cat('Average_Spent_PRE_P1_Non_Exposed_matched_data')
  cat("\n")
  cat(mean(finaldata_sample[Buyer_Pre_P1=="1" & group=="0",]$Dol_per_Trip_PRE_P1))
  cat("\n\n")
  cat('Average_Spent_PRE_P1_Exposed_matched_data')
  cat("\n")
  cat(mean(finaldata_sample[Buyer_Pre_P1=="1" & group!="0",]$Dol_per_Trip_PRE_P1))
  sink()
}

# remove demographic variables
finaldata <- droplevels(finaldata)
finaldata$panid <- as.character(finaldata$panid)
initial_data <- finaldata[,c(4:(which(names(finaldata)=="state")-1),which(names(finaldata)=="person_1_gender"),which(names(finaldata)=="number_of_children_in_living_Un"),(which(names(finaldata)=="Mosaic")+1):length(names(finaldata))),with=F]
names(initial_data)
options(warn=-1)

#GMT
xVarsDemos <- setdiff(names(finaldata),names(initial_data))

######rm(finaldata,data_nonexposed1,temp_data,data_i,sample_control_data,sample_data,data_exposed,data_nonexposed0,finaldata_sample,finaldata_num,new_data_nonexposed_11,new_data_nonexposed_01)
options(warn=0)

# get number of products in the data
num_products <- length(grep("Buyer_Pre_P",colnames(initial_data)))-1


#exclude POST variables
indexPOS <- grep("POS",colnames(initial_data))
indexPos <- grep("Pos",colnames(initial_data))
indexPre1 <- grep("Buyer_Pre_",colnames(initial_data))
indexPre2 <- grep("Nonbuyer_Pre_",colnames(initial_data))
# check pre vars to exclude (only buyer_pre and nonbuyer_pre should be excluded)
(colnames(initial_data)[c(indexPre1,indexPre2)])
initial_vars_to_exclude <- colnames(initial_data)[c(indexPOS,indexPos,indexPre1,indexPre2)]
indexPREPOS <- grep("PRE_POS",colnames(initial_data))
indexPrePos <- grep("Pre_Pos",colnames(initial_data))
PREPOS_vars_to_include <- colnames(initial_data)[c(indexPREPOS,indexPrePos)]
ALL_vars_to_exclude <- setdiff(initial_vars_to_exclude,c(PREPOS_vars_to_include,all_mandatory_vars,scoring_vars))
initial_data <- as.data.frame(initial_data)
initial_data <- initial_data[ , !names(initial_data) %in% c(ALL_vars_to_exclude)] 

#exclude category variables(exclude P0's)
indexP0 <- grep("0",colnames(initial_data))
P0_vars_to_exclude <- colnames(initial_data)[indexP0]
indexMODEL <- grep("MODEL",colnames(initial_data))
ProScore_vars_to_include <- colnames(initial_data)[indexMODEL]
P0_vars_to_exclude <- setdiff(P0_vars_to_exclude,c(ProScore_vars_to_include,all_mandatory_vars,scoring_vars))
initial_data <- initial_data[ , !names(initial_data) %in% c(P0_vars_to_exclude)] 

#exclude new variables that are only for reporting
indexPERC <- grep("Perc_",colnames(initial_data))
indexPriPerUn <- grep("Pr_per_",colnames(initial_data))
vars_to_exclude <- colnames(initial_data)[c(indexPERC,indexPriPerUn)]
initial_data <- initial_data[ , !names(initial_data) %in% c(vars_to_exclude)] 

# segments for outliers detection
initial_data <- as.data.table(initial_data)
data_NB_NE_B <- initial_data[Buyer_Pre_P1=="0" & group=="0"  & Buyer_Pos_P1=="1",]
#data_B_E_NB <- initial_data[Buyer_Pre_P1=="1" & group!="0" & Buyer_Pos_P1=="0",]
pen_reduction <- initial_data[Buyer_Pre_P1=="1" & group=="0" & Buyer_Pos_P1=="0",]
occ_reduction <- initial_data[group=="0" & Buyer_Pos_P1=="1" & Trps_POS_P1<Trps_PRE_P1,]
dol_reduction <- initial_data[group=="0" & Buyer_Pos_P1=="1" & Dol_per_Trip_POS_P1<Dol_per_Trip_PRE_P1,]

# get variables that need to have negative sign
negative_index <- c()
for(i in 3:num_products){
  indexCompetitorVars <- grep(paste0("P",i),colnames(initial_data))  
  indexCompetitorVars2 <- grep(paste0(i,"_"),colnames(initial_data)) 
  indexGAS <- grep("AVG_PRICE",colnames(initial_data)) 
  negative_index <- c(negative_index,indexCompetitorVars,indexCompetitorVars2,indexGAS)
}
negativevars <- unique(colnames(initial_data)[negative_index])

if(P2_Competitor==T){
  indexNOCompetitorVars1 <- grep("P2",colnames(initial_data))
  indexNOCompetitorVars2 <- grep(paste0(2,"_"),colnames(initial_data)) 
  indexNOCompetitorVars <- c(indexNOCompetitorVars1,indexNOCompetitorVars2)
  NOCompetitorVars <- colnames(initial_data)[indexNOCompetitorVars]
  negativevars <- unique(c(negativevars,NOCompetitorVars))
}

# get variables that need to have positive sign
positive_index1 <- grep("P1",colnames(initial_data))  
positive_index2 <- grep("1_",colnames(initial_data))  
positive_index3 <- grep("MODEL",colnames(initial_data)) 
positive_index <- c(positive_index1,positive_index2,positive_index3)
positivevars <- colnames(initial_data)[positive_index]

# # kruskal wallis test for pair comparison of publisher's levels keep aggregated levels based on the spent aggregation
# kruskal_var <- 'Dol_per_Trip_POS_P1'
# buyers_data <- initial_data[Buyer_Pos_P1=='1',]
# kruskal_randoms <- c()
# for(i in random_campaigns){
#   kt <- kruskal.test(eval(parse(text=kruskal_var))~eval(parse(text=i)),data=buyers_data)
#   cat(i);print(kt)
#   if(kt$p.value<0.10){
#     kruskal_randoms <- append(kruskal_randoms,i)
#   } else {
#     initial_data[,eval(parse(text=i)):=NULL]
#   }
# }
# 
# warning('In case of aggregation based on Kruskal-Wallis test, make sure that the other level of each sub-campaign is named as Other_XXX')
# for(i in kruskal_randoms){
#   lev <- setdiff(levels(buyers_data[,eval(parse(text=i))]),'none')
#   if(length(lev)>2){
#     aggr <- c()
#     for(j in 1:(length(lev)-1)){
#       for( k in (j+1):(length(lev))){       
#         df <- subset(buyers_data, eval(parse(text=i))  %in% c(lev[j],lev[k]))
#         l1 <- list(c(lev[j],lev[k]),P_Value=try(round(kruskal.test(df[,eval(parse(text=kruskal_var))]~df[,eval(parse(text=i))])$p.value,5),silent=TRUE))
#         if(l1[2]=="NaN") next
#         vectorElements <- unlist(l1)
#         listnames <- names(vectorElements) <- c(paste0(i,':'),paste0(i,':'),'P_value:')
#         listout <- paste(listnames,vectorElements)
#         write.table(listout,file=paste0(output,'Kruskal.csv'), row.names = FALSE, col.names = FALSE, quote = FALSE,append=TRUE)  
#         if(l1[['P_Value']]<0.05){
#           aggr <- append(aggr,c(sapply(l1[1],`[`,1),sapply(l1[1],`[`,2)))
#           aggr <- setdiff(aggr,c('none',paste0("Other_",i)))
#         }
#       }  
#     }
#     if(length(unique(aggr))>0){
#       initial_data[!eval(parse(text=i)) %in% c(aggr,'none'),eval(parse(text=i)):=paste0("Other_",i)]
#     } else {
#       initial_data[,eval(parse(text=i)):=NULL]
#     }
#   }
# }

initial_data <- droplevels(initial_data)
saveRDS(initial_data,"initial_data.RDS")
random_campaigns <- intersect(random_campaigns,names(initial_data))

if(length(random_campaigns)>0){
  random_campaigns_factors <- paste0("+(0+group|",random_campaigns,")")
} else {
  random_campaigns_factors <- NULL
}
random_demos_factors <- paste0( "+(0+group|",random_demos,")" )

# Create Descriptives for Campaign breaks
if(length(random_campaigns)>0){
  output_checks <- paste0(output,"Descriptive Statistics for Publishers and Demographics/")
  randoms <- c(random_demos,random_campaigns)
  for(i in randoms){
    write.csv(table(initial_data[,eval(parse(text=i))]),file=paste0(output_checks,i,'.csv'),row.names=F)
    write.csv(table(initial_data$group,initial_data[,eval(parse(text=i))]),file=paste0(output_checks,i,'_group.csv'))
    
    dolocc_data <- initial_data[Buyer_Pos_P1=="1",]
    
    write.csv(table(dolocc_data[,eval(parse(text=i))]),file=paste0(output_checks,i,'_buyers.csv'),row.names=F)
    write.csv(table(dolocc_data$group,dolocc_data[,eval(parse(text=i))]),file=paste0(output_checks,i,'_group_buyers.csv')) 
  }
  
  # check exposed and non exposed HHs counts for target brand Buyers in the campaign period
  write.csv(table(initial_data$group,initial_data$Buyer_Pos_P1),file=paste0(output_checks,'buyers_pos_group.csv'))
  
  # check exposed and non exposed HHs counts for target brand Buyers in the pre-campaign period
  write.csv(table(initial_data$group,initial_data$Buyer_Pre_P1),file=paste0(output_checks,'buyers_pre_group.csv'))
  
  # calculate correlations between levels of the random intercept variables
  finaldata_cor <- as.data.frame(initial_data)
  
  for(i in randoms){
    for(level in unique(eval(parse(text=paste0("finaldata_cor$",i))))){
      finaldata_cor[paste(i, level, sep = "_")] <- ifelse(eval(parse(text=paste0("finaldata_cor$",i))) == level, 1, 0)
    }
  }
  
  cor.data <- finaldata_cor[,c((which(names(finaldata_cor)=="panid")+1):length(finaldata_cor))]
  cor.matrix <- cor(cor.data)
  write.csv(cor.matrix,file=paste0(output_checks,"random_correlation_matrix.csv"))
  
  mel.cc <- melt(cor.matrix)
  heat_map <- ggplot(mel.cc,aes(Var1, Var2))+geom_tile(aes(fill=value))+ scale_fill_viridis(name="Correlation",option="magma") + theme(axis.text.x = element_text(angle=90, vjust=0.5))+ coord_equal() 
  ggsave(filename=paste0(output_checks,"correlation_heat_map.pdf"),heat_map,width = 10, height = 10, units = "in")
  
  #Make descriptive plots for publishers, creatives etc.
  # create descriptive plots for the DolOcc breaks
  buyers_data <- initial_data[Buyer_Pos_P1=="1",]
  for(i in random_campaigns){
    buyers_data <- buyers_data[eval(parse(text=i))!="none",]
    buyers_data <- droplevels(buyers_data)
    buyers_data <- as.data.frame(buyers_data)
    res <- ldply(levels(buyers_data[,i]),function(x) round(mean(buyers_data[buyers_data[,i]==x,]$Dol_per_Trip_POS_P1),2))
    df <-cbind(res,levels(buyers_data[,i]))
    colnames(df) <- c('Average_Spent_during_the_campaign_period','Levels')
    
    p <- ggplot(df,aes(x=Levels,y=Average_Spent_during_the_campaign_period))+geom_bar(stat='identity')+ theme(axis.text.x = element_text(angle=90, vjust=0.5))+ylab('Average Spent for the target brand during the campaign period')+
      theme(legend.position="none")+theme(panel.background = element_rect(fill = 'white'))+
      geom_text(stat='identity',aes(label=Average_Spent_during_the_campaign_period,group=Levels), position = position_dodge(0.9), vjust = 0)  
    ggsave(filename=paste0(output_checks,i,'_Mean_Spent.pdf'),p,height = 8,width = 15)
    
    p <- ggplot(buyers_data,aes_string(i,'Dol_per_Trip_POS_P1'))+geom_boxplot()+theme(axis.text.x = element_text(angle = 90, vjust = 0.5))+
      ylab('Spent per Trip for the target bramd during the campaign period')
    ggsave(filename=paste0(output_checks,i,'_Spent_BoxPlot.pdf'),p,height = 8,width = 15)
    buyers_data <- as.data.table(buyers_data)
  } 
  
  # create descriptive plots for the Occ breaks
  options(warn=-1)
  a1 <- buyers_data[Trps_POS_P1>=5,Trps_POS_P1:="5"]
  options(warn=0)
  a1[,Trips_growth:=log(Trps_POS_P1/(Trps_PRE_P1+1))]
  a1$Trps_POS_P1 <- as.factor(a1$Trps_POS_P1)
  a1 <- a1[Trps_POS_P1=="5",Trps_POS_P1:="5+"]
  
  response_var <- c("Trps_POS_P1")
  for(i in random_campaigns){
    a1 <- as.data.table(a1)
    a1 <- a1[eval(parse(text=i))!="none",]
    a1 <- droplevels(a1)
    rr <- c(response_var,i)
    a1 <- as.data.frame(a1)
    df <- a1[,rr]
    mel <- melt(df,id.var=i)
    p <- ggplot(mel,aes_string(i))+geom_bar(aes(y=..count..,fill=as.factor(value)),position=position_dodge(0.9))+
      theme(axis.text.x = element_text(angle = 90, vjust = 0.5))+guides(fill=guide_legend(title=response_var))+
      geom_text(stat='count',aes(label=..count..,group=value), position = position_dodge(0.9), vjust = 0)  
    ggsave(filename=paste0(output_checks,i,'_Trips.pdf'),p,height = 8,width = 15)
    
    a2 <- as.data.table(initial_data[Buyer_Pos_P1=="1",])
    a2 <- a2[eval(parse(text=i))!="none",]
    a2 <- as.data.frame(a2)
    a2 <- droplevels(a2)
    res <- ldply(levels(a2[,i]),function(x) round(mean(a2[a2[,i]==x,]$Trps_POS_P1),2))
    df <-cbind(res,levels(a2[,i]))
    colnames(df) <- c('Average_Trips_during_the_campaign_period','Levels')
    p <- ggplot(df,aes(x=Levels,y=Average_Trips_during_the_campaign_period))+geom_bar(stat='identity')+ theme(axis.text.x = element_text(angle=90, vjust=0.5))+ylab('Average Trips for the target brand during the campaign period')+
      theme(legend.position="none")+theme(panel.background = element_rect(fill = 'white'))+
      geom_text(stat='identity',aes(label=Average_Trips_during_the_campaign_period,group=Levels), position = position_dodge(0.9), vjust = 0)  
    ggsave(filename=paste0(output_checks,i,'_Mean_Trips.pdf'),p,height = 8,width = 15)
  }
  
  # create descriptive plots for the Pen breaks
  response_var <- c("Buyer_Pos_P1")
  for(i in random_campaigns){
    data <- as.data.table(copy(initial_data))
    data <- data[eval(parse(text=i))!="none",]
    data <- droplevels(data)
    data <- as.data.frame(data)
    rr <- c(response_var,i)
    df <- data[,rr]
    mel <- melt(df,id.var=i)
    p <- ggplot(mel,aes_string(x=i))+geom_bar(aes(y=..count..,fill=as.factor(value)),position=position_dodge(0.9))+
      theme(axis.text.x = element_text(angle = 90, vjust = 0.5))+guides(fill=guide_legend(title=response_var))+
      geom_text(stat='count',aes(label=..count..,group=value), position = position_dodge(0.9), vjust = 0)  
    ggsave(paste0(output_checks,i,'_Buyers_during_Campaign.pdf'),p,height = 8,width = 15)  
  }
}

rm(dolocc_data,mel.cc,finaldata_cor,cor.data,cor.matrix,buyers_data,a1,a2,data)

#Create functions for modelling process and outputs
ggCaterpillar <- function(re,  QQ=FALSE, likeDotplot=FALSE) {
  require(ggplot2)
  f <- function(x) {
    pv   <- attr(x, "postVar")
    cols <- 1:(dim(pv)[1])
    se   <- unlist(lapply(cols, function(i) sqrt(pv[i, i, ])))
    ord  <- unlist(lapply(x, order)) + rep((0:(ncol(x) - 1)) * nrow(x), each=nrow(x))
    pDf  <- data.frame(y=unlist(x)[ord],
                       ci=1.28*se[ord], #80%CI:1.28, 90%CI:1.645, 95%CI: 1.96
                       nQQ=rep(qnorm(ppoints(nrow(x))), ncol(x)),
                       ID=factor(rep(rownames(x), ncol(x))[ord], levels=rownames(x)[ord]),
                       ind=gl(ncol(x), nrow(x), labels=names(x)))
    
    if(QQ) {  ## normal QQ-plot
      p <- ggplot(pDf, aes(nQQ, y))
      p <- p + facet_wrap(~ ind, scales="free")
      p <- p + xlab("Standard normal quantiles") + ylab("Random effect quantiles")
    } else {  ## caterpillar dotplot
      p <- ggplot(pDf, aes(ID, y)) #+ coord_flip()
      if(likeDotplot) {  ## imitate dotplot() -> same scales for random effects
        p <- p + facet_wrap(~ ind)
      } else {           ## different scales for random effects
        p <- p + facet_wrap(~ ind, scales="free_y")
      }
      p <- p + xlab("Levels") + ylab("Random effects")
    }
    
    p <- p + theme(legend.position="none")
    p <- p + geom_hline(yintercept=0)
    p <- p + geom_errorbar(aes(ymin=y-ci, ymax=y+ci), width=0, colour="black")
    p <- p + geom_point(aes(size=1.2), colour="blue") + theme(axis.text.x = element_text(angle=90, vjust=0.5)) 
    
    return(p)
    #ggsave(paste0(output,"dotplots.pdf"),p)
  }
  
  lapply(re,f)
}

se.ranef <- function (object) {   
  se.bygroup <- ranef(object, condVar = TRUE) 
  n.groupings <- length(se.bygroup) 
  for (m in 1:n.groupings) {        
    vars.m <- attr(se.bygroup[[m]], "postVar")  
    K <- dim(vars.m)[1]   
    J <- dim(vars.m)[3] 
    se.bygroup[[m]] <- array(NA, c(J, K))   
    for (j in 1:J) {          
      se.bygroup[[m]][j, ] <- sqrt(diag(as.matrix(vars.m[,, j])))  }        
    names.full <- dimnames(se.bygroup)      
    dimnames(se.bygroup[[m]]) <- list(names.full[[1]], names.full[[2]]) }
  return(se.bygroup) }

# create nloptwrap optimization method
defaultControl <- list(algorithm="NLOPT_LN_BOBYQA",xtol_rel=1e-6,maxeval=1e5)
nloptwrap2 <- function(fn,par,lower,upper,control=list(),...) {
  for (n in names(defaultControl)) 
    if (is.null(control[[n]])) control[[n]] <- defaultControl[[n]]
    res <- nloptr(x0=par,eval_f=fn,lb=lower,ub=upper,opts=control,...)
    with(res,list(par=solution,
                  fval=objective,
                  feval=iterations,
                  conv=if (status>0) 0 else status,
                  message=message))
}

augment.ranef.mer <- function(x,
                              ci.level=0.9,
                              reorder=TRUE,
                              order.var=1) {
  tmpf <- function(z) {
    if (is.character(order.var) && !order.var %in% names(z)) {
      order.var <- 1
      warning("order.var not found, resetting to 1")
    }
    ## would use plyr::name_rows, but want levels first
    zz <- data.frame(level=rownames(z),z,check.names=FALSE)
    if (reorder) {
      ## if numeric order var, add 1 to account for level column
      ov <- if (is.numeric(order.var)) order.var+1 else order.var
      zz$level <- reorder(zz$level, zz[,order.var+1], FUN=identity)
    }
    ## Q-Q values, for each column separately
    qq <- c(apply(z,2,function(y) {
      qnorm(ppoints(nrow(z)))[order(order(y))]
    }))
    rownames(zz) <- NULL
    pv   <- attr(z, "postVar")
    cols <- 1:(dim(pv)[1])
    se   <- unlist(lapply(cols, function(i) sqrt(pv[i, i, ])))
    ## n.b.: depends on explicit column-major ordering of se/melt
    zzz <- cbind(melt(zz,id.vars="level",value.name="estimate"),
                 qq=qq,std.error=se)
    ## reorder columns:
    subset(zzz,select=c(variable, level, estimate, qq, std.error))
  }
  dd <- ldply(x,tmpf,.id="grp")
  ci.val <- -qnorm((1-ci.level)/2)
  transform(dd,
            p=2*pnorm(-abs(estimate/std.error)), ## 2-tailed p-val
            lb=estimate-ci.val*std.error,
            ub=estimate+ci.val*std.error)
}


"""