addprocs(3)
using DataStructures, DataArrays , DataFrames, StatsFuns, GLM , Distributions, MixedModels, StatsBase, StatLib, JSON
@everywhere using DataStructures, DataArrays , DataFrames, StatsFuns, GLM , Distributions, MixedModels, StatsBase, StatLib, JSON

 
const cfgDefaults=OrderedDict( :P2_Competitor => true
                        ,:pvalue_lvl => 0.20  #pvalue_lvl = 0.20 
                        ,:excludedBreaks => String[]    #["estimated_hh_income","hh_age","number_of_children_in_living_un","person_1_gender"]
                        ,:excludedLevels => ["none"]
                        ,:excludedKeys => String[]
                        ,:exposed_flag_var => :exposed_flag
                        ,:sigLevel => "0.2"
                        ,:random_demos => [:estimated_hh_income,:hh_age,:number_of_children_in_living_Un,:person_1_gender]
                        ,:random_campaigns => []
                        ,:dropvars => [:exposed_flag]
                        ,:scoring_vars => [:Prd_1_Net_Pr_PRE,:Prd_1_Net_Pr_POS,:Buyer_Pos_P0,:Buyer_Pre_P0]
                        ,:occ_y_var => :Trps_POS_P1
                        ,:occ_logvar => :Trps_PRE_P1
                        ,:dolocc_y_var => :Dol_per_Trip_POS_P1
                        ,:dolocc_logvar => :Dol_per_Trip_PRE_P1
                        ,:pen_y_var => :Buyer_Pos_P1
                        ,:pen_logvar => :Buyer_Pre_P1
                        ,:TotalModelsOnly=>false
                       )

root = pwd()


#root="/mnt/resource/analytics/models"
#Default
cfgDefaults[:random_demos] = [:estimated_hh_income,:hh_age,:number_of_children_in_living_Un,:person_1_gender]
cfgDefaults[:random_campaigns] = [:Publisher_Fct1,:Targeting_Fct1]
cfgDefaults[:exposed_flag_var] = :exposed_flag_new


# --------------------------------- CAMPAIGNS ----------------------------------------
#NatC : 
cfgDefaults[:random_demos] = [:estimated_hh_income,:hh_age,:person_1_gender]
cfgDefaults[:random_campaigns] = [:Targeting_fct,:Media_Source_fct,:Media_Type_fct,:Publisher_fct,:Creative_fct]      
cfgDefaults[:exposed_flag_var] = :exposed_flag_new
root="/mnt/resource/analytics/models/NatChoice"

#-- CDW
cfgDefaults[:random_demos] = [:estimated_hh_income,:hh_age,:number_of_children_in_living_Un,:person_1_gender]
cfgDefaults[:random_campaigns] = [:Publisher_Fct1,:Targeting_Fct1]
cfgDefaults[:exposed_flag_var] = :exposed_flag_new
root="/mnt/resource/analytics/models/CDW"


# JennieO - :creative :publisher1 :placement
cfgDefaults[:random_demos] = [:estimated_hh_income,:hh_age,:number_of_children_in_living_Un,:person_1_gender]
cfgDefaults[:random_campaigns] = [:creative,:publisher1,:placement]
cfgDefaults[:exposed_flag_var] = :exposed_flag_new
root="/mnt/resource/analytics/models/Jenny-o"


# Rev7 
cfgDefaults[:random_demos] = [:hh_age,:estimated_hh_income,:number_of_children_in_living_Un,:person_1_gender]
cfgDefaults[:random_campaigns] = [:creativename, :media_source,:media_type,:publisher,:targeting]
cfgDefaults[:exposed_flag_var] = :exposed_flag_new
root="/mnt/resource/analytics/models/rev"


#ALL#10  - /mapr/mapr04p/analytics0001/analytic_users/Models/All_10_944/All_10_Modeling_944/Output 
root="/mnt/resource/analytics/models/ALL#10"
cfgDefaults[:random_demos] = [:estimated_hh_income,:hh_age,:number_of_children_in_living_Un,:person_1_gender]
cfgDefaults[:random_campaigns] = [:creative,:ad_type,:publisher,:frequency_type]
cfgDefaults[:exposed_flag_var] = :exposed_flag_new


#ALL#11  - /mapr/mapr04p/analytics0001/analytic_users/Models/All_11_983/All_11_983_Modeling/Output
root="/mnt/resource/analytics/models/ALL#11"
cfgDefaults[:random_demos] = [:estimated_hh_income, :hh_age, :number_of_children_in_living_Un, :person_1_gender]
cfgDefaults[:random_campaigns] = [:creative,:ad_type,:publisher,:frequency_type]
cfgDefaults[:exposed_flag_var] = :exposed_flag_new


#CDW#6  - /mapr/mapr04p/analytics0001/analytic_users/Models/CDW_6_949/CDW6_modeling/Output7”
root="/mnt/resource/analytics/models/CDW#6"
cfgDefaults[:exposed_flag_var] = :exposed_flag_new
cfgDefaults[:random_demos] = [:estimated_hh_income, :hh_age, :number_of_children_in_living_Un, :person_1_gender]
cfgDefaults[:random_campaigns] = [:viewability_break, :Publisher_Fct1,:Targeting_Fct1]


#HormelChili#8  - /mapr/mapr04p/analytics0001/analytic_users/Models/Hormel_Chili_8/Hormel_Chili8_model/output
root="/mnt/resource/analytics/models/HormelChili#8"
cfgDefaults[:exposed_flag_var] = :exposed_flag_new
cfgDefaults[:random_demos] = [:estimated_hh_income, :hh_age, :number_of_children_in_living_Un, :person_1_gender]
cfgDefaults[:random_campaigns] = [:publisher,:media_type,:creativename]


#Rev#8  -        "/mapr/mapr04p/analytics0001/analytic_users/Models/Hormel_Rev_8_967/Hormel_Rev_8_Data/Output1/"
root="/mnt/resource/analytics/models/Rev#8"
cfgDefaults[:exposed_flag_var] = :exposed_flag_new
cfgDefaults[:random_demos] = [:estimated_hh_income, :hh_age, :number_of_children_in_living_Un, :person_1_gender]
cfgDefaults[:random_campaigns] = [:Creative_fct_1,:Targeting_fct_1,:Publisher_fct_1,:MediaType_fct_1,:MediaSource_fct_1]



# ------------------------------- END CAMPAIGNS --------------------------------------



cd(root)
function loadDF()
    #cd("/media/u01/analytics/scoring/Healthy/modeling5/")
    #df_data = readtable("csv_final_healthychoice1_I3.csv",header=true); #lowercase(df_data)
    #df_h = readtable("Headers_healthy_choice3.csv",header=false); #lowercase(df_data)
    #names!(df_data, convert(Array{Symbol}, df_h[:x1]) ) 
  
    df_data = readtable(root*"/orig.csv",header=false);
    df_h = readtable(root*"/origHead.csv",header=false);  
    names!(df_data, convert(Array{Symbol}, df_h[:x1]) )
end
df_in=loadDF()



function Pre_out(df_in::DataFrame)
     df_cat_pre = df_in[df_in[:Buyer_Pre_P0] .==1 , [:Prd_0_Net_Pr_PRE,:experian_id]]
     df_cat_pos = df_in[(df_in[:Buyer_Pre_P0] .==0) & (df_in[:Buyer_Pos_P0] .==1) , [:experian_id]]
     median_df = median(df_cat_pre[:Prd_0_Net_Pr_PRE])
     df_cat_pre[:Prd_0_Net_Pr_PRE_med1] = abs(df_cat_pre[:Prd_0_Net_Pr_PRE]-median_df)
     MAD=median(df_cat_pre[:Prd_0_Net_Pr_PRE_med1])
     df_cat_pre[:Prd_0_Net_Pr_PRE_med2] = (0.6745*(df_cat_pre[:Prd_0_Net_Pr_PRE]-median_df))/MAD
     df_cat_pre_zsc = df_cat_pre[abs(df_cat_pre[:Prd_0_Net_Pr_PRE_med2]) .< 3.5,:]
     df_cat_pre_zsc_1 = df_cat_pre_zsc[:,[:experian_id]]
     df_cat_pre_zsc_f = vcat(df_cat_pos,df_cat_pre_zsc_1)
     df_in_pout =  join(df_in, df_cat_pre_zsc_f, on =  :experian_id , kind = :inner);
end
df_in = Pre_out(df_in);



function isValid(df_data::DataFrame,cfg::OrderedDict)
    function checkValid(iarr::Array{Symbol})  length(setdiff(iarr, names(df_data))) > 0 ? false : true end
    !checkValid(cfg[:all_mandatory_vars]) ? error("ERROR: Not all mandatory_vars in dataset ") : println("VALID : mandatory_vars") 
    !checkValid(cfg[:scoring_vars]) ? error("ERROR: Not all scoring_vars in dataset ") : println("VALID : scoring_vars") 
end






function getCFG(df_in::DataFrame)
    cfg=StatLib.loadCFG(cfgDefaults, pwd()*"/app.cfg")
    #cfg[:exposed_flag_var] = :exposed_flag_new                  # Go In app.cfg
    #cfg[:random_campaigns] = [:Publisher_Fct1,:Targeting_Fct1]  # Go In app.cfg

    cfg[:allrandoms] = vcat(cfg[:random_demos],cfg[:random_campaigns])
    #cfg[:ProScore] = grep1("MODEL",names(df_in))   
    ps = filter(x->contains(string(x), "MODEL"), names(df_in)) # Set the ProScore variable in the dataset   #push!(cfg[:all_mandatory_vars], cfg[:ProScore])
    cfg[:ProScore] = length(ps) > 0 ? ps[1] : :MISSING_MODEL_VARIABLE_IN_DATA
    cfg[:num_products] = length( grep("Buyer_Pre_P",names(df_in)))-1  # get number of products in the data
    #rework vars after loading cfg
    cfg[:random_campaigns] = intersect(cfg[:random_campaigns],names(df_in))
 
    cfg[:occ_logvar_colname] = Symbol("LOG_"*string(cfg[:occ_logvar]))
    cfg[:dolocc_logvar_colname] = Symbol("LOG_"*string(cfg[:dolocc_logvar]))
    cfg[:pen_logvar_colname] = Symbol("LOG_"*string(cfg[:pen_logvar]))
    
    cfg[:all_mandatory_vars] = vcat( [:experian_id,
                                      cfg[:pen_y_var],cfg[:pen_logvar],
                                      cfg[:occ_y_var], cfg[:occ_logvar],
                                      cfg[:dolocc_y_var], cfg[:dolocc_logvar]
                                     ]
                                     , cfg[:random_demos]
                                   )
    return cfg
end
cfg=getCFG(df_in)

isValid(df_in,cfg)



function reworkCFG!(df_in::DataFrame,cfg::OrderedDict)
    xflags=[cfg[:exposed_flag_var],:exposed_flag,:Nonbuyer_Pre_P1]  # :Nonbuyer_Pre_P1 no longer used
    features = setdiff(names(df_in),xflags)
    if :state in names(df_in)
        cfg[:xVarsDemos]=vcat([:experian_id, :banner, :Prd_0_Qty_PRE],features[findfirst(features, :state):findfirst(features, :Mosaic)])   # exclude demos between....
    else
        cfg[:xVarsDemos]=Symbol[]
    end
    cfg[:xVarsDemos]=setdiff(cfg[:xVarsDemos],[:person_1_gender,:number_of_children_in_living_Un]) # exclude person, # from non-used demos
    cfg[:xVarsPost] = grep(["POS","Pos","Buyer_Pre_","Nonbuyer_Pre_"],features)   #exclude POST variables
    cfg[:iVarsPREPOS] = grep(["PRE_POS","Pre_Pos"],features) 
    features=setdiff(features, setdiff(  vcat(cfg[:xVarsDemos],cfg[:xVarsPost]) ,cfg[:iVarsPREPOS])  )
    cfg[:xVarsP0] =  setdiff(grep("0", features ),[cfg[:ProScore]] ) #exclude category variables(exclude P0's)  
    features=setdiff(features,cfg[:xVarsP0])
    cfg[:xVarsReports] = grep(["Perc_","Pr_per_"],features)  #exclude Reporting vars
    features=setdiff(features,cfg[:xVarsReports])
    cfg[:xvars] = vcat(xflags,cfg[:xVarsDemos], cfg[:xVarsPost],cfg[:xVarsP0], cfg[:xVarsReports],cfg[:dropvars])
    features=setdiff(features,cfg[:dropvars])
    cfg[:ivars] = vcat(cfg[:iVarsPREPOS],cfg[:all_mandatory_vars],cfg[:scoring_vars])
    cfg[:ALL_vars_to_exclude] = setdiff(cfg[:xvars], cfg[:ivars])
    df_in[:group] = df_in[cfg[:exposed_flag_var]]
    df_in[:panid] = df_in[:experian_id] 
    features = setdiff(unique(vcat(features,cfg[:iVarsPREPOS],cfg[:all_mandatory_vars],cfg[:scoring_vars],[:group,:panid])  ) ,[:experian_id] )
    cfg[:negativevars] = grep(vec(hcat([["P"*string(i),string(i)*"_"] for i=3:cfg[:num_products]]...)),features)   # get variables that need to have negative sign
    cfg[:positivevars] = grep(["P1", "1_","MODEL"], features)    # get variables that need to have positive sign
    if cfg[:P2_Competitor] == true  
        cfg[:negativevars] = unique(vcat(cfg[:negativevars],grep(["P2","2_"],features))) 
    else
        cfg[:positivevars] = unique(vcat(cfg[:positivevars],grep(["P2","2_"],features))) 
    end
    return df_in[features]
end

dfd = reworkCFG!(df_in,cfg)



#######################################
#--------- Data Manipulation ---------#
#######################################

function data_Prep(dfd::DataFrame, cfg::OrderedDict)
    dfd[:isO]=false
    dfd[dfd[:number_of_children_in_living_Un].>=4,:number_of_children_in_living_Un] = 4  # aggregate #of children for 4+ L_114
    if typeof(dfd[:group]) in [DataArray{String,1}] 
        dfd[ findin(dfd[:group],["//N","\\N"]), :group] = "0" 
        dfd[DataArrays.isna(dfd[:group]), :group]="0"
        dfd[:group] = [parse(Int64,s) for s = dfd[:group]]
    else
        dfd[ DataArrays.isnan(dfd[:group]), :group] = 0
    end
    vars = DataFrame(names=names(dfd),eltypes=eltypes(dfd))
    for c in setdiff(vars[findin(vars[:eltypes],[String]),:names],cfg[:allrandoms]) # set variables as.numeric and replace NA's with zero
        print("Convert String: String->Numeric: ",c)
        try dfd[c] = map( x -> DataArrays.isna.(x) ?  NaN : convert(Float64, x)  , dfd[c]) catch e println("  (failed)") end  #NA to NaN
        try dfd[c] = convert(Array{Float64}, dfd[c]) catch e end   # To Float64
    end
    vars = DataFrame(names=names(dfd),eltypes=eltypes(dfd))
    for c in setdiff(vars[findin(vars[:eltypes],[Float64]),:names],cfg[:allrandoms])  # replace NaN's with zero for numeric variables 
        println("Replace Float64 NaN (0.0): ",c)
        dfd[ DataArrays.isnan(dfd[c]), c] = 0.0
    end
    for c in  setdiff(vars[findin(vars[:eltypes],[Int64]),:names],cfg[:allrandoms])    
        println("Replace Int64 NaN (0) : ",c)
        dfd[ DataArrays.isnan(dfd[c]), c] = 0
    end
    # NOTE : Do QCs here
    dfd[dfd[:person_1_gender].=="U",:isO] = true   # remove HHs with no gender info
    #dfd[dfd[:person_1_gender].=="U",:whyO] = "person_1_gender=U; remove HHs with no gender info"
    dfd[findin(dfd[:estimated_hh_income],["U","L"]),:estimated_hh_income]="L" # aggregate U and L levels of hh income

    for r in cfg[:random_campaigns]    # check and drop exposed HHs with no publisher info or non-exposed HHs with publisher info
        dfd[findin(dfd[r],["\\N","NULL","0","NONE"])  ,r] ="none"
        #dfd[ (dfd[:isO].==false) & (dfd[r].!="none") & (dfd[:group].==0) ,:whyO] = "non exposed HHs with publisher info"
        dfd[ (dfd[:isO].==false) & (dfd[r].!="none") & (dfd[:group].==0) ,:isO] = true
        #println(r," non exposed HHs with publisher info : ",nrow(dfd[dfd[:whyO].=="non exposed HHs with publisher info",:] )   )
        #dfd[ (dfd[:isO].==false) & (dfd[r].=="none") & (dfd[:group].!=0) ,:whyO] = "exposed HHs with no publisher info"   
        dfd[ (dfd[:isO].==false) & (dfd[r].=="none") & (dfd[:group].!=0) ,:isO] = true
        #println(r," exposed HHs with no publisher info : ",nrow(dfd[dfd[:whyO].=="exposed HHs with no publisher info",:] )   )
    end
    # segments for outliers detection
    dfd[:data_NB_NE_B] = false
    dfd[ (dfd[:Buyer_Pre_P1].==0 ) & (dfd[:group].==0 ) & (dfd[:Buyer_Pos_P1].==1 ) ,:data_NB_NE_B] = true
    dfd[:data_B_E_NB] = false
    dfd[ (dfd[:Buyer_Pre_P1].==1 ) & (dfd[:group].==0 ) & (dfd[:Buyer_Pos_P1].==0 ) ,:data_B_E_NB] = true
    dfd[:pen_reduction] = false
    dfd[ (dfd[:Buyer_Pre_P1].==1) & (dfd[:group].==0) & (dfd[:Buyer_Pos_P1].==0 )  ,:pen_reduction] = true
    dfd[:occ_reduction] = false
    dfd[ (dfd[:group].==0) & (dfd[:Buyer_Pos_P1].==1) & (dfd[:Trps_POS_P1].< dfd[:Trps_PRE_P1] ) ,:occ_reduction] = true
    dfd[:dolocc_reduction] = false
    dfd[  (dfd[:group].==0) & (dfd[:Buyer_Pos_P1].==1) & (dfd[:Dol_per_Trip_POS_P1].< dfd[:Dol_per_Trip_PRE_P1] )  , :dolocc_reduction] = true
    return dfd[dfd[:isO].==false, : ]   #[setdiff(names(dfd),[:isO,:whyO])] 
end

dfd = data_Prep(dfd, cfg);




function MatchMe(dfd::DataFrame,cfg::OrderedDict)
    df=dfd[dfd[:isO].==false,:]
    df_exp     = df[df[:group].==1,:]
    df_unexp   = df[df[:group].==0,:]
    df_exp_dim   = nrow(df_exp)
    df_unexp_dim = nrow(df_unexp)
    new_unexp_dim = df_unexp_dim*(df_unexp_dim>2000000 ? 0.3 : df_unexp_dim>1000000 ? 0.4 : df_unexp_dim>750000 ? 0.6 : 0.7)
    if length(string(cfg[:ProScore])) == 0
        df_unexp_1 =  df[(df[:group].==0)&(df[:Buyer_Pre_P1].==1),:]
        df_unexp_0 =  df[(df[:group].==0)&(df[:Buyer_Pre_P1].==0),:]
        
        df_exp_1_dim = nrow(df[(df[:group].==1)&(df[:Buyer_Pre_P1].==1),:])
        df_exp_0_dim = nrow(df[(df[:group].==1)&(df[:Buyer_Pre_P1].==0),:])
        df_unexp_1_dim = nrow(df_unexp_1)
        df_unexp_0_dim =  nrow(df_unexp_0)     
        dim_sample0 = round(Int64, (new_unexp_dim-df_exp_dim  ) / (1+(df_exp_1_dim / df_exp_0_dim)) )
        dim_sample1 = round(Int64, new_unexp_dim - df_exp_dim - dim_sample0)    
        
        new_df_unexp_1 = df_unexp_1[sample(1:size(df_unexp_1,1), dim_sample1 ),:]
        new_df_unexp_0 = df_unexp_0[sample(1:size(df_unexp_0,1), dim_sample0 ),:]
        dfd_sample  = vcat(df_exp,new_df_unexp_1,new_df_unexp_0)
    elseif length(string(cfg[:ProScore])) > 0    
        sample_control_data=similar(df_unexp, 0)
        for (key, value) in countmap(df_exp[cfg[:ProScore]])
            sample_dim=round(Int64,new_unexp_dim*(value/df_exp_dim))
            temp_data = df_unexp[df_unexp[cfg[:ProScore]].==key,:]
            if sample_dim > size(temp_data,1) sample_dim = size(temp_data,1) end  #??
            samp_data = temp_data[sample(1:size(temp_data,1), sample_dim, replace=false),:]
            sample_control_data = vcat(sample_control_data,    samp_data   )
        end
        
        sample_data = vcat(sample_control_data,df_exp)
        sample_df_unexp = sample_data[sample_data[:group].==0,:]
        sample_df_exp   = sample_data[sample_data[:group].==1,:]
        sample_df_unexp_1 = sample_data[(sample_data[:group].==0)&(sample_data[:Buyer_Pre_P1].==1),:]  
        sample_df_unexp_0 = sample_data[(sample_data[:group].==0)&(sample_data[:Buyer_Pre_P1].==0),:]    
        sample_df_exp_1 =  sample_data[(sample_data[:group].==1)&(sample_data[:Buyer_Pre_P1].==1) ,:]
        sample_df_exp_0 =  sample_data[(sample_data[:group].==1)&(sample_data[:Buyer_Pre_P1].==0) ,:]
        sample_df_unexp_1_dim = nrow(sample_df_unexp_1)
        sample_df_unexp_0_dim = nrow(sample_df_unexp_0)
        sample_df_exp_1_dim = nrow(sample_df_exp_1)
        sample_df_exp_0_dim = nrow(sample_df_exp_0)
        dim_sampleA = round(Int64,(sample_df_exp_1_dim/sample_df_exp_0_dim)*sample_df_unexp_0_dim)
        dim_sampleB = round(Int64,(sample_df_exp_0_dim/sample_df_exp_1_dim)*sample_df_unexp_1_dim)
        
        if sample_df_unexp_1_dim/sample_df_unexp_0_dim > sample_df_exp_1_dim/sample_df_exp_0_dim
            new_df_unexp_1 = sample_df_unexp_1[sample(1:sample_df_unexp_1_dim,dim_sampleA , replace=false),:]
            dfd_sample = vcat(sample_df_exp,new_df_unexp_1,sample_df_unexp_0)
        else
            new_df_unexp_0 = sample_df_unexp_0[sample(1:sample_df_unexp_0_dim, dim_sampleB, replace=false ),:]
            dfd_sample = vcat(sample_df_exp,sample_df_unexp_1,new_df_unexp_0)
        end
    end 
    rows2remove = setdiff(dfd[dfd[:isO].==false, :panid],dfd_sample[:panid])
    #dfd[findin(dfd[:panid],rows2remove),:whyO]="NoMatch"
    dfd[findin(dfd[:panid],rows2remove),:isO]=true 
    return dfd[dfd[:isO].==false, : ]  #[setdiff(names(dfd),[:isO,:whyO])] 
end

dfd = MatchMe(dfd,cfg)

lowercase!(dfd)
cfg=lowercase(cfg)

#SAVE FILE
writetable(root*"/matched_dfd.csv", dfd)
#dfd = readtable(root*"/matched_dfd.csv",header=true);

######################################
#------------MODEL OBJECTS-----------#  [:fea_or_dis_trps_shr_dpp_p1,:fea_or_dis_trps_shr_dpp_p2,:fea_or_dis_trps_shr_dpp_p3,:fea_or_dis_trps_shr_dpp_p4]
######################################

iocc = Dict(:modelName=>:occ, :raneff=>cfg[:random_campaigns], :y_var=>:trps_pos_p1, :logvar=>:LOG_trps_pre_p1, :logvarOrig=>:trps_pre_p1 )
idolocc = Dict(:modelName=>:dolocc, :raneff=>cfg[:random_campaigns], :y_var=>:dol_per_trip_pos_p1, :logvar=>:LOG_dol_per_trip_pre_p1, :logvarOrig=>:dol_per_trip_pre_p1)
ipen = Dict(:modelName=>:pen, :raneff=>cfg[:random_campaigns], :y_var=>:buyer_pos_p1, :logvar=>:LOG_buyer_pre_p1, :logvarOrig=>:buyer_pre_p1 )

custom_vars = [:iso,:whyo,:data_nb_ne_b, :data_b_e_nb ,:pen_reduction,:occ_reduction,:dolocc_reduction]  
for m in [iocc,idolocc,ipen] dfd[m[:logvar]]=log(Array(dfd[m[:logvarOrig]]+1)) end

function genExcludeVars!(iocc::Dict,idolocc::Dict,ipen::Dict)  
    iocc[:exclude_vars] = vcat( custom_vars,  
                                [ :buyer_pos_p1, iocc[:logvarOrig]
                                  ,idolocc[:y_var],idolocc[:logvar],idolocc[:logvarOrig] 
                                 ,ipen[:y_var],ipen[:logvar],ipen[:logvarOrig] 
                               ]
                              )
    idolocc[:exclude_vars] = vcat( custom_vars,  
                                   [ :buyer_pos_p1, idolocc[:logvarOrig]
                                           ,iocc[:y_var],iocc[:logvar],iocc[:logvarOrig] 
                                           ,ipen[:y_var],ipen[:logvar],ipen[:logvarOrig] 
                                   ]
                                 )    
    
    ipen[:exclude_vars] = vcat( custom_vars,
                                [ ipen[:logvarOrig]
                                  ,iocc[:y_var],iocc[:logvar],iocc[:logvarOrig] 
                                  ,idolocc[:y_var],idolocc[:logvar],idolocc[:logvarOrig] 
                                ]
                                 )        
end
genExcludeVars!(iocc,idolocc,ipen)


function expandM(di::Dict)
    d2=deepcopy(di)
    if d2[:modelName]==:occ  
        d2[:dist] = Poisson()
        d2[:lnk] = LogLink()
        d2[:Buyer_Pos_P1_is1] = true
    end
    if d2[:modelName]==:dolocc  
        d2[:dist] = Gamma()
        d2[:lnk] = LogLink()
        d2[:Buyer_Pos_P1_is1] = true
    end
    if d2[:modelName]==:pen
        d2[:dist] = Bernoulli()
        d2[:lnk] = LogitLink()
        d2[:Buyer_Pos_P1_is1] = false
    end
    return d2
end
#expandM(t)




#function checksingularity(f::Formula, dfd::DataFrame, tolerance = 1.e-8)
#    mf = ModelFrame(f, dfd)
#    mm = ModelMatrix(mf)
#    qrf = qrfact!(mm.m, Val{true})
#    vals = abs.(diag(qrf[:R]))
#    firstbad = findfirst(x -> x < min(tolerance, 0.5) * vals[1], vals)
#    if firstbad == 0
#        return Symbol[]
#    end
#    mf.terms.terms[view(mm.assign[qrf[:p]], firstbad:length(vals))]
#end
"""
singularity_x = checksingularity(genF(m[:y_var],vars), dfd)

 if isa(e, Base.LinAlg.PosDefException)
                    v=this.vfactors[e.info-1]
                    push!(this.xvars,v)
                    println("!!! Multicollinearity, removing :",v,"~~~",e.info-1, "\n~~~",e)
                else
vcat(f.lhs,f.rhs.args[3:end])[error-1]


function checksingularity(form::Formula, data::DataFrame, tolerance = 1.e-8)
    mf = ModelFrame(form, data)
    mm = ModelMatrix(mf)
    qrf = qrfact!(mm.m, Val{true})
    vals = abs.(diag(qrf[:R]))
    firstbad = findfirst(x -> x < min(tolerance, 0.5) * vals[1], vals)
    if firstbad == 0
        return Symbol[]
    end
    mf.terms.terms[view(mm.assign[qrf[:p]], firstbad:length(vals))]
end

"""



function featureSelection(dfd::DataFrame, m::Dict)
    function rmVars(v::DataArray{Any}) rmVars(convert(Array{Symbol},v)) end
    function rmVars(v::Array{Any}) rmVars(convert(Array{Symbol},v)) end
    function rmVars(v::Array{Symbol})
        #v=setdiff(v,[:group])
        return setdiff(vars,v)  
    end        
    vars=setdiff(names(dfd),   vcat(m[:exclude_vars],m[:y_var],:panid,cfg[:random_demos],cfg[:random_campaigns],cfg[:scoring_vars]) )
    upperModName = uppercase( string(  m[:modelName]  )  )
    println( upperModName*" : SingleValue") #SingleValue
    removed_SingleLevelVars=FS_singleLevel(dfd,vars)
    vars = rmVars(removed_SingleLevelVars)
    println(upperModName*" : Singularity : "*string(genF(m[:y_var],vars))) # Singularity
    singularity_x = checksingularity(genF(m[:y_var],vars), dfd)
    vars = rmVars(singularity_x)
    println(upperModName*" : PVals") #PVals
    f=genF(m[:y_var],vars)
    g1 = glm( f, dfd[convert(Array{Symbol},vcat(f.lhs,f.rhs.args[3:end]))]  , m[:dist], m[:lnk] )
    sdf = coefDF(g1)
    g1_x= sdf[sdf[:pval].>0.7,:vars]
    vars = rmVars(g1_x)
    function chkSigns(m::Dict, vars::Array{Symbol}, dfd::DataFrame, cfg::OrderedDict)  # Pvalue & Signs
        vars=unique(vcat(vars,[:group]))
        f=genF(m[:y_var],vars)
        g = glm( f, dfd[convert(Array{Symbol},vcat(f.lhs,f.rhs.args[3:end]))]  , m[:dist], m[:lnk] )
        sdf = coefDF(g1)
        neutralvars = setdiff(vars,vcat(cfg[:negativevars],cfg[:positivevars])) 
        neg=intersect(cfg[:negativevars],sdf[sdf[:coef].<0,:vars])
        pos=intersect(cfg[:positivevars],sdf[sdf[:coef].>0,:vars])
        varstokeep = intersect(vcat(neutralvars, pos,neg) ,  sdf[sdf[:pval].<cfg[:pvalue_lvl] ,:vars] )
        varstokeep =  convert(Array{Symbol},varstokeep)
        return g, varstokeep
    end
    println(upperModName*" : SIGN Check 1") 
    (g3, initialvars) = chkSigns(m, vars, dfd, cfg)
    println(upperModName*" : SIGN Check 2") 
    (g4, vars_2) = chkSigns(m, initialvars, dfd, cfg)
    function getCorrVars(m::RegressionModel, dfd::DataFrame, vars_2::Array{Symbol})
        sdf = coefDF(m)
        vars_2 = convert(Array{Symbol}, setdiff(vars_2, filter(x-> typeof(dfd[x]) in [PooledDataArray{String,UInt8,1}], vars_2)) )
        rm_lst=Symbol[]
        if (length(vars_2) > 1) & (   length(getColswithType("num", dfd, vars_2 ) ) > 1  )
            stackdf = corDFD(dfd,vars_2)
            stackdf[:variable_pval] = [ sdf[sdf[:vars].==c,:pval][1]   for c in stackdf[:variable]]
            stackdf[:vars_pval] = [ sdf[sdf[:vars].==c,:pval][1]   for c in stackdf[:vars]] 
            stackdf[:most_Sig] = map((x,y) -> x < y ? "variable" : "vars" ,stackdf[:variable_pval],stackdf[:vars_pval])
     
            for row in eachrow(stackdf[(stackdf[:value].> 0.8) | (stackdf[:value].<-0.8),:])
                if row[:vars] == "group"
                    push!(rm_lst,row[:variable])
                elseif row[:variable] == "group"
                    push!(rm_lst,row[:vars])
                else
                    row[:most_Sig] == "variable" ? push!(rm_lst,row[:vars]) : push!(rm_lst,row[:variable])
                end
            end    
        end
        return rm_lst
    end
    println(upperModName*" : Correlation") 
    corrvars_x = getCorrVars(g4, dfd,vars_2  )
    vars_2 = setdiff(vars_2,corrvars_x)    
    (g5, finalvars) =  chkSigns(m, convert(Array{Symbol},vars_2), dfd, cfg)
    
    #VIF is only valid on multiple cols
    if length(finalvars) > 2     
        println( upperModName *" : Z & Vif") #Z & Vif
        f=genF(m[:y_var],finalvars)
        g2 = glm( f, dfd[convert(Array{Symbol},vcat(f.lhs,f.rhs.args[3:end]))]  , m[:dist], m[:lnk] )
        sdf=vifDF(g2)
        z = sdf[abs(sdf[:zval]).<1.96,:vars]
        v = sdf[ !DataArrays.isna(sdf[:vif])&(sdf[:vif].>15),:vars]
        g2_x =intersect(z,v)
        finalvars = setdiff(finalvars,g2_x)
        println("vif_vars: ",g2_x)
    end
    finalvars = setdiff(finalvars,[:group])# reorder for group
    finalvars = convert(Array{Symbol},vcat(finalvars,[:group]) )
    return finalvars
end


factor_cols=vcat( [ cfg[:proscore], :group, :panid], cfg[:allrandoms] )
for c in setdiff(factor_cols,[:panid, cfg[:proscore]]) #cfg[:random_campaigns]
    if !( typeof(dfd[c]) in [  DataArray{String,1}  ]) 
        println("converting to Strings : ", c," of type : ",typeof(dfd[c]))
        dfd[c] = map(x->string(x),dfd[c])
        dfd[c] = convert(Array{String},dfd[c]) 
    end
end
poolit!(dfd,factor_cols)


iocc[:finalvars] = featureSelection(dfd[(dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1),:], expandM(iocc))
idolocc[:finalvars]   = featureSelection(dfd[(dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1),:], expandM(idolocc))
ipen[:finalvars]  = featureSelection(dfd[(dfd[:iso].==false) ,:], expandM(ipen))

#m=expandM(ipen)
#glm(genF(m[:y_var],m[:finalvars]), dfd[vcat(m[:y_var],m[:finalvars])], m[:dist], m[:lnk])



# --- Removing outlliers - to ensure overall campaign uplift
function getOutliers(r::DataFrame, dfd::DataFrame, m::Dict, segment::Symbol,pct::Int64) # :data_nb_ne_b, :data_b_e_nb, :pen_reduction, :occ_reduction, :dolocc_reduction
    if m[:Buyer_Pos_P1_is1]
        println("pre=1")
        odf = dfd[(dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1)&(dfd[segment].==true),:panid]
    else
        println("pre!=1")
        odf = dfd[(dfd[:iso].==false)&(dfd[segment].==true),:panid]
    end
    pctnum = Integer(round(length(odf) / 100) * pct)
    return sort(r[findin(r[:panid],odf) ,:], cols=[order(:resids, rev = true)] )[1:pctnum,:panid]
end

function genGLM(dfd::DataFrame, m::Dict)
    f=genF(m[:y_var],m[:finalvars])
    println("GLM for ",m[:modelName]," ::: ",f)
    dfd_tmp = m[:modelName] in [:occ, :dolocc] ? dfd[(dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1),:] : dfd[(dfd[:iso].==false),:] 
    g = glm(f, dfd_tmp[vcat(m[:y_var],m[:finalvars])], m[:dist], m[:lnk])
    println("GLM Residuals for ",m[:modelName])
    r = DataFrame(panid=dfd_tmp[:panid], resids=StatLib.xResiduals(g))
    return r, g 
end

function genlists(dfd::DataFrame, iocc::Dict,idolocc::Dict,ipen::Dict)   
    r1, g1 = genGLM( dfd[(dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1),:], expandM(iocc))
    r2, g2 = genGLM( dfd[(dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1),:], expandM(idolocc))
    r3, g3 = genGLM( dfd[(dfd[:iso].==false),:], expandM(ipen))
    rlist = [r1,r2,r3]
    glist = [g1,g2,g3]
    return rlist, glist
end
rlist, glist = genlists(dfd,iocc,idolocc,ipen) 

for i in 1:5
    mlist = [expandM(iocc),expandM(idolocc),expandM(ipen)]
    #mliftβ = [ g.sdf[g.sdf[:vars].==:group,:coef][1] for g in glist]
    mliftβ = [ sdf[sdf[:vars].==:group,:coef][1] for sdf in [coefDF(m) for m in glist]]
    minidx = find(x->x==minimum(mliftβ),mliftβ)[1]
    if  mliftβ[minidx] < 0.0
        #mx=mlist[minidx]
        println("WRONG : ",mliftβ," for ",mlist[minidx][:modelName])
        panids = getOutliers(rlist[minidx], dfd, mlist[minidx], :data_nb_ne_b,5)
        dfd[findin(dfd[:panid],panids) ,:iso] = true
        rlist, glist = genlists(dfd,iocc,idolocc,ipen) 
    else
        println("All good : ",mliftβ)
        break
    end
end



# ---- SAVE RESULTS -----------
delete!(iocc, :exclude_vars)
delete!(idolocc, :exclude_vars)
delete!(ipen, :exclude_vars)

modelsDict = Dict()
modelsDict[:iocc] = iocc
modelsDict[:idolocc] = idolocc
modelsDict[:ipen] = ipen


function unpool(dfd::DataFrame)
    v_factors = Symbol[]
    for c in names(dfd)
        if typeof(dfd[c]) in [ PooledDataArray{String,UInt8,1} ]
            println("         Converting : ", c )
            push!(v_factors,c)
            dfd[c] = Array(dfd[c])
        end
    end
    return v_factors
end

modelsDict[:factors] = unpool(dfd) 

jmod_fname = root*"/dfd_model.json"
mod_fname = root*"/dfd_model.csv" 

writetable(root*"/dfd_postOutliers.csv", dfd)
cols = vcat([:panid, :iso, :group, :buyer_pos_p1],cfg[:random_campaigns])
cols = vcat(cols,iocc[:finalvars],[iocc[:y_var], iocc[:logvar]])
cols = vcat(cols,idolocc[:finalvars],[idolocc[:y_var], idolocc[:logvar]])
cols = unique(vcat(cols,ipen[:finalvars],[ipen[:y_var], ipen[:logvar]]) )
writetable(mod_fname, dfd[ (dfd[:iso].==false) ,cols])


#using JSON

function saveModels(d::Dict,fname::String)
    j = JSON.json([ d])
    open(fname,"w") do f
        write(f,j)
    end
end
saveModels(modelsDict, jmod_fname)


# ---- END SAVE RESULTS -------
"""


##j = readModels(jmod_fname)
##group & raneff


function raneffStats(raneff::Array{Symbol})
    #ranef = modelsDict[:ipen][:raneff]
    
end
"""









# ------------------------------------------------------------
# -------- RESTART FROM HERE ------------
# ------------------------------------------------------------

addprocs([("iriadmin@10.63.36.22", 1), ("iriadmin@10.63.36.23", 1)])
addprocs(2)
using DataStructures, DataArrays , DataFrames, StatsFuns, GLM , Distributions, MixedModels, StatsBase, JSON, StatLib
@everywhere using DataStructures, DataArrays , DataFrames, StatsFuns, GLM , Distributions, MixedModels, StatsBase, JSON, StatLib


#root=pwd()
#root = "/mnt/resource/analytics/Natural_choice_5_851_modeling"   
root="/mnt/resource/analytics/models/Jenny-o"
#root="/mnt/resource/analytics/scoring/Jennie-o_1_020916_Modeling"
root="/mnt/resource/analytics/models/NatChoice"
root="/mnt/resource/analytics/models/rev"
root="/mnt/resource/analytics/models/ALL#10"
root="/mnt/resource/analytics/models/ALL#11"
root="/mnt/resource/analytics/models/CDW#6"
root="/mnt/resource/analytics/models/HormelChili#8"
root="/mnt/resource/analytics/models/Rev#8"


jmod_fname = root*"/dfd_model.json"
mod_fname = root*"/dfd_model.csv" 








#  --- functions from StatLib ---
@everywhere function expandM(di::Dict)
    d2=deepcopy(di)
    if d2[:modelName]==:occ  
        d2[:dist] = Poisson()
        d2[:lnk] = LogLink()
        d2[:Buyer_Pos_P1_is1] = true
    end
    if d2[:modelName]==:dolocc  
        d2[:dist] = Gamma()
        d2[:lnk] = LogLink()
        d2[:Buyer_Pos_P1_is1] = true
    end
    if d2[:modelName]==:pen
        d2[:dist] = Bernoulli()
        d2[:lnk] = LogitLink()
        d2[:Buyer_Pos_P1_is1] = false
    end
    return d2
end



@everywhere function readModels(fname::String)
    if isfile(fname)
        d=Dict()
        j = JSON.parsefile(fname)[1]
        #for (key, value) in dict
        for (k, v) in j    
            d2=Dict()
            if Symbol(k)==:mocc  
                d2[:dist] = Poisson()
                d2[:lnk] = LogLink()
                d2[:Buyer_Pos_P1_is1] = true
            end
            if Symbol(k)==:mdolocc  
                d2[:dist] = Gamma()
                d2[:lnk] = LogLink()
                d2[:Buyer_Pos_P1_is1] = true
            end
            if Symbol(k)==:mpen  
                d2[:dist] = Bernoulli()
                d2[:lnk] = LogitLink()
                d2[:Buyer_Pos_P1_is1] = false
            end
            
            if typeof(v) == Array{Any,1}
                d[Symbol(k)] = [Symbol(vx) for vx in v]
            else
                for (k1, v1) in v
                    if typeof(v1) in [ Array{Any,1} ]
                        v2=[Symbol(v1x) for v1x in v1]
                    else
                        v2=Symbol(v1)
                    end
                    d2[Symbol(k1)] = v2
                end
                d[Symbol(k)] = d2
            end
        end
        return d
    else
        return "file not found!"
    end
end



@everywhere function poolit!(dfd::DataFrame, vars::Array{Symbol}=Symbol[])
    if length(vars) == 0   vars=names(dfd) end
    for (i, c) in enumerate(dfd.columns)
        if typeof(c) == DataVector{String}
            if dfd.colindex.names[i] in vars
                println("[",i,"]  : ",dfd.colindex.names[i])
                #unique(dfd[i])
                dfd[i] = pool(dfd[i])
            end
        end
        #typeof(c) == DataVector{String} && println("[",i,"]  : ",dfd.colindex.names[i])
    end
end

@everywhere  function genF(y::Symbol, iv::Array{Symbol},ranef::Array{Symbol}=Symbol[])  
        vars=setdiff(iv,vcat([y],ranef))
        f = string(y)*" ~ 1"
        f = length(vars) > 0 ?  f * reduce(*, [ " + "*  string(c) for c in unique(vars) ] ) : f
        f = length(ranef) > 0 ?  f * reduce(*, [ " + "*  "(1 | "*string(c)*")" for c in unique(ranef) ] )  : f
        eval(parse(f))
end

@everywhere function coefDF(m::RegressionModel)
    DataFrame(parameter=vcat([:intercept],m.mf.terms.terms)  #g.model.mm.assign
             , coef=coef(m)
             , stderr=stderr(m)
             , zval=coef(m)./stderr(m) 
             , pval= ccdf(FDist(1, dof_residual(m)), abs2(coef(m)./stderr(m))  )
             )
end


# -------------------------------------


@everywhere function runGlmm(mod_fname::String, jmod_fname::String, modelname::Symbol, ranef::Symbol=:empty)
    println("Loding data : ")
    dfd = readtable(mod_fname,header=true);
    modelsDict = readModels(jmod_fname)
    poolit!(dfd,modelsDict[:factors])
    m=modelsDict[modelname]
    #ranef = ranef==[:empty] ? m[:raneff]  : ranef
    if ranef==:empty
        v_out=Dict()
        for r in m[:raneff]
            v_out[r] = runGlmm(dfd, m, [r])
        end
        return v_out
    else
        return runGlmm(dfd, m, [ranef])
    end
end


@everywhere function runGlmm(dfd::DataFrame,m::Dict, ranef::Array{Symbol}=[:empty])
    function genFx(y::Symbol, iv::Array{Symbol},ranef::Array{Symbol})  
        vars=setdiff(iv,vcat([y],ranef,[:group]))
        eval(parse( string(y)*" ~ 1"* reduce(*, [ " + "*  string(c) for c in vars ] ) * reduce(*, [ " + "*  "(1 | "*string(c)*")" for c in ranef ] )  ) )
    end   
    ranef = ranef==[:empty] ? m[:raneff]  : ranef
    m=expandM(m)
    f=genFx(m[:y_var],m[:finalvars],ranef)
    cols = convert(Array{Symbol},vcat(m[:finalvars], [m[:y_var], m[:logvar]], ranef ))
    #runGlmm(dfd[ (dfd[:buyer_pos_p1].==1),:],expandM(iocc),[:creative] )
    println("Ok - Running ",m[:modelName]," - ",ranef," Glmm on proc ",myid()," with : ",f)
    if m[:Buyer_Pos_P1_is1]
        return gmm1 = fit!(glmm(f, dfd[(dfd[:buyer_pos_p1].==1),cols] , m[:dist], m[:lnk]))
    else
        return gmm1 = fit!(glmm(f, dfd[cols], m[:dist], m[:lnk]))
    end
end


@everywhere function genModelList(modelsDict::Dict)
    #for (k,v) in OrderedDict(m=>modelsDict[m][:raneff] for m in [:ipen,:idolocc,:iocc]) for r in v println(k,"-",r) end end
    q = Symbol[]
    cnt=0
    for (k,v) in OrderedDict(m=>modelsDict[m][:raneff] for m in [:ipen,:idolocc,:iocc]) 
        for r in v push!(q,k); push!(q,r); cnt=cnt+1 end 
    end
    #x = reshape(q, (2,2))
    x=reshape(q, (2,cnt))
    #x'
    permutedims(x, [2, 1])
end



# -----------------  CLUSTERED ---------------------------

function genM(dfd::DataFrame, modelsDict::Dict, includeResiduals::Bool=false)   
    function xResiduals(g::RegressionModel)
        resp = g.model.rr 
        sign(resp.y - resp.mu) .* sqrt(resp.devresid)
    end
    iocc = expandM(modelsDict[:iocc])
    idolocc = expandM(modelsDict[:idolocc])
    ipen = expandM(modelsDict[:ipen])
        
    glist = Dict()
    f=genF(ipen[:y_var],ipen[:finalvars])
    glist[:ipen] = glm(f, dfd[vcat(ipen[:y_var],ipen[:finalvars])], ipen[:dist], ipen[:lnk])
    if includeResiduals
        glist[:r_ipen] = DataFrame(panid=dfd[:panid], resids=xResiduals(glist[:ipen]))
    end
    
    dfd=dfd[(dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1),:]
    f=genF(iocc[:y_var],iocc[:finalvars])
    glist[:iocc] = glm(f, dfd[vcat(iocc[:y_var],iocc[:finalvars])], iocc[:dist], iocc[:lnk])
    if includeResiduals
        glist[:r_iocc] = DataFrame(panid=dfd[:panid], resids=xResiduals(glist[:iocc]))
    end    
    
    f=genF(idolocc[:y_var],idolocc[:finalvars])
    glist[:idolocc] = glm(f, dfd[vcat(idolocc[:y_var],idolocc[:finalvars])], idolocc[:dist], idolocc[:lnk])
    if includeResiduals
        glist[:r_idolocc] = DataFrame(panid=dfd[:panid], resids=xResiduals(glist[:idolocc]))
    end
    return glist
end    
    
    
   
function ranefMe(m::GeneralizedLinearMixedModel)
    DataFrame(parameter=levels(m.LMM.trms[1].f), coef=vec(ranef(m, named=true)'[1]), stderr=vec(condVar(m)[1]) )
end

function getranef(m::Dict)
    xdf = DataFrame(level=String[], ranef = Float64[], stderr = Float64[], name = Symbol[] )
    for (key,value) in m
        println(key)
        rdf = ranefMe(value)
        rdf[:name] = key
        xdf = vcat(xdf,rdf)
    end
    return xdf
end
    

    
    
    

function runSequentialModels(mod_fname::String, jmod_fname::String, shelf::OrderedDict)
    for v in values(shelf)
            v[:results] = runGlmm( mod_fname,jmod_fname, v[:model], v[:renef])            
    end
end

       
    
    
    
function processReady(shelf::OrderedDict) 
    for (k,v) in  filter((k,v)-> (v[:status]==:running)&(isready(v[:channel])==true) ,shelf)
        v[:results] = take!(v[:channel])   #fetch(v[:channel])
        v[:status] = :complete
        v[:worker] = 0
        println("Task Completed : ",v[:model]," ~ ",v[:renef])
        close(v[:channel])
    end
end
    
    
runningW(shelf::OrderedDict) = filter((k,v)-> v[:status]==:running ,shelf) 
hasRunning(shelf::OrderedDict) = length(runningW(shelf)) > 0 ? true : false 
readyW(shelf::OrderedDict) = filter((k,v)-> v[:status]==:ready ,shelf) 
hasReady(shelf::OrderedDict) = length(readyW(shelf)) > 0 ? true : false     
NotcompleteW(shelf::OrderedDict) = filter((k,v)-> v[:status]!=:complete ,shelf) 
isCompleteW(shelf::OrderedDict) = length(NotcompleteW(shelf)) == 0 ? true : false    

statusW(shelf::OrderedDict) =  [ string(v[:model])*":"*string(v[:renef])*"   status: "*string(v[:status]) for v in values(shelf)] 
    
    
    
    function runclusteredModels(mod_fname::String, jmod_fname::String, shelf::OrderedDict)
     wids = workers()    
     for v in values(shelf)  v[:status]=:ready; v[:worker]=0; v[:channel]=Channel(1) end 
     freeW() = setdiff(wids,[v[:worker] for (k,v) in filter((k,v)-> v[:status]==:running ,shelf)]) 
     hasfreeW() = length(freeW()) > 0 ? true : false
     nextFreeW() = length(freeW()) > 0 ? freeW()[1]   : NA
        
     function runTask(mkey::Symbol)
        wid = nextFreeW() 
        if isna(wid)
            return false
        else
            w = shelf[mkey]
            w[:status] = :running
            w[:worker] = wid
            @async put!(w[:channel], remotecall_fetch(runGlmm, wid, mod_fname, jmod_fname, w[:model], w[:renef] )) 
            return true
        end
     end

     waitforFreeWorker() = while !hasfreeW() println("waiting"); processReady(shelf); sleep(15) end
     @async for (k,v) in shelf
         waitforFreeWorker()
         println("running : ",k)
         runTask(k)
     end

end
        
    

    
function distributeDataset()
    #jmod_fname = root*"/dfd_model.json"
    #mod_fname
    #ppp="/mnt/resource/analytics/models/"; run(`ls $ppp`)
    ips = ["10.63.36.22","10.63.36.23"]
    for ip in ips
            cmd=`scp $root/dfd_model.* $ip:$root/`
            println(cmd)
            run(cmd)
        #run(`ssh $ip mkdir $root`)
        #run(`scp $root/dfd_model.* $ip:$root/`)
    end
end
        

macro checked_lib(libname, path)
    println("GTPath  : ",path,"  ~  ",libname)
    ((VERSION >= v"0.4.0-dev+3844" ? Base.Libdl.dlopen_e : Base.dlopen_e)(path) == C_NULL) && error("Unable to load \n\n$libname ($path)\n\nPlease re-run Pkg.build(package), and restart Julia.")
    quote const $(esc(libname)) = $path end
end

    
    
modelsDict = readModels(jmod_fname)       
ml = genModelList(modelsDict)
shelf = OrderedDict(Symbol(string(ml[i,:][1])*"_"*string(ml[i,:][2]))=> Dict{Symbol,Any}( :model=>ml[i,:][1],:renef=>ml[i,:][2]) for i in 1:length(ml[:,1]))   
        
function modelMe(mod_fname::String, jmod_fname::String, shelf::OrderedDict)
    if (length(workers())==1)&(workers()[1]==1)
        println("\n\nRUNNING SEQUENTIAL WITH 1 CORE\n\n")
        runSequentialModels(mod_fname, jmod_fname, shelf) 
        #gmm1 = fit!(glmm(form, dat, Bernoulli()), false, 0)
    else
        runclusteredModels(mod_fname, jmod_fname, shelf)
        #statusW(shelf)
        while !isCompleteW(shelf) println("Not Complete yet!"); sleep(5); if !hasReady(shelf) processReady(shelf) end end
        #shelf        
    end      
end         
modelMe(mod_fname, jmod_fname, shelf)
            
function consolidateModels(modelsDict::Dict, mod_fname::String, shelf::OrderedDict)
    for (k,v) in shelf
        v[:sdf]= ranefMe(v[:results])
        v[:sdf][:model] = v[:model]
        v[:sdf][:ranef] = v[:renef]
    end        
    rdf = vcat([v[:sdf]for v in values(shelf)])
    ranDF = deepcopy(rdf)
    #writetable(root*"/raneff_out_df.csv", rdf)
    rdf[:zval] = NA
    rdf[:pval] = NA
    rdf[:modelType] = "MixedModels"
    # rdf[(rdf[:model].==:ipen)&(rdf[:break].==:placement),:]
       
    dfd = readtable(mod_fname,header=true);
    glist = genM(dfd, modelsDict)
    go_pen=coefDF(glist[:ipen])
    go_dolocc=coefDF(glist[:idolocc])
    go_occ=coefDF(glist[:iocc])
    go_occ[:model] = "occ"
    go_dolocc[:model] = "dolocc"
    go_pen[:model] = "pen"
    xgo = vcat(go_occ,go_dolocc,go_pen)
    glmDF = deepcopy(xgo)
    #writetable(root*"/glm_out_df.csv", xgo)
    xgo[:ranef] = NA
    xgo[:modelType] = "Glm"
    campaign = vcat(rdf,xgo)
    return ranDF, glmDF, campaign
end
    
(ranDF, glmDF, campaign) = consolidateModels(modelsDict, mod_fname, shelf)
        
writetable(root*"/glm_out_df.csv", glmDF)
writetable(root*"/raneff_out_df.csv", ranDF)
writetable(root*"/campaign_df.csv", campaign)
        
        
        
        
        
 """       
#@less Base.LOAD_CACHE_PATH[1]
Base.LOAD_CACHE_PATH[1]="/mapr/mapr04p/analytics0001/analytic_users/jpkg/v0.5"
ENV["JULIA_PKGDIR"]  = "/mapr/mapr04p/analytics0001/analytic_users/jpkg"
push!(LOAD_PATH, "/mapr/mapr04p/analytics0001/analytic_users/jpkg/v0.5")
pop!(LOAD_PATH)
Pkg.init()
Pkg.add("MixedModels")
Pkg.add("DataStructures")
Pkg.add("DataArrays")
Pkg.add("DataFrames")
Pkg.add("StatsFuns")
Pkg.add("GLM")
Pkg.add("Distributions")
Pkg.add("StatsBase")
Pkg.add("NLopt")
Pkg.add("JSON")
Pkg.clone("https://github.com/MozzyTrd/StatLib.jl.git")
using DataStructures, DataArrays , DataFrames, StatsFuns, GLM , Distributions, MixedModels, StatsBase, JSON, StatLib
        
Pkg.add("HDF5")
Pkg.add("JavaCall") 
Pkg.add("JuMP")
        Pkg.clone("https://github.com/MozzyTrd/StatStack.jl.git")
Pkg.add("NLsolve")
Pkg.add("DBAPI")
Pkg.add("JavaCall")
Pkg.add("JDBC")
Pkg.add("HDF5")
Pkg.add("JLD")
Pkg.add("RCall")
Pkg.add("Feather")
Pkg.add("DecisionTree")
Pkg.add("XGBoost")
       
using HDF5, JavaCall, JuMP, StatStack, NLsolve, DBAPI, JavaCall, JDBC, HDF5, JLD, RCall, Feather, DecisionTree, XGBoost
        
        
"""     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


# --------------- SEQUENTIALLY -----------------
    


function genM(dfd::DataFrame, iocc::Dict,idolocc::Dict,ipen::Dict, includeResiduals::Bool=false)   
    function xResiduals(g::RegressionModel)
        resp = g.model.rr 
        sign(resp.y - resp.mu) .* sqrt(resp.devresid)
    end
    glist = Dict()
    f=genF(ipen[:y_var],ipen[:finalvars])
    glist[:ipen] = glm(f, dfd[vcat(ipen[:y_var],ipen[:finalvars])], ipen[:dist], ipen[:lnk])
    if includeResiduals
        glist[:r_ipen] = DataFrame(panid=dfd[:panid], resids=xResiduals(glist[:ipen]))
    end
    
    dfd=dfd[(dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1),:]
    f=genF(iocc[:y_var],iocc[:finalvars])
    glist[:iocc] = glm(f, dfd[vcat(iocc[:y_var],iocc[:finalvars])], iocc[:dist], iocc[:lnk])
    if includeResiduals
        glist[:r_iocc] = DataFrame(panid=dfd[:panid], resids=xResiduals(glist[:iocc]))
    end    
    
    f=genF(idolocc[:y_var],idolocc[:finalvars])
    glist[:idolocc] = glm(f, dfd[vcat(idolocc[:y_var],idolocc[:finalvars])], idolocc[:dist], idolocc[:lnk])
    if includeResiduals
        glist[:r_idolocc] = DataFrame(panid=dfd[:panid], resids=xResiduals(glist[:idolocc]))
    end
    return glist
end
    
    
function ranefMe(m::GeneralizedLinearMixedModel)
    DataFrame(level=levels(m.LMM.trms[1].f), ranef=vec(ranef(m, named=true)'[1]), stderr=vec(condVar(m)[1]) )
end

function getranef(m::Dict)
    xdf = DataFrame(level=String[], ranef = Float64[], stderr = Float64[], name = Symbol[] )
    for (key,value) in m
        println(key)
        rdf = ranefMe(value)
        rdf[:name] = key
        xdf = vcat(xdf,rdf)
    end
    return xdf
end
    
    
function runSequentialModels(modelsDict::Dict)
    mocc=Dict()
    for r in modelsDict[:iocc][:raneff]
        mocc[r] = runGlmm( mod_fname,jmod_fname, :iocc, r)
    end
    mdolocc=Dict()
    for r in modelsDict[:idolocc][:raneff]
       mdolocc[r] = runGlmm( mod_fname,jmod_fname, :idolocc, r)
    end
    mpen=Dict()
    for r in modelsDict[:ipen][:raneff]
        mpen[r] = runGlmm( mod_fname,jmod_fname, :ipen, r)
    end

    o_occ = getranef(mocc)
    o_dolocc = getranef(mdolocc)
    o_pen = getranef(mpen)
    o_occ[:model] = "occ"
    o_dolocc[:model] = "dolocc"
    o_pen[:model] = "pen"
    rdf = vcat(o_occ, o_dolocc, o_pen)
    writetable(root*"/raneff_out_df.csv", rdf)

    modelsDict = readModels(jmod_fname)
    dfd = readtable(mod_fname,header=true);
    dfd[:iso]=false
    iocc = modelsDict[:iocc]
    idolocc = modelsDict[:idolocc]
    ipen = modelsDict[:ipen]

     glist = genM(dfd,expandM(iocc),expandM(idolocc),expandM(ipen)) 
     go_pen=coefDF(glist[:ipen])
     go_dolocc=coefDF(glist[:idolocc])
     go_occ=coefDF(glist[:iocc])
     go_occ[:model] = "occ"
     go_dolocc[:model] = "dolocc"
     go_pen[:model] = "pen"
     xgo = vcat(go_occ,go_dolocc,go_pen)
     writetable(root*"/glm_out_df.csv", xgo)        
        
     return  xgo, rdf
end

modelsDict = readModels(jmod_fname)
runSequentialModels(modelsDict)    





        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
# --------------------------------------------------------------------    
        
ml = genModelList(modelsDict)
shelf = OrderedDict(Symbol(string(ml[i,:][1])*"_"*string(ml[i,:][2]))=> 
        Dict(:model=>ml[i,:][1],:renef=>ml[i,:][2],:status=>:ready, :worker=>0, :channel=>Channel(1)) for i in 1:length(ml[:,1]))
(length(workers())==1)&(workers()[1]==1)        
        


    
mout=Dict()
p=1
for r in modelsDict[:ipen][:raneff]
   p=p+1    
   mout[r] = Channel(1)
   @async put!(mout[r], remotecall_fetch(runGlmm,p, mod_fname, jmod_fname, :ipen, r))
end
[isready(mout[k]) for k in keys(mout)]
#  for in in 1:100000  println([isready(mout[k]) for k in keys(mout)]); println(now()); sleep(15) end
isready(c)    
    
    
    
    
    

    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

# ---------- NEW ---------------
#http://docs.julialang.org/en/release-0.5/stdlib/parallel/#Base.Future
#c = Channel(1)
#@async put!(c, remotecall_fetch(long_computation, p))
#isready(c)

shelf=Dict()
p=1
for r in modelsDict[:ipen][:raneff]
   p=p+1    
   shelf[r] = Channel(1)
   @async put!(shelf[r], remotecall_fetch(runGlmm,p, mod_fname, jmod_fname, :ipen, r))
end
[isready(shelf[k]) for k in keys(shelf)]
#  for in in 1:100000  println([isready(shelf[k]) for k in keys(shelf)]); println(now()); sleep(15) end
isready(c)


mpen=Dict()
p=1
for r in modelsDict[:ipen][:raneff]
   p=p+1
   @async mpen[r] = remotecall_fetch( runGlmm,p, mod_fname, jmod_fname, :ipen, r )
end
#@async xdocc[:publisher_fct] = remotecall_fetch( runGlmm,2, mod_fname, jmod_fname, :idolocc, :publisher1 )


#per Raneff
#@time mocc = runGlmm( mod_fname,jmod_fname, :iocc, :publisher1)

#@time mocc = runGlmm( mod_fname,jmod_fname, :iocc)
#@time mdolocc = runGlmm( mod_fname,jmod_fname, :idolocc)
#@time mpen = runGlmm( mod_fname,jmod_fname, :ipen)



"""
@everywhere function runmodels(mod_fname::String, jmod_fname::String, modelname::Symbol, rf::Symbol=:empty )
    println("Loding data : ")
    dfd = readtable(mod_fname,header=true);
    modelsDict = readModels(jmod_fname)
    m=expandM(modelsDict[modelname])
    #repool!(dfd,m[:raneff])
    poolit!(dfd,modelsDict[:factors])
    runmodels(dfd, m, rf ) 
end


@everywhere function runmodels(dfd::DataFrame, m::Dict, rf::Symbol=:empty)
    function genFx(y::Symbol, iv::Array{Symbol},ranef::Array{Symbol})  
        vars=setdiff(iv,vcat([y],ranef,[:group]))
        eval(parse( string(y)*" ~ 1"* reduce(*, [ " + "*  string(c) for c in vars ] ) * reduce(*, [ " + "*  "(1 | "*string(c)*")" for c in ranef ] )  ) )
    end
    v_out = OrderedDict()
    println("start glmm on : ",myid())
    if rf==:empty
        rarr = m[:raneff]
    else
        rarr = [rf]
    end
    for r in rarr
        f=genFx(m[:y_var],m[:finalvars],[r])
        println(f)
        if m[:Buyer_Pos_P1_is1]
            gdfd = dfd[ (dfd[:buyer_pos_p1].==1) ,convert(Array{Symbol},vcat(m[:finalvars], [m[:y_var], m[:logvar]],m[:raneff]  ))]
        else
            gdfd = dfd[convert(Array{Symbol},vcat(m[:finalvars], [m[:y_var], m[:logvar]],m[:raneff]  ))]
        end
        println("O.K. running glmm!! : ",names(gdfd))
        gmm1 = fit!(glmm(f, gdfd , m[:dist]  ,m[:lnk])  )
        
        v_out[r] = gmm1
    end
    #gout = runGlmm(fdfd, raneff, m)        
    #put!(ques[Symbol(m.modelName)],v_out)
    if length(v_out) == 1
        return v_out[[key for key in keys(v_out)][1]]
    else
        return v_out
    end
    println("ending runGLMM!!!")
end


# --- TEST ----
@time mocc = runmodels( mod_fname,jmod_fname, :iocc)
@time mdolocc = runmodels( mod_fname,jmod_fname, :idolocc)
@time mpen = runmodels( mod_fname,jmod_fname, :ipen)
"""


function ranefMe(m::GeneralizedLinearMixedModel)
    DataFrame(level=levels(m.LMM.trms[1].f), ranef=vec(ranef(m, named=true)'[1]), stderr=vec(condVar(m)[1]) )
end

function getranef(m::OrderedDict)
    xdf = DataFrame(level=String[], ranef = Float64[], stderr = Float64[], name = Symbol[] )
    for (key,value) in m
        println(key)
        rdf = ranefMe(value)
        rdf[:name] = key
        xdf = vcat(xdf,rdf)
    end
    return xdf
end

o_occ = getranef(mocc)
o_dolocc = getranef(mdolocc)
o_pen = getranef(mpen)
o_occ[:model] = "occ"
o_dolocc[:model] = "dolocc"
o_pen[:model] = "pen"
rdf = vcat(o_occ, o_dolocc, o_pen)
writetable(root*"/raneff_out_df.csv", rdf)


modelsDict = readModels(jmod_fname)
dfd = readtable(mod_fname,header=true);
dfd[:iso]=false
iocc = modelsDict[:iocc]
idolocc = modelsDict[:idolocc]
ipen = modelsDict[:ipen]
#rlist, glist = genlists(dfd,iocc,idolocc,ipen)


#function genGLM(dfd::DataFrame, m::Dict)
#    f=genF(m[:y_var],m[:finalvars])
#    g = glm(f, dfd[vcat(m[:y_var],m[:finalvars])], m[:dist], m[:lnk])
#    return g 
#end


function genM(dfd::DataFrame, iocc::Dict,idolocc::Dict,ipen::Dict, includeResiduals::Bool=false)   
    function xResiduals(g::RegressionModel)
        resp = g.model.rr 
        sign(resp.y - resp.mu) .* sqrt(resp.devresid)
    end
    glist = Dict()
    f=genF(ipen[:y_var],ipen[:finalvars])
    glist[:ipen] = glm(f, dfd[vcat(ipen[:y_var],ipen[:finalvars])], ipen[:dist], ipen[:lnk])
    if includeResiduals
        glist[:r_ipen] = DataFrame(panid=dfd[:panid], resids=xResiduals(glist[:ipen]))
    end
    
    dfd=dfd[(dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1),:]
    f=genF(iocc[:y_var],iocc[:finalvars])
    glist[:iocc] = glm(f, dfd[vcat(iocc[:y_var],iocc[:finalvars])], iocc[:dist], iocc[:lnk])
    if includeResiduals
        glist[:r_iocc] = DataFrame(panid=dfd[:panid], resids=xResiduals(glist[:iocc]))
    end    
    
    f=genF(idolocc[:y_var],idolocc[:finalvars])
    glist[:idolocc] = glm(f, dfd[vcat(idolocc[:y_var],idolocc[:finalvars])], idolocc[:dist], idolocc[:lnk])
    if includeResiduals
        glist[:r_idolocc] = DataFrame(panid=dfd[:panid], resids=xResiduals(glist[:idolocc]))
    end
    
    #f=genF(m[:y_var],m[:finalvars]); 
    #glist[] = glm(f, dfd[vcat(m[:y_var],m[:finalvars])], m[:dist], m[:lnk])
    
    
    #for m in [expandM(iocc),expandM(idolocc),expandM(ipen)]
    #    f=genF(m[:y_var],m[:finalvars])
    #    if m[:Buyer_Pos_P1_is1]
    #        dfd = dfd[(dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1),:]
    #    else
    #        dfd = dfd[(dfd[:iso].==false),:]
    #    end
    #    g = glm(f, dfd[vcat(m[:y_var],m[:finalvars])], m[:dist], m[:lnk])
    #    #r = DataFrame(panid=dfd[:panid], resids=xResiduals(g))
    #    glist[m[:modelName]] = g
    #end
    return glist
end
glist = genM(dfd,expandM(iocc),expandM(idolocc),expandM(ipen)) 


go_pen=coefDF(glist[:ipen])
go_dolocc=coefDF(glist[:idolocc])
go_occ=coefDF(glist[:iocc])
go_occ[:model] = "occ"
go_dolocc[:model] = "dolocc"
go_pen[:model] = "pen"
xgo = vcat(go_occ,go_dolocc,go_pen)
#writetable(root*"/raneff_out_df.csv", dx)
writetable(root*"/glm_out_df.csv", xgo)


#mocc = runmodels( mod_fname,modelsDict, :iocc)
#runmodels( mod_fname,modelsDict, :iocc)
#runmodels( mod_fname,modelsDict, :iocc)

docc = Dict()
#@async d[:iocc] = remotecall_fetch( runmodels,2, mod_fname, modelsDict, :iocc )
@async docc[:publisher_fct] = remotecall_fetch( runmodels,2, mod_fname, jmod_fname, :iocc, :publisher_fct )
#= runmodels( mod_fname,modelsDict, :iocc, :publisher_fct)

"""

begin
    d = Dict()
    @sync @async for (idx, pid) in enumerate(workers())
         d[idx] = remotecall_fetch(getipaddr,pid)
    end
end



begin
    a = cell(nworkers())
    @sync @async for (idx, pid) in enumerate(workers())
        a[idx] = remotecall_fetch(pid, sleep, 2)
    end
end

"""









function gethostworkers()
    d = Dict()
    @sync @async for (idx, pid) in enumerate(workers())
        d[idx] = remotecall_fetch(getipaddr,pid)
    end
    iout=Dict()
    for ip in [string(ip) for ip in unique(values(d))]
        a=Int64[]
        for (key, value) in d
            if string(value) == ip
                push!(a,key)
            end
        end    
        iout[ip] = sort(a)
    end
    return iout
end




# =====================================================================================================================================
# =====================================================================================================================================
# ---- MODELINg END ------
# =====================================================================================================================================
# =====================================================================================================================================
#ssh-keygen -t rsa
#cat .ssh/id_rsa.pub | ssh b@B 'cat >> .ssh/authorized_keys'
#chmod 700 .ssh
#chmod 640 .ssh/authorized_keys
# addprocs(["user@host"], tunnel=true, dir="~/julia-79599ada44/bin/")
# addprocs(["iriadmin@10.63.36.22"])
# addprocs(["iriadmin@10.63.36.23"])
# addprocs([("host1", :auto), ("host2", 5), "host3"]) will launch as many workers as cores on host1, 5 workers on host2 and a single worker on host3.
# addprocs([("iriadmin@10.63.36.22", :auto), ("iriadmin@10.63.36.23", 5)])
# addprocs([("iriadmin@10.63.36.22", :auto), ("iriadmin@10.63.36.23", :auto)])
#r = @spawnat 2 ENV

a=modDict[:mpen]
f=genFmula(a[:y_var], vcat(a[:finalvars],[:group]), a[:logvar])
dfd = readtable(mod_fname,header=true);
#g = glm(f,   dfd[(dfd[:buyer_pos_p1].==1),:] , dist, lnk )
g = glm(f,   dfd , a[:dist], a[:lnk] )
#g = glm(f,dfd[convert(Array{Symbol},vcat(f.lhs,f.rhs.args[3:end]))]  , dist, lnk )




























# --- clustering ---

@everywhere function runRemotemodels(fname::String, raneff::Array{Symbol} ,m::MModel)
    #fdfd = Feather.read(fname)
    dfd = readtable(fname,header=true);
    repool!(dfd,raneff)
    runmodels(dfd, raneff ,m)
end

function genQues()
    ques=OrderedDict()
    ques[:occ]=RemoteChannel(1)
    ques[:dolocc]=RemoteChannel(1)
    ques[:pen]=RemoteChannel(1)
    return ques
end
ques = genQues()



# ------------------------------------


@everywhere function runGlmm(dfd::DataFrame, raneff::Array{Symbol}, m::MModel)
    function genFmula(y::Symbol, iv::Array{Symbol},ranef::Array{Symbol})  
        vars=setdiff(iv,vcat([y],ranef))
        eval(parse( string(y)*" ~ 1"* reduce(*, [ " + "*  string(c) for c in vars ] ) * reduce(*, [ " + "*  "(1 | "*string(c)*")" for c in ranef ] )  ) )
    end
    v_out = OrderedDict()
    for r in cfg[:random_campaigns]
        #f = xgenFmula(m.y_var,m.finalvars,cfg[:random_campaigns]) 
        f = genFmula(m.y_var,m.finalvars,[r])
        println(f)
        if m.Buyer_Pos_P1_is1
    #        #gmm1 = fit!(glmm(f, dfd[ (dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1) ,convert(Array{Symbol},vcat(m.finalvars, [m.y_var, m.logvar],cfg[:random_campaigns]))] , m.dist  ,m.lnk)  ) 
            #gmm1 = fit!(glmm(f, dfd[ (dfd[:buyer_pos_p1].==1) ,convert(Array{Symbol},vcat(m.finalvars, [m.y_var, m.logvar],raneff  ))] , m.dist  ,m.lnk)  ) 
            dfd = dfd[ (dfd[:buyer_pos_p1].==1) ,convert(Array{Symbol},vcat(m.finalvars, [m.y_var, m.logvar],raneff  ))]
        else
    #        #gmm1 = fit!(glmm(f, dfd[ (dfd[:iso].==false) ,convert(Array{Symbol},vcat(m.finalvars, [m.y_var, m.logvar],cfg[:random_campaigns]))] , m.dist  ,m.lnk)  )
            #gmm1 = fit!(glmm(f, dfd[convert(Array{Symbol},vcat(m.finalvars, [m.y_var, m.logvar],raneff ))] , m.dist  ,m.lnk)  )
            dfd = dfd[convert(Array{Symbol},vcat(m.finalvars, [m.y_var, m.logvar],raneff  ))]
        end  
        println("O.K. running glmm!! : ",names(dfd))
        gmm1 = fit!(glmm(f, dfd , m.dist  ,m.lnk)  )
    #    v_out[r] = gmm1
    end
    return v_out
end

#runGlmm(dfd, mocc)

@everywhere function runmodels(fname::String, raneff::Array{Symbol} ,m::MModel,ques::OrderedDict)
    fdfd = Feather.read("dfd.feather")
    #poolit!(fdfd,raneff)
    println("start glmm on : ",myid())
    gout = runGlmm(fdfd, raneff, m)        
    put!(ques[Symbol(m.modelName)],gout)
    println("ending runGLMM!!!")
end

# --- Create Feather Files ----
#using Feather, CategoricalArrays
#@everywhere using Feather, CategoricalArrays
for c in cfg[:random_campaigns]
    dfd[c] = Array(dfd[c])
    #dfd[c] = categorical(dfd[c])
    factor_cols
end
cols = vcat(mocc.finalvars,mdolocc.finalvars, [:buyer_pos_p1, mdolocc.y_var, mdolocc.logvar, mocc.y_var, mocc.logvar],cfg[:random_campaigns])
cols = vcat(cols,mpen.finalvars,[mpen.y_var, mpen.logvar])
Feather.write("dfd.feather", dfd[ (dfd[:iso].==false) ,cols])
# --- END Feather Files ----

#test
take!(ques[:dolocc]); runmodels( "/mnt/resource/analytics/CDW5_792/dfd.feather", cfg[:random_campaigns] ,mdolocc,ques)


@async @spawnat 2 runmodels( "/mnt/resource/analytics/CDW5_792/dfd.feather", cfg[:random_campaigns] ,mdolocc,ques)


@async @spawnat 2 testid(ques)

#fetch(ques[:occ])
#take!(ques[:dolocc])




dfd2=dfd[ (dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1), convert(Array{Symbol},vcat(m.finalvars, [m.y_var, m.logvar],cfg[:random_campaigns]))]
trps_pos_p1 ~ 1 + prd_1_qty_pre + cpn_un_pre_p4 + (1 | publisher_fct1)
Poisson()
LogLink()

gmm1 = fit!(glmm(trps_pos_p1 ~ 1 + prd_1_qty_pre + cpn_un_pre_p4 + (1 | publisher_fct1), dfd[ (dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1),:] ,   Poisson()  ))


# ------------------------------------------------------------------------------------------------------------
# ---------- END END END -------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------







type xGLM
    vfactors::Vector{Symbol}
    fmula::DataFrames.Formula
    xvars::Vector{Symbol}   
    model::Any #DataFrameRegressionModel
    sdf::DataFrame
    wasSuccessful::Bool
    resids::DataFrame
 
    function xGLM(dfd::DataFrame, dist::Distribution, y_var::Symbol, logvar::Symbol , lnk::Link , vfactors::Array{Symbol} )  
        this=new()
        this.wasSuccessful=false
        this.xvars=Symbol[]
        this.vfactors=setdiff(vfactors, [y_var,logvar])
        for l in 1:30
            this.fmula = genFmula(y_var, this.vfactors, logvar  )
            try
                f=this.fmula
                this.model = glm(f,  dfd[convert(Array{Symbol},vcat(f.lhs,f.rhs.args[3:end]))]  , dist, lnk )
                
                this.resids = DataFrame(panid=dfd[:panid], resids=xResiduals(this.model))
                println("starting GLM")
                this.sdf = DataFrame(vars=vcat([:intercept],this.model.mf.terms.terms)  #g.model.mm.assign
                                     , coef=coef(this.model)
                                     , se=stderr(this.model)
                                     , zval=coef(this.model)./stderr(this.model) 
                                     ,pval= ccdf(FDist(1, dof_residual(this.model)), abs2(coef(this.model)./stderr(this.model))))   
                """
                g=mocc.glm1_pvals.model
                g.model.mm.assign
                g=m.glm1_pvals
                DataFrame(vars=vcat([:intercept],g.model.mf.terms.terms)
                          , coef=coef(g.model)
                          , se=stderr(g.model)
                          , zval=coef(g.model)./stderr(g.model) 
                          ,pval= ccdf(FDist(1, dof_residual(g.model)), abs2(coef(g.model)./stderr(g.model))))  
                
                
                featureSelection(  dfd[(dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1),:]  , mdolocc)
                g=mdolocc.glm1_pvals.model
                vfactors=vars
                vfactors=setdiff(vfactors, [m.y_var,m.logvar])
                f = genFmula(m.y_var, vfactors, m.logvar  )
                g = glm(f,  dfd[convert(Array{Symbol},vcat(f.lhs,f.rhs.args[3:end]))]  , m.dist, m.lnk )
                
                DataFrame(vars=vcat([:intercept],g.mf.terms.terms)
                          , coef=coef(g)
                          , se=stderr(g)
                          , zval=coef(g)./stderr(g) 
                          , pval= ccdf(FDist(1, dof_residual(g)), abs2(coef(g)./stderr(g))  )
                         )  
                
                
                
                """
                this.wasSuccessful=true
                break
            catch e
                if isa(e, Base.LinAlg.PosDefException)
                    v=this.vfactors[e.info-1]
                    push!(this.xvars,v)
                    println("!!! Multicollinearity, removing :",v,"~~~",e.info-1, "\n~~~",e)
                else
                    println("....",e)
                    break
                end
            end
        end
        return this 
    end
end
 





#function m2dict(m::MModel,raneff::Array{Symbol}, vars::Array{Symbol}=Symbol[] )
#    if length(vars)==0 vars=m.finalvars end
#    return Dict(:modelName=>m.modelName ,:raneff=>cfg[:random_campaigns], :y_var=>m.y_var, :logvar=>m.logvar, :finalvars=>vars )
#end
##m2dict(mocc,cfg[:random_campaigns])
#function m2Dict(mocc::MModel, mdolocc::MModel, mpen::MModel, raneff::Array{Symbol} )
#    return Dict(:mocc=>m2dict(mocc, raneff ),:mdolocc=>m2dict(mdolocc,raneff), :mpen=>m2dict(mpen,raneff))
#end
## m2dict(mocc,mdolocc,mpen  cfg[:random_campaigns] )



#iocc = Dict(:modelName=>"occ", :raneff=>cfg[:random_campaigns], :y_var=>cfg[:occ_y_var], :logvar=>cfg[:occ_logvar] )
#idocc = Dict(:modelName=>"dolocc", :raneff=>cfg[:random_campaigns], :y_var=>cfg[:dolocc_y_var], :logvar=>cfg[:dolocc_logvar] )
#ipen = Dict(:modelName=>"pen", :raneff=>cfg[:random_campaigns], :y_var=>cfg[:pen_y_var], :logvar=>cfg[:pen_logvar] )








function featureSelection(dfd::DataFrame, m::MModel)
    function rmVars(v::Array{Symbol})
        v=setdiff(v,[:group])
        return setdiff(vars,v)  
    end        
    
    custom_vars=[:dolocc_reduction,:occ_reduction,:pen_reduction,:data_b_e_nb,:data_nb_ne_b,:whyo,:iso]
    required_vars=vcat([m.y_var,:panid,m.logvar],cfg[:random_demos],cfg[:random_campaigns],cfg[:scoring_vars])
    vars=setdiff(vcat(m.vars,[m.logvar_colname]),vcat(required_vars,custom_vars))
    
    println(uppercase(mocc.modelName)*" : SingleValue") #SingleValue
    m.removed_SingleLevelVars=FS_singleLevel(dfd,vars)
    vars = rmVars(m.removed_SingleLevelVars)
    
    println(uppercase(mocc.modelName)*" : Singularity : "*string(genFmula(m.y_var,vars,m.logvar))) # Singularity
    m.singularity_x = checksingularity(genFmula(m.y_var,vars,m.logvar), dfd)
    vars = rmVars(m.singularity_x)
    
    println(uppercase(mocc.modelName)*" : PVals") #PVals
    m.glm1_pvals = iGLM(dfd, m.dist, m.y_var, m.logvar, m.lnk , vars  )  
    g1=m.glm1_pvals
    m.glm1_pvals_x=g1.sdf[g1.sdf[:pval].>0.7,:vars]
    vars = rmVars(m.glm1_pvals_x)
    
    println(uppercase(mocc.modelName)*" : Z & Vif") #Z & Vif
    m.glm2_ZnVIF = iGLM(dfd, m.dist, m.y_var, m.logvar, m.lnk ,vars  ) 
    g2=m.glm2_ZnVIF
    vif!(g2)
    z = g2.sdf[abs(g2.sdf[:zval]).<1.96,:vars]
    v = g2.sdf[ !DataArrays.isna(g2.sdf[:vif])&(g2.sdf[:vif].>15),:vars]
    m.glm2_ZnVIF_x =intersect(z,v)
    vars = rmVars(m.glm2_ZnVIF_x )


    function chkSigns(m::MModel, vars::Array{Symbol}, dfd::DataFrame, cfg::OrderedDict)  # Pvalue & Signs
        vars=unique(vcat(vars,[:group]))
        g = iGLM(dfd, m.dist, m.y_var, m.logvar, m.lnk , vars  )  
        neutralvars = setdiff(vars,vcat(cfg[:negativevars],cfg[:positivevars])) 
        neg=intersect(cfg[:negativevars],g.sdf[g.sdf[:coef].<0,:vars])
        pos=intersect(cfg[:positivevars],g.sdf[g.sdf[:coef].>0,:vars])
        varstokeep = intersect(vcat(neutralvars, pos,neg) ,  g.sdf[ g.sdf[:pval].<cfg[:pvalue_lvl] ,:vars] )
        return g, varstokeep
    end

    println(uppercase(mocc.modelName)*" : SIGN Check 1") 
    (m.glm3_PnSigns, initialvars) = chkSigns(m, vars, dfd, cfg)
    println(uppercase(mocc.modelName)*" : SIGN Check 2") 
    (m.glm4_PnSignsClean, vars_2) = chkSigns(m, convert(Array{Symbol},initialvars) , dfd, cfg)


    function getCorrVars(dfd::DataFrame, vars_2::Array{Symbol})
        rm_lst=Symbol[]
        if (length(vars_2) > 1) & (   length(getColswithType("num", dfd, convert(Array{Symbol},vars_2) ) ) > 1  )
            stackdf = corDFD(dfd,vars_2)
            stackdf[:variable_pval] = [ m.glm4_PnSignsClean.sdf[m.glm4_PnSignsClean.sdf[:vars].==c,:pval][1]   for c in stackdf[:variable]]
            stackdf[:vars_pval] = [ m.glm4_PnSignsClean.sdf[m.glm4_PnSignsClean.sdf[:vars].==c,:pval][1]   for c in stackdf[:vars]] 
            stackdf[:most_Sig] = map((x,y) -> x < y ? "variable" : "vars" ,stackdf[:variable_pval],stackdf[:vars_pval])
     
            for row in eachrow(stackdf[(stackdf[:value].> 0.8) | (stackdf[:value].<-0.8),:])
                if row[:vars] == "group"
                    push!(rm_lst,row[:variable])
                elseif row[:variable] == "group"
                    push!(rm_lst,row[:vars])
                else
                    row[:most_Sig] == "variable" ? push!(rm_lst,row[:vars]) : push!(rm_lst,row[:variable])
                end
            end    
        end
        return rm_lst
    end
    
    println(uppercase(mocc.modelName)*" : Correlation") 
    m.corrvars_x = getCorrVars(dfd,convert(Array{Symbol},setdiff(vars_2,factor_cols)))
    vars_2 = setdiff(vars_2,m.corrvars_x)
    
    (m.glm5, m.finalvars) =  chkSigns(m, convert(Array{Symbol},vars_2), dfd, cfg)
    
    println(uppercase(mocc.modelName)*" : Final Review") # Final Review:
    m.glm6_final = iGLM(dfd, m.dist, m.y_var, m.logvar, m.lnk , convert(Array{Symbol},vcat(m.finalvars,[:group]))  )
    #rename!(m.glm6_final.resids,:resids,Symbol(m.modelName*"_residual"))
    m.df_resid = m.glm6_final.resids
    return m.glm6_final
end








type MDolOcc <: MModel
    vars::Vector{Symbol}
    #finalvars::Vector{Symbol}
    #y_var::Symbol
    #dist::Distribution
    #lnk::Link
    exclude_vars::Vector{Symbol}
    removed_SingleLevelVars::Vector{Symbol}
    singularity_x::Vector{Symbol}
    glm1_pvals::iGLM
    glm1_pvals_x::Vector{Symbol}
    glm2_ZnVIF::iGLM
    glm2_ZnVIF_x::Vector{Symbol}
    glm3_PnSigns::iGLM
    glm3_PnSigns_x::Vector{Symbol}
    glm4_PnSignsClean::iGLM
    glm4_PnSignsClean_x::Vector{Symbol}
    glm5::iGLM
    glm5_Z_x::Vector{Symbol}
    corrvars_x::Vector{Symbol}
    glm6_final::iGLM
    Buyer_Pos_P1_is1::Bool
    modelName::String
    #logvar::Symbol
    #logvar_colname::Symbol
    #fdf::DataFrame
    #rdf::DataFrame
    #df_resid::DataFrame
    #groupDeviance::Float64
    function MDolOcc(dfd::DataFrame,cfg::OrderedDict=Dict()) 
        this=new(); this.modelName=:dolocc; 
        #this.logvar=cfg[:dolocc_logvar]; this.y_var=cfg[:dolocc_y_var]; this.dist=Gamma(); this.logvar_colname = cfg[:dolocc_logvar_colname]; this.lnk=LogLink()
        this.removed_SingleLevelVars=Symbol[]
        this.glm1_pvals_x=Symbol[]
        this.glm2_ZnVIF_x=Symbol[]
        this.glm3_PnSigns_x=Symbol[]
        this.glm4_PnSignsClean_x=Symbol[]
        this.corrvars_x=Symbol[]
        this.exclude_vars= Symbol[ cfg[:occ_y_var],cfg[:occ_logvar],cfg[:occ_logvar_colname],cfg[:pen_y_var],cfg[:pen_logvar],cfg[:pen_logvar_colname], :buyer_pos_p1 ]
        this.Buyer_Pos_P1_is1=true
        #dfd[this.logvar_colname] = log(Array(dfd[this.logvar]+1))
        this.vars=setdiff(names(dfd),vcat(this.exclude_vars,  [:iso,:whyo,:data_nb_ne_b, :data_b_e_nb ,:pen_reduction,:occ_reduction,:dolocc_reduction] ))
        this.vars=setdiff(this.vars,[this.logvar, this.logvar_colname])
        return this 
    end
end
mdolocc = MDolOcc(dfd,cfg)





type MOcc <: MModel
    vars::Vector{Symbol}
    #finalvars::Vector{Symbol}
    #y_var::Symbol
    #dist::Distribution
    #lnk::Link
    exclude_vars::Vector{Symbol}
    singularity_x::Vector{Symbol}
    removed_SingleLevelVars::Vector{Symbol}
    glm1_pvals::iGLM
    glm1_pvals_x::Vector{Symbol}
    glm2_ZnVIF::iGLM
    glm2_ZnVIF_x::Vector{Symbol}
    glm3_PnSigns::iGLM
    glm3_PnSigns_x::Vector{Symbol}
    glm4_PnSignsClean::iGLM
    glm4_PnSignsClean_x::Vector{Symbol}
    glm5::iGLM
    glm5_Z_x::Vector{Symbol}
    corrvars_x::Vector{Symbol}
    glm6_final::iGLM
    Buyer_Pos_P1_is1::Bool
    modelName::String
    #logvar::Symbol
    #logvar_colname::Symbol
    #fdf::DataFrame
    #rdf::DataFrame
    #df_resid::DataFrame
    #groupDeviance::Float64
    function MOcc(dfd::DataFrame,cfg::OrderedDict=Dict()) 
        this=new(); this.modelName=:occ; 
        #this.logvar=cfg[:occ_logvar]; this.y_var=cfg[:occ_y_var]; this.dist=Poisson(); this.logvar_colname = cfg[:occ_logvar_colname]; this.lnk=LogLink()
        this.removed_SingleLevelVars=Symbol[]
        this.glm1_pvals_x=Symbol[]
        this.glm2_ZnVIF_x=Symbol[]
        this.glm3_PnSigns_x=Symbol[]
        this.glm4_PnSignsClean_x=Symbol[]
        this.corrvars_x=Symbol[]
        this.exclude_vars=Symbol[cfg[:dolocc_y_var],cfg[:dolocc_logvar],cfg[:dolocc_logvar_colname],cfg[:pen_y_var],cfg[:pen_logvar],cfg[:pen_logvar_colname], :buyer_pos_p1 ]
        this.Buyer_Pos_P1_is1=true
        #dfd[this.logvar_colname] = log(Array(dfd[this.logvar]+1))
        this.vars=setdiff(names(dfd),vcat(this.exclude_vars,[:iso,:whyo,:data_nb_ne_b, :data_b_e_nb ,:pen_reduction,:occ_reduction,:dolocc_reduction]))
        this.vars=setdiff(this.vars,[this.logvar, this.logvar_colname])
        return this 
    end
end
mocc = MOcc(dfd,cfg)



type MPen <: MModel
    vars::Vector{Symbol}
    #finalvars::Vector{Symbol}
    #y_var::Symbol
    #dist::Distribution
    #lnk::Link
    exclude_vars::Vector{Symbol}
    singularity_x::Vector{Symbol}
    removed_SingleLevelVars::Vector{Symbol}
    glm1_pvals::iGLM
    glm1_pvals_x::Vector{Symbol}
    glm2_ZnVIF::iGLM
    glm2_ZnVIF_x::Vector{Symbol}
    glm3_PnSigns::iGLM
    glm3_PnSigns_x::Vector{Symbol}
    glm4_PnSignsClean::iGLM
    glm4_PnSignsClean_x::Vector{Symbol}
    glm5::iGLM
    glm5_Z_x::Vector{Symbol}
    corrvars_x::Vector{Symbol}
    glm6_final::iGLM
    Buyer_Pos_P1_is1::Bool
    modelName::String
    #logvar::Symbol
    #logvar_colname::Symbol
    #fdf::DataFrame
    #rdf::DataFrame
    #df_resid::DataFrame
    #groupDeviance::Float64
    function MPen(dfd::DataFrame,cfg::OrderedDict=Dict()) 
        this=new(); this.modelName=:pen; 
        #this.logvar=cfg[:pen_logvar]; this.y_var=cfg[:pen_y_var]; this.dist=Bernoulli() #Binomial(); this.logvar_colname = cfg[:pen_logvar_colname]; this.lnk=LogitLink()
        this.removed_SingleLevelVars=Symbol[]
        this.glm1_pvals_x=Symbol[]
        this.glm2_ZnVIF_x=Symbol[]
        this.glm3_PnSigns_x=Symbol[]
        this.glm4_PnSignsClean_x=Symbol[]
        this.corrvars_x=Symbol[]     
        this.exclude_vars=Symbol[cfg[:occ_y_var],cfg[:occ_logvar],cfg[:occ_logvar_colname],cfg[:dolocc_y_var],cfg[:dolocc_logvar],cfg[:dolocc_logvar_colname] ]
        this.Buyer_Pos_P1_is1=false
        #dfd[this.logvar_colname] = log(Array(dfd[this.logvar]+1))
        this.vars=setdiff(names(dfd),vcat(this.exclude_vars,[:iso,:whyo,:data_nb_ne_b, :data_b_e_nb ,:pen_reduction,:occ_reduction,:dolocc_reduction]))    
        this.vars=setdiff(this.vars,[this.logvar, this.logvar_colname])
        return this 
    end
end
mpen = MPen(dfd,cfg)





# =========================================================================
# ==== Backup feature selection ================


iocc = Dict(:modelName=>:occ, :raneff=>cfg[:random_campaigns], :y_var=>:trps_pos_p1, :logvar=>:LOG_trps_pre_p1, :logvarOrig=>:trps_pre_p1 )
idolocc = Dict(:modelName=>:dolocc, :raneff=>cfg[:random_campaigns], :y_var=>:dol_per_trip_pos_p1, :logvar=>:LOG_dol_per_trip_pre_p1, :logvarOrig=>:dol_per_trip_pre_p1)
ipen = Dict(:modelName=>:pen, :raneff=>cfg[:random_campaigns], :y_var=>:buyer_pos_p1, :logvar=>:LOG_buyer_pre_p1, :logvarOrig=>:buyer_pre_p1 )

custom_vars = [:iso,:whyo,:data_nb_ne_b, :data_b_e_nb ,:pen_reduction,:occ_reduction,:dolocc_reduction]  
for m in [iocc,idolocc,ipen] dfd[m[:logvar]]=log(Array(dfd[m[:logvarOrig]]+1)) end

function genExcludeVars!(iocc::Dict,idolocc::Dict,ipen::Dict)  
    iocc[:exclude_vars] = vcat( custom_vars,  
                                [ :buyer_pos_p1, iocc[:logvarOrig]
                                  ,idolocc[:y_var],idolocc[:logvar],idolocc[:logvarOrig] 
                                 ,ipen[:y_var],ipen[:logvar],ipen[:logvarOrig] 
                               ]
                              )
    idolocc[:exclude_vars] = vcat( custom_vars,  
                                   [ :buyer_pos_p1, idolocc[:logvarOrig]
                                           ,iocc[:y_var],iocc[:logvar],iocc[:logvarOrig] 
                                           ,ipen[:y_var],ipen[:logvar],ipen[:logvarOrig] 
                                   ]
                                 )    
    
    ipen[:exclude_vars] = vcat( custom_vars,
                                [ ipen[:logvarOrig]
                                  ,iocc[:y_var],iocc[:logvar],iocc[:logvarOrig] 
                                  ,idolocc[:y_var],idolocc[:logvar],idolocc[:logvarOrig] 
                                ]
                                 )        
end
genExcludeVars!(iocc,idolocc,ipen)


function expandM(di::Dict)
    d2=deepcopy(di)
    if d2[:modelName]==:occ  
        d2[:dist] = Poisson()
        d2[:lnk] = LogLink()
        d2[:Buyer_Pos_P1_is1] = true
    end
    if d2[:modelName]==:dolocc  
        d2[:dist] = Gamma()
        d2[:lnk] = LogLink()
        d2[:Buyer_Pos_P1_is1] = true
    end
    if d2[:modelName]==:pen
        d2[:dist] = Bernoulli()
        d2[:lnk] = LogitLink()
        d2[:Buyer_Pos_P1_is1] = false
    end
    return d2
end
#expandM(t)

type iGLM
    vars::Vector{Symbol}
    f::DataFrames.Formula
    g::Any #DataFrameRegressionModel
    sdf::DataFrame
    resids::DataFrame
    function iGLM(dfd::DataFrame, m::Dict, vars::Array{Symbol} )  
        this=new()
        this.vars=Symbol[] 
        m = expandM(m)
        vars=setdiff(vars, [m[:y_var],m[:logvar]])
        f = genFmula( m[:y_var], vars )
        g = glm(f, dfd[convert(Array{Symbol},vcat(f.lhs,f.rhs.args[3:end]))]  , m[:dist], m[:lnk] )    #g.model.mm.assign
        

        #resp = g.model.rr
        #resids = sign(resp.y - resp.mu) .* sqrt(resp.devresid)
        #resids = sign(g.model.rr.y - g.model.rr.mu) .* sqrt(g.model.rr.devresid)
        resp = g.model.rr
        devresid = sqrt(map(x->x < 0 ? 0.0:x, resp.devresid) ) # because we could do abs(), but we want to order by extream pos vals, so set to zero
        #resids = sign(resp.y - resp.mu) .* devresid
        #resids = DataFrame(panid=dfd[:panid], resids=xResiduals(g) )
        resids = DataFrame(panid=dfd[:panid], resids=sign(resp.y - resp.mu) .* devresid )
        sdf = DataFrame(vars=vcat([:intercept],g.mf.terms.terms)  #g.model.mm.assign
                                     , coef=coef(g)
                                     , se=stderr(g)
                                     , zval=coef(g)./stderr(g) 
                                     , pval= ccdf(FDist(1, dof_residual(g)), abs2(coef(g)./stderr(g))  )
                        ) 
        this.f = f
        this.g = g
        this.sdf = sdf
        this.resids = resids
        this.vars = vars
        return this 
    end
end
#g1 = iGLM(dfd, m, vars  )



function vif!(g::iGLM)
    vdf=vif(g.g)
    vdf[:vars] = convert(Array{Symbol}, vdf[:variable])
    g.sdf = join(g.sdf,vdf[[:vars,:vif]], on = :vars, kind=:outer)
end




function checksingularity(form::Formula, data::DataFrame, tolerance = 1.e-8)
    mf = ModelFrame(form, data)
    mm = ModelMatrix(mf)
    qrf = qrfact!(mm.m, Val{true})
    vals = abs.(diag(qrf[:R]))
    firstbad = findfirst(x -> x < min(tolerance, 0.5) * vals[1], vals)
    if firstbad == 0
        return Symbol[]
    end
    mf.terms.terms[view(mm.assign[qrf[:p]], firstbad:length(vals))]
end



# =======================================================================================
# =======================================================================================





function featureSelection(dfd::DataFrame, m::Dict)
    function rmVars(v::DataArray{Any}) rmVars(convert(Array{Symbol},v)) end
    function rmVars(v::Array{Any}) rmVars(convert(Array{Symbol},v)) end
    function rmVars(v::Array{Symbol})
        v=setdiff(v,[:group])
        return setdiff(vars,v)  
    end        
    vars=setdiff(names(dfd),   vcat(m[:exclude_vars],m[:y_var],:panid,cfg[:random_demos],cfg[:random_campaigns],cfg[:scoring_vars]) )

    upperModName = uppercase( string(  m[:modelName]  )  )
    println( upperModName*" : SingleValue") #SingleValue
    removed_SingleLevelVars=FS_singleLevel(dfd,vars)
    vars = rmVars(removed_SingleLevelVars)
    
    println(upperModName*" : Singularity : "*string(genFmula(m[:y_var],vars))) # Singularity
    singularity_x = checksingularity(genFmula(m[:y_var],vars), dfd)
    vars = rmVars(singularity_x)
    
    println(upperModName*" : PVals") #PVals
    g1 = iGLM(dfd, m, vars  )  
    g1_x=g1.sdf[g1.sdf[:pval].>0.7,:vars]
    vars = rmVars(g1_x)
    
    println( upperModName *" : Z & Vif") #Z & Vif
    g2 = iGLM(dfd, m, vars  ) 
    vif!(g2)
    z = g2.sdf[abs(g2.sdf[:zval]).<1.96,:vars]
    v = g2.sdf[ !DataArrays.isna(g2.sdf[:vif])&(g2.sdf[:vif].>15),:vars]
    g2_x =intersect(z,v)
    vars = rmVars(g2_x)


    function chkSigns(m::Dict, vars::Array{Symbol}, dfd::DataFrame, cfg::OrderedDict)  # Pvalue & Signs
        vars=unique(vcat(vars,[:group]))
        g = iGLM(dfd, m, vars)  
        neutralvars = setdiff(vars,vcat(cfg[:negativevars],cfg[:positivevars])) 
        neg=intersect(cfg[:negativevars],g.sdf[g.sdf[:coef].<0,:vars])
        pos=intersect(cfg[:positivevars],g.sdf[g.sdf[:coef].>0,:vars])
        varstokeep = intersect(vcat(neutralvars, pos,neg) ,  g.sdf[ g.sdf[:pval].<cfg[:pvalue_lvl] ,:vars] )
        return g, varstokeep
    end

    println(upperModName*" : SIGN Check 1") 
    (g3, initialvars) = chkSigns(m, vars, dfd, cfg)
    println(upperModName*" : SIGN Check 2") 
    (g4, vars_2) = chkSigns(m, convert(Array{Symbol},initialvars) , dfd, cfg)


    function getCorrVars(dfd::DataFrame, vars_2::Array{Symbol})
        rm_lst=Symbol[]
        if (length(vars_2) > 1) & (   length(getColswithType("num", dfd, convert(Array{Symbol},vars_2) ) ) > 1  )
            stackdf = corDFD(dfd,vars_2)
            stackdf[:variable_pval] = [ g4.sdf[g4.sdf[:vars].==c,:pval][1]   for c in stackdf[:variable]]
            stackdf[:vars_pval] = [ g4.sdf[g4.sdf[:vars].==c,:pval][1]   for c in stackdf[:vars]] 
            stackdf[:most_Sig] = map((x,y) -> x < y ? "variable" : "vars" ,stackdf[:variable_pval],stackdf[:vars_pval])
     
            for row in eachrow(stackdf[(stackdf[:value].> 0.8) | (stackdf[:value].<-0.8),:])
                if row[:vars] == "group"
                    push!(rm_lst,row[:variable])
                elseif row[:variable] == "group"
                    push!(rm_lst,row[:vars])
                else
                    row[:most_Sig] == "variable" ? push!(rm_lst,row[:vars]) : push!(rm_lst,row[:variable])
                end
            end    
        end
        return rm_lst
    end
    
    println(upperModName*" : Correlation") 
    corrvars_x = getCorrVars(dfd,convert(Array{Symbol},setdiff(vars_2,factor_cols)))
    vars_2 = setdiff(vars_2,corrvars_x)
    
    (g5, finalvars) =  chkSigns(m, convert(Array{Symbol},vars_2), dfd, cfg)
    m[:finalvars] = convert(Array{Symbol},finalvars)
    
    println(upperModName*" : Final Review") # Final Review:
    g6 = iGLM(dfd, m , convert(Array{Symbol},vcat(finalvars,[:group]))  )
    #rename!(m.glm6_final.resids,:resids,Symbol(m.modelName*"_residual"))
    #m.df_resid = m.glm6_final.resids
    return g6
end


factor_cols=vcat( [ cfg[:proscore], :group, :panid], cfg[:allrandoms] )

for c in setdiff(factor_cols,[:panid, cfg[:proscore]]) #cfg[:random_campaigns]
    if !( typeof(dfd[c]) in [  DataArray{String,1}  ]) 
        println("converting to Strings : ", c," of type : ",typeof(dfd[c]))
        dfd[c] = map(x->string(x),dfd[c])
        dfd[c] = convert(Array{String},dfd[c]) 
    end
end
poolit!(dfd,factor_cols)

 

g6_occ = featureSelection(dfd[(dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1),:], iocc)
g6_dolocc   = featureSelection(dfd[(dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1),:], idolocc)
g6_pen  = featureSelection(dfd[(dfd[:iso].==false) ,:], ipen)



# ======================================================================================


type iGLM
    vars::Vector{Symbol}
    f::DataFrames.Formula
    g::Any #DataFrameRegressionModel
    sdf::DataFrame
    resids::DataFrame
    function iGLM(dfd::DataFrame, m::Dict, vars::Array{Symbol} )  
        this=new()
        this.vars=Symbol[] 
        m = expandM(m)
        vars=setdiff(vars, [m[:y_var],m[:logvar]])
        f = genFmula( m[:y_var], vars )
        g = glm(f, dfd[convert(Array{Symbol},vcat(f.lhs,f.rhs.args[3:end]))]  , m[:dist], m[:lnk] )    #g.model.mm.assign
        

        #resp = g.model.rr
        #resids = sign(resp.y - resp.mu) .* sqrt(resp.devresid)
        #resids = sign(g.model.rr.y - g.model.rr.mu) .* sqrt(g.model.rr.devresid)
        resp = g.model.rr
        devresid = sqrt(map(x->x < 0 ? 0.0:x, resp.devresid) ) # because we could do abs(), but we want to order by extream pos vals, so set to zero
        #resids = sign(resp.y - resp.mu) .* devresid
        #resids = DataFrame(panid=dfd[:panid], resids=xResiduals(g) )
        resids = DataFrame(panid=dfd[:panid], resids=sign(resp.y - resp.mu) .* devresid )
        sdf = DataFrame(vars=vcat([:intercept],g.mf.terms.terms)  #g.model.mm.assign
                                     , coef=coef(g)
                                     , se=stderr(g)
                                     , zval=coef(g)./stderr(g) 
                                     , pval= ccdf(FDist(1, dof_residual(g)), abs2(coef(g)./stderr(g))  )
                        ) 
        this.f = f
        this.g = g
        this.sdf = sdf
        this.resids = resids
        this.vars = vars
        return this 
    end
end
#g1 = iGLM(dfd, m, vars  )



#include("/media/u01/analytics/RegTools/diagnostics.jl")
#include("/media/u01/analytics/RegTools/misc.jl")
#include("/media/u01/analytics/RegTools/modsel.jl")


function vif!(g::iGLM)
    vdf=vif(g.g)
    vdf[:vars] = convert(Array{Symbol}, vdf[:variable])
    g.sdf = join(g.sdf,vdf[[:vars,:vif]], on = :vars, kind=:outer)
end






# =======================================================================================
# =======================================================================================




"""

function vif(dfrm::RegressionModel)
    X = dfrm.mf.df[2:end]
    rhs = extractrhs(dfrm)
    result = DataFrame(variable=rhs.rhsarray, vif=0.0)
    i = 1
    for (var in rhs.rhsarray)
        lhs = parse(var)
        rhsnew = replace(rhs.rhsstring, "+"*var, "")
        rhsnew = rhsnew[2:end]
        rhsnew = parse(rhsnew)
        fnew = Formula(lhs, rhsnew)
        newfit = fit(LinearModel, fnew, X)
        r2 = rsquared(newfit)
        result[:vif][i] = 1 / (1-r2)
        i = i + 1
    end
    result
end


function rsquared(dfrm::RegressionModel)
    SStot = sum((dfrm.model.rr.y - mean(dfrm.model.rr.y)).^2)
    SSres = sum((dfrm.model.rr.y - dfrm.model.rr.mu).^2)
    return (1-(SSres/SStot))
end


julia> Pkg.status()
16 required packages:
 - DataFrames                    0.8.4
 - DataStructures                0.4.6
 - Distributions                 0.11.0
 - Feather                       0.2.1
 - GLM                           0.6.0
 - Gadfly                        0.5.1
 - HDF5                          0.6.6
 - JLD                           0.6.4
 - JSON                          0.8.0
 - JavaCall                      0.4.2
 - JuMP                          0.14.1
 - MixedModels                   0.5.7+             master
 - NLopt                         0.3.3
 - RCall                         0.5.2
 - StatsBase                     0.11.1
 - StatsFuns                     0.3.1
51 additional packages:
 - AxisAlgorithms                0.1.5
 - BinDeps                       0.4.5
 - Blosc                         0.1.7
 - Calculus                      0.1.15
 - CategoricalArrays             0.0.6
 - ColorTypes                    0.2.11
 - Colors                        0.6.9
 - Combinatorics                 0.3.2
 - Compat                        0.9.2
 - Compose                       0.4.4
 - Contour                       0.2.0
 - DataArrays                    0.3.8
 - DataStreams                   0.1.1
 - Distances                     0.3.2
 - FileIO                        0.2.0
 - FixedPointNumbers             0.2.1
 - FixedSizeArrays               0.2.4
 - FlatBuffers                   0.1.2
 - ForwardDiff                   0.2.5
 - GZip                          0.2.20
 - Hexagons                      0.0.4
 - Hiccup                        0.0.3
 - Interpolations                0.3.6
 - Iterators                     0.1.10
 - Juno                          0.2.3
 - KernelDensity                 0.3.0
 - Lazy                          0.11.4
 - LegacyStrings                 0.1.1
 - Loess                         0.0.7
 - MacroTools                    0.3.2
 - MathProgBase                  0.5.6
 - Measures                      0.0.3
 - Media                         0.2.3
 - NaNMath                       0.2.1
 - NamedArrays                   0.5.2
 - NullableArrays                0.0.10
 - Optim                         0.6.1
 - PDMats                        0.5.0
 - Polynomials                   0.1.0
 - PositiveFactorizations        0.0.2
 - Ratios                        0.0.4
 - Reexport                      0.0.3
 - RegTools                      0.0.0-             master (unregistered)
 - ReverseDiffSparse             0.5.8
 - Rmath                         0.1.3
 - SHA                           0.2.1
 - Showoff                       0.0.7
 - SortingAlgorithms             0.1.0
 - URIParser                     0.1.6
 - WeakRefStrings                0.2.0
 - WoodburyMatrices              0.2.0




"""
#    include("/home/iriadmin/.julia/v0.5/RegTools/src/diagnostics.jl")
#    include("/home/iriadmin/.julia/v0.5/RegTools/src/misc.jl")
#    include("/home/iriadmin/.julia/v0.5/RegTools/src/modsel.jl")



            
            

 function runclusteredModels(shelf::OrderedDict)
     wids = workers()   
    #ml = genModelList(modelsDict)
    #shelf = OrderedDict(Symbol(string(ml[i,:][1])*"_"*string(ml[i,:][2]))=> 
    #    Dict(:model=>ml[i,:][1],:renef=>ml[i,:][2],:status=>:ready, :worker=>0, :channel=>Channel(1)) for i in 1:length(ml[:,1]))
            
     for v in values(shelf)  v[:status]=:ready; v[:worker]=0; v[:channel]=Channel(1) end 
        
     function processReady(shelf::OrderedDict) 
    for (k,v) in  filter((k,v)-> (v[:status]==:running)&(isready(v[:channel])==true) ,shelf)
        v[:results] = take!(v[:channel])   #fetch(v[:channel])
        v[:status] = :complete
        v[:worker] = 0
        println("Task Completed : ",v[:model]," ~ ",v[:renef])
        close(v[:channel])
    end
     end
        
     freeW() = setdiff(wids,[v[:worker] for (k,v) in filter((k,v)-> v[:status]==:running ,shelf)]) 
     runningW() = filter((k,v)-> v[:status]==:running ,shelf) 
     hasfreeW() = length(freeW()) > 0 ? true : false
     nextFreeW() = length(freeW()) > 0 ? freeW()[1]   : NA
     hasRunning() = length(runningW()) > 0 ? true : false

     function runTask(mkey::Symbol)
        wid = nextFreeW() 
        if isna(wid)
            return false
        else
            w = shelf[mkey]
            w[:status] = :running
            w[:worker] = wid
            @async put!(w[:channel], remotecall_fetch(runGlmm, wid, mod_fname, jmod_fname, w[:model], w[:renef] )) 
            return true
        end
     end

     #gethostworkers()
     waitforFreeWorker() = while !hasfreeW() println("waiting"); processReady(shelf); sleep(15) end
     @async for (k,v) in shelf
         waitforFreeWorker()
         println("running : ",k)
         runTask(k)
     end
     #if hasRunning() processReady(); end
                   
    return shelf
end
        
    
