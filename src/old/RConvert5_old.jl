
using DataStructures, DataFrames, StatsFuns, GLM , JuMP,NLopt, HDF5, JLD, Distributions, MixedModels, RCall, StatsBase, xCommon

function loadDF()
    #cd("/media/u01/analytics/scoring/Healthy/modeling5/")
    #df_data = readtable("csv_final_healthychoice1_I3.csv",header=true); #lowercase(df_data)
    #df_h = readtable("Headers_healthy_choice3.csv",header=false); #lowercase(df_data)
    #names!(df_data, convert(Array{Symbol}, df_h[:x1]) ) 
    cd("/media/u01/analytics/scoring/CDW5_792/")
    df_data = readtable("csv_final_cdw5_792.csv",header=false);
    df_h = readtable("Headers_cdw5.csv",header=false);
    names!(df_data, convert(Array{Symbol}, df_h[:x1]) )
end
df_in=loadDF()
    


function isValid(df_data::DataFrame,cfg::OrderedDict)
    function checkValid(iarr::Array{Symbol})  length(setdiff(iarr, names(df_data))) > 0 ? false : true end
    !checkValid(cfg[:all_mandatory_vars]) ? error("ERROR: Not all mandatory_vars in dataset ") : println("VALID : mandatory_vars") 
    !checkValid(cfg[:scoring_vars]) ? error("ERROR: Not all scoring_vars in dataset ") : println("VALID : scoring_vars") 
end



"""
    cfg[:occ_y_var] = :Trps_POS_P1
    cfg[:occ_logvar] = :Trps_PRE_P1
    cfg[:dolocc_y_var] = :Dol_per_Trip_POS_P1
    cfg[:dolocc_logvar] = :Dol_per_Trip_PRE_P1
    cfg[:pen_y_var] = :Buyer_Pos_P1
    cfg[:pen_logvar] = :Buyer_Pre_P1
"""
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
                       )


function getCFG(df_in::DataFrame)
    cfg=xCommon.loadCFG(cfgDefaults, pwd()*"/app.cfg")
    cfg[:exposed_flag_var] = :exposed_flag_new                  # Go In app.cfg
    cfg[:random_campaigns] = [:Publisher_Fct1,:Targeting_Fct1]  # Go In app.cfg

    cfg[:allrandoms] = vcat(cfg[:random_demos],cfg[:random_campaigns])
    #cfg[:ProScore] = grep1("MODEL",names(df_in))   
    ps = filter(x->contains(string(x), "MODEL"), names(df_in)) # Set the ProScore variable in the dataset   #push!(cfg[:all_mandatory_vars], cfg[:ProScore])
    cfg[:ProScore] = length(ps) > 0 ? ps[1] : :MISSING_MODEL_VARIABLE_IN_DATA
    cfg[:num_products] = length( grep("Buyer_Pre_P",names(df_in)))-1  # get number of products in the data
    #rework vars after loading cfg
    cfg[:random_campaigns] = intersect(cfg[:random_campaigns],names(df_in))
 
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
    xflags=[cfg[:exposed_flag_var],:exposed_flag]
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
    dfd[:whyO]="" 
    
    dfd[dfd[:number_of_children_in_living_Un].>=4,:number_of_children_in_living_Un] = 4  # aggregate #of children for 4+ L_114
    #vars = getvars()   # :GROUP
    #vars = DataFrame(names=names(dfd),eltypes=eltypes(dfd))
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
        try dfd[c] = map( x -> DataArrays.isna(x) ?  NaN : convert(Float64, x)  , dfd[c]) catch e println("  (failed)") end  #NA to NaN
        try dfd[c] = convert(Array{Float64}, dfd[c]) catch e end   # To Float64
    end
    vars = DataFrame(names=names(dfd),eltypes=eltypes(dfd))
    #vars = getvars()
    for c in setdiff(vars[findin(vars[:eltypes],[Float64]),:names],cfg[:allrandoms])  # replace NaN's with zero for numeric variables 
        println("Replace Float64 NaN (0.0): ",c)
        dfd[ DataArrays.isnan(dfd[c]), c] = 0.0
    end
    for c in  setdiff(vars[findin(vars[:eltypes],[Int64]),:names],cfg[:allrandoms])    
        println("Replace Int64 NaN (0) : ",c)
        dfd[ DataArrays.isnan(dfd[c]), c] = 0
    end
    
    
    # create factor variables -  Need to understand how this impacts the data passed to R - might need to do this in R 
    #pool!(dfd, cfg[:allrandoms])   
    #pool!(dfd, [,:group]) 
   
    # NOTE : Do QCs here
    dfd[dfd[:person_1_gender].=="U",:isO] = true   # remove HHs with no gender info
    dfd[dfd[:person_1_gender].=="U",:whyO] = "person_1_gender=U; remove HHs with no gender info"
    dfd[findin(dfd[:estimated_hh_income],["U","L"]),:estimated_hh_income]="L" # aggregate U and L levels of hh income

    for r in cfg[:random_campaigns]    # check and drop exposed HHs with no publisher info or non-exposed HHs with publisher info
        dfd[findin(dfd[r],["\\N","NULL","0","NONE"])  ,r] ="none"
        dfd[ (dfd[:isO].==false) & (dfd[r].!="none") & (dfd[:group].==0) ,:whyO] = "non exposed HHs with publisher info"
        dfd[ (dfd[:isO].==false) & (dfd[r].!="none") & (dfd[:group].==0) ,:isO] = true
        println(r," non exposed HHs with publisher info : ",nrow(dfd[dfd[:whyO].=="non exposed HHs with publisher info",:] )   )
        dfd[ (dfd[:isO].==false) & (dfd[r].=="none") & (dfd[:group].!=0) ,:whyO] = "exposed HHs with no publisher info"   
        dfd[ (dfd[:isO].==false) & (dfd[r].=="none") & (dfd[:group].!=0) ,:isO] = true
        println(r," exposed HHs with no publisher info : ",nrow(dfd[dfd[:whyO].=="exposed HHs with no publisher info",:] )   )
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
    #df_sample=DataFrame
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
    
"""
    #Generate Report!!!! ---   still need to remove :isO
    function dict2df(d::Dict,kc::String,c::String)
        dfo=DataFrame(k=Int64[],v=Int64[])
        for (key, value) in d
            push!(dfo,[key, value])
        end
        rename!(dfo,:k,Symbol(kc))
        rename!(dfo,:v,Symbol(c))
        return dfo
    end
    ds0=dict2df(countmap(dfd_sample[ dfd_sample[:group].==0,cfg[:ProScore]]),"ProScore","out_unexp")
    ds1=dict2df(countmap(dfd_sample[ dfd_sample[:group].==1,cfg[:ProScore]]),"ProScore","out_exp")
    do0=dict2df(countmap(df[ df[:group].==0,cfg[:ProScore]]),"ProScore","in_unexp")
    do1=dict2df(countmap(df[ df[:group].==1,cfg[:ProScore]]),"ProScore","in_exp")
    dfo=join(join(join(do0, do1, on = :ProScore,kind=:outer), ds0, on = :ProScore,kind=:outer),ds1, on = :ProScore,kind=:outer)
    for c in [:out_unexp, :out_exp, :in_unexp, :in_exp]
        dfo[DataArrays.isna(dfo[c]), c] = 0
    end
    b_ds0=dict2df(countmap(dfd_sample[ dfd_sample[:group].==0,:Buyer_Pre_P1]),"Buyer_Pre_P1","out_unexp")
    b_ds1=dict2df(countmap(dfd_sample[ dfd_sample[:group].==1,:Buyer_Pre_P1]),"Buyer_Pre_P1","out_exp")
    b_do0=dict2df(countmap(dfd[ dfd[:group].==0,:Buyer_Pre_P1]),"Buyer_Pre_P1","in_unexp")
    b_do1=dict2df(countmap(dfd[ dfd[:group].==1,:Buyer_Pre_P1]),"Buyer_Pre_P1","in_exp")
    dfx=DataFrame(Buyer_Pre_P1=[0,1],Buyer_Pre_Period=["No","Yes"])
    b_dfo1=join(b_do0, b_do1, on = :Buyer_Pre_P1,kind=:outer)
    b_dfo1[:in_proportion] = b_dfo1[:in_exp]./sum(b_dfo1[:in_exp])
    b_dfo2=join(b_ds0, b_ds1, on = :Buyer_Pre_P1,kind=:outer)
    b_dfo2[:out_proportion] =b_dfo2[:out_exp]./sum(b_dfo2[:out_exp])
    b_dfo=join(dfx,join(b_dfo1, b_dfo2, on = :Buyer_Pre_P1,kind=:outer), on = :Buyer_Pre_P1,kind=:outer)
    delete!(b_dfo, :Buyer_Pre_P1 )
    #still need to Write QC to file
"""
    rows2remove = setdiff(dfd[dfd[:isO].==false, :panid],dfd_sample[:panid])
    dfd[findin(dfd[:panid],rows2remove),:whyO]="NoMatch"
    dfd[findin(dfd[:panid],rows2remove),:isO]=true 
    
    return dfd[dfd[:isO].==false, : ]  #[setdiff(names(dfd),[:isO,:whyO])] 
end

dfd = MatchMe(dfd,cfg)

lowercase!(dfd)
cfg=lowercase(cfg)

#Not Used
dfd=dfd[setdiff(names(dfd),[:nonbuyer_pre_p1])]



######################################
#------------MODEL OBJECTS-----------#
######################################
abstract MModel 

#!!! Multicollinearity, removing :[:fea_or_dis_trps_shr_dpp_p1,:fea_or_dis_trps_shr_dpp_p2,:fea_or_dis_trps_shr_dpp_p3,:fea_or_dis_trps_shr_dpp_p4]
type xGLM
    vfactors::Vector{Symbol}
    fmula::DataFrames.Formula
    xvars::Vector{Symbol}   
    model::Any #DataFrameRegressionModel
    sdf::DataFrame
    wasSuccessful::Bool
 
    function xGLM(dfd::DataFrame, dist::Distribution, y_var::Symbol, logvar::Symbol , lnk::Link , vfactors::Array{Symbol} )  
        this=new()
        this.wasSuccessful=false
        this.xvars=Symbol[]
        this.vfactors=setdiff(vfactors, [y_var,logvar])
        for l in 1:30
            #log_colname=Symbol("LOG_"*string(logvar))
            #this.vfactors=vcat(setdiff(this.vfactors,this.xvars),[log_colname])
            this.fmula = genFmula(y_var, this.vfactors, logvar  )
            try
                #this.model = glm(this.fmula,  dfd[vcat(this.vfactors,[y_var])], dist, LogLink(), offset=log(Array(dfd[logvar]  + 1)) )
                #dfd[log_colname]=log(Array(dfd[logvar]+1))
                f=this.fmula
                ##this.model = glm(this.fmula,  dfd[vcat(this.vfactors,[y_var,log_colname])], dist, m.lnk )   # LogLink()
                this.model = glm(f,  dfd[convert(Array{Symbol},vcat(f.lhs,f.rhs.args[3:end]))]  , dist, lnk )
                
                this.sdf = DataFrame(vars=vcat([:intercept],this.model.mf.terms.terms), coef=coef(this.model), se=stderr(this.model), zval=coef(this.model)./stderr(this.model) ,pval= ccdf(FDist(1, df_residual(this.model)), abs2(coef(this.model)./stderr(this.model))))            
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




type MDolOcc <: MModel
    vars::Vector{Symbol}
    finalvars::Vector{Symbol}
    y_var::Symbol
    dist::Distribution
    lnk::Link
    #mandatory_vars::Vector{Symbol}
    exclude_vars::Vector{Symbol}
    removed_SingleLevelVars::Vector{Symbol}
    singularity_x::Vector{Symbol}
    glm1_pvals::xGLM
    glm1_pvals_x::Vector{Symbol}
    glm2_ZnVIF::xGLM
    glm2_ZnVIF_x::Vector{Symbol}
    glm3_PnSigns::xGLM
    glm3_PnSigns_x::Vector{Symbol}
    glm4_PnSignsClean::xGLM
    glm4_PnSignsClean_x::Vector{Symbol}
    glm5::xGLM
    glm5_Z_x::Vector{Symbol}
    corrvars_x::Vector{Symbol}
    glm6_final::xGLM
    Buyer_Pos_P1_is1::Bool
    modelName::String
    logvar::Symbol
    function MDolOcc(dfd::DataFrame,cfg::OrderedDict=Dict()) 
        this=new(); this.modelName="dolocc"; this.logvar=cfg[:dolocc_logvar]; this.y_var=cfg[:dolocc_y_var]; this.dist=Gamma()
        this.lnk=LogLink()
        #this.mandatory_vars=Symbol[this.y_var]
        this.removed_SingleLevelVars=Symbol[]
        this.glm1_pvals_x=Symbol[]
        this.glm2_ZnVIF_x=Symbol[]
        this.glm3_PnSigns_x=Symbol[]
        this.glm4_PnSignsClean_x=Symbol[]
        this.corrvars_x=Symbol[]
        this.exclude_vars= Symbol[ cfg[:occ_y_var],cfg[:occ_logvar],cfg[:pen_y_var],cfg[:pen_logvar], :buyer_pos_p1 ]
        this.Buyer_Pos_P1_is1=true
        this.vars=setdiff(names(dfd),vcat(this.exclude_vars,[:iso,:whyo,:data_nb_ne_b, :data_b_e_nb ,:pen_reduction,:occ_reduction,:dolocc_reduction]))
        return this 
    end
end
mdolocc = MDolOcc(dfd,cfg)





type MOcc <: MModel
    vars::Vector{Symbol}
    finalvars::Vector{Symbol}
    y_var::Symbol
    dist::Distribution
    lnk::Link
    #mandatory_vars::Vector{Symbol}
    exclude_vars::Vector{Symbol}
    singularity_x::Vector{Symbol}
    removed_SingleLevelVars::Vector{Symbol}
    glm1_pvals::xGLM
    glm1_pvals_x::Vector{Symbol}
    glm2_ZnVIF::xGLM
    glm2_ZnVIF_x::Vector{Symbol}
    glm3_PnSigns::xGLM
    glm3_PnSigns_x::Vector{Symbol}
    glm4_PnSignsClean::xGLM
    glm4_PnSignsClean_x::Vector{Symbol}
    glm5::xGLM
    glm5_Z_x::Vector{Symbol}
    corrvars_x::Vector{Symbol}
    glm6_final::xGLM
    Buyer_Pos_P1_is1::Bool
    modelName::String
    logvar::Symbol
    function MOcc(dfd::DataFrame,cfg::OrderedDict=Dict()) 
        this=new(); this.modelName="occ"; this.logvar=cfg[:occ_logvar]; this.y_var=cfg[:occ_y_var]; this.dist=Poisson()
        this.lnk=LogLink()
        #this.mandatory_vars=Symbol[this.y_var]
        this.removed_SingleLevelVars=Symbol[]
        this.glm1_pvals_x=Symbol[]
        this.glm2_ZnVIF_x=Symbol[]
        this.glm3_PnSigns_x=Symbol[]
        this.glm4_PnSignsClean_x=Symbol[]
        this.corrvars_x=Symbol[]
        this.exclude_vars=Symbol[cfg[:dolocc_y_var],cfg[:dolocc_logvar],cfg[:pen_y_var],cfg[:pen_logvar], :buyer_pos_p1 ]
        this.Buyer_Pos_P1_is1=true
        this.vars=setdiff(names(dfd),vcat(this.exclude_vars,[:iso,:whyo,:data_nb_ne_b, :data_b_e_nb ,:pen_reduction,:occ_reduction,:dolocc_reduction]))
        return this 
    end
end
mocc = MOcc(dfd,cfg)



type MPen <: MModel
    vars::Vector{Symbol}
    finalvars::Vector{Symbol}
    y_var::Symbol
    dist::Distribution
    lnk::Link
    #mandatory_vars::Vector{Symbol}
    exclude_vars::Vector{Symbol}
    singularity_x::Vector{Symbol}
    removed_SingleLevelVars::Vector{Symbol}
    glm1_pvals::xGLM
    glm1_pvals_x::Vector{Symbol}
    glm2_ZnVIF::xGLM
    glm2_ZnVIF_x::Vector{Symbol}
    glm3_PnSigns::xGLM
    glm3_PnSigns_x::Vector{Symbol}
    glm4_PnSignsClean::xGLM
    glm4_PnSignsClean_x::Vector{Symbol}
    glm5::xGLM
    glm5_Z_x::Vector{Symbol}
    corrvars_x::Vector{Symbol}
    glm6_final::xGLM
    Buyer_Pos_P1_is1::Bool
    modelName::String
    logvar::Symbol
    function MPen(dfd::DataFrame,cfg::OrderedDict=Dict()) 
        this=new(); this.modelName="pen"; this.logvar=cfg[:pen_logvar]; this.y_var=cfg[:pen_y_var]; this.dist=Bernoulli() #Binomial()
        this.lnk=LogitLink()
        #this.mandatory_vars=Symbol[this.y_var]
        this.removed_SingleLevelVars=Symbol[]
        this.glm1_pvals_x=Symbol[]
        this.glm2_ZnVIF_x=Symbol[]
        this.glm3_PnSigns_x=Symbol[]
        this.glm4_PnSignsClean_x=Symbol[]
        this.corrvars_x=Symbol[]     
        this.exclude_vars=Symbol[cfg[:occ_y_var],cfg[:occ_logvar],cfg[:dolocc_y_var],cfg[:dolocc_logvar] ]
        this.Buyer_Pos_P1_is1=false
        this.vars=setdiff(names(dfd),vcat(this.exclude_vars,[:iso,:whyo,:data_nb_ne_b, :data_b_e_nb ,:pen_reduction,:occ_reduction,:dolocc_reduction]))    
        return this 
    end
end
mpen = MPen(dfd,cfg)



    include("/home/rmadmin/.julia/v0.4/RegTools/src/diagnostics.jl")
    include("/home/rmadmin/.julia/v0.4/RegTools/src/misc.jl")
    include("/home/rmadmin/.julia/v0.4/RegTools/src/modsel.jl")

#include("/media/u01/analytics/RegTools/diagnostics.jl")
#include("/media/u01/analytics/RegTools/misc.jl")
#include("/media/u01/analytics/RegTools/modsel.jl")


function vif!(g::xGLM)
    vdf=vif(g.model)
    vdf[:vars] = convert(Array{Symbol}, vdf[:variable])
    g.sdf = join(g.sdf,vdf[[:vars,:vif]], on = :vars, kind=:outer)
end




"""
    checksingularity(form::Formula, data::DataFrame, tolerance = 1.e-8)

Return a vector of terms in `form` that produce singularity in the model matrix
"""
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
# -- test --
#m=mocc
#custom_vars=[:dolocc_reduction,:occ_reduction,:pen_reduction,:data_b_e_nb,:data_nb_ne_b,:whyo,:iso]
#required_vars=vcat([m.y_var,:panid,m.logvar],cfg[:random_demos],cfg[:random_campaigns],cfg[:scoring_vars])
#vars=setdiff(m.vars,vcat(required_vars,custom_vars))
#m.removed_SingleLevelVars=FS_singleLevel(dfd,vars)
#vars = rmVars(m.removed_SingleLevelVars)
#f=genFmula(m.y_var,vars,m.logvar)

#checksingularity(f, dfd)
# -- test end ---




# =======================================================================================
# =======================================================================================

using IRImodels

factor_cols=vcat( [ cfg[:proscore], :group, :panid], cfg[:allrandoms] )

function featureSelection(dfd::DataFrame, m::MModel)
    function rmVars(v::Array{Symbol})
        v=setdiff(v,[:group])
        return setdiff(vars,v)
    end
    
    
    log_colname=Symbol("LOG_"*string(m.logvar))
    dfd[log_colname]=log(Array(dfd[m.logvar]+1))
    #this.vfactors=vcat(setdiff(this.vfactors,this.xvars),[log_colname])
    #this.fmula = genFmula(y_var, this.vfactors, logvar  )
            
                
    
    custom_vars=[:dolocc_reduction,:occ_reduction,:pen_reduction,:data_b_e_nb,:data_nb_ne_b,:whyo,:iso]
    required_vars=vcat([m.y_var,:panid,m.logvar],cfg[:random_demos],cfg[:random_campaigns],cfg[:scoring_vars])
    vars=setdiff(vcat(m.vars,[log_colname]),vcat(required_vars,custom_vars))
    
    println(uppercase(mocc.modelName)*" : SingleValue") #SingleValue
    m.removed_SingleLevelVars=FS_singleLevel(dfd,vars)
    vars = rmVars(m.removed_SingleLevelVars)
    
    println(uppercase(mocc.modelName)*" : Singularity : "*string(genFmula(m.y_var,vars,m.logvar))) # Singularity
    m.singularity_x = checksingularity(genFmula(m.y_var,vars,m.logvar), dfd)
    vars = rmVars(m.singularity_x)
    
    println(uppercase(mocc.modelName)*" : PVals") #PVals
    m.glm1_pvals = xGLM(dfd, m.dist, m.y_var, m.logvar, m.lnk , vars  )  
    g1=m.glm1_pvals
    m.glm1_pvals_x=g1.sdf[g1.sdf[:pval].>0.7,:vars]
    vars = rmVars(vcat(m.glm1_pvals_x, g1.xvars ) )
    
    println(uppercase(mocc.modelName)*" : Z & Vif") #Z & Vif
    m.glm2_ZnVIF = xGLM(dfd, m.dist, m.y_var, m.logvar, m.lnk ,vars  ) 
    g2=m.glm2_ZnVIF
    vif!(g2)
    z = g2.sdf[abs(g2.sdf[:zval]).<1.96,:vars]
    v = g2.sdf[ !DataArrays.isna(g2.sdf[:vif])&(g2.sdf[:vif].>15),:vars]
    m.glm2_ZnVIF_x =intersect(z,v)
    vars = rmVars(vcat(m.glm2_ZnVIF_x, g2.xvars) )


    function chkSigns(m::MModel, vars::Array{Symbol}, dfd::DataFrame, cfg::OrderedDict)  # Pvalue & Signs
        g = xGLM(dfd, m.dist, m.y_var, m.logvar, m.lnk , vars  )  
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
            stackdf = corrDF(dfd,vars_2)
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
    m.glm6_final = xGLM(dfd, m.dist, m.y_var, m.logvar, m.lnk , convert(Array{Symbol},vcat(m.finalvars,[:group]))  )
end

poolit!(dfd,factor_cols)

featureSelection(dfd[(dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1),:], mocc)
featureSelection(dfd[(dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1),:], mdolocc)
featureSelection(dfd[(dfd[:iso].==false) ,:], mpen)




# ------------------------------------------------------------------------------------------------------------
# ------ Analysis of Dev -------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

"""
    dropterm(f::Formula, trm::Symbol)

Return a copy of `f` without the term `trm`.

# Examples

julia> dropterm(foo ~ 1 + bar + baz, :bar)
foo ~ 1 + baz
"""
function dropterm(f::Formula, trm::Symbol)
    rhs = copy(f.rhs)
    args = rhs.args
    if !(Meta.isexpr(rhs, :call) && args[1] == :+ && (tpos = findlast(args, trm)) > 0)
        throw(ArgumentError("$trm is not a summand in `$rhs`"))
    end
    Formula(f.lhs, Expr(:call, :+, deleteat!(args, [1, tpos])...))
end


"""
    drop1(m::DataFrameRegressionModel, terms::Vector{Symbol}=Symbol[])

Perform a `drop1` analysis of deviance on `terms` in model `m`

Model `m` is modified and refit dropping the terms in `terms` one at a time.
An analysis of deviance is applied to the refit model.
"""
function drop1(m, terms::Vector{Symbol}=Symbol[])
    form = Formula(m.mf.terms)
    basedev = deviance(m)
    deviances, degsfreedom, Χ², pvalues, nms = [basedev], [0], [0.], [NaN], ["<none>"]
    ncoef = length(coef(m))
    dat = m.mf.df
    D = Distribution(m)
    dinst = D <: Poisson ? Poisson() :
        D == Bernoulli ? Bernoulli() :
        D == Binomial ? Binomial() :
        D <: Gamma ? Gamma(1.,1.) :
        throw(ArgumentError("Distribution $D not known in table"))
    L = Link(m)()
    if length(terms) == 0
        terms = filter(x -> isa(x, Symbol) || isa(x, Expr), form.rhs.args[2:end])
    end
    for trm in terms
        newform = dropterm(form, trm)
        newm = glm(newform, dat, dinst, L)
        newdev = deviance(newm)
        newdof = ncoef - length(coef(newm))
        newΧ² = newdev - basedev
        push!(deviances, newdev)
        push!(degsfreedom, newdof)
        push!(Χ², newΧ²)
        push!(pvalues, ccdf(Chisq(newdof), newΧ²))
        push!(nms, string(trm))
    end
    CoefTable([deviances, Χ², degsfreedom, pvalues], ["Deviance", "Χ²", "degfree", "P[> Χ²]"], nms)
end

Distributions.Distribution{T,D,L}(m::GlmResp{T,D,L}) = D
Distributions.Distribution(m::GeneralizedLinearModel) = Distribution(m.rr)
Distributions.Distribution{T<:GeneralizedLinearModel}(m::DataFrames.DataFrameRegressionModel{T}) = Distribution(m.model)

GLM.Link{T,D,L}(m::GlmResp{T,D,L}) = L
GLM.Link(m::GeneralizedLinearModel) = Link(m.rr)
GLM.Link{T<:GeneralizedLinearModel}(m::DataFrames.DataFrameRegressionModel{T}) = Link(m.model)


# ------- 
poolit!(dfd,cfg[:random_campaigns])
f=genFmula(mocc.y_var,vcat(mocc.finalvars,[:group],cfg[:random_campaigns]),mocc.logvar)
model_vars = convert(Array{Symbol},vcat(f.lhs,f.rhs.args[3:end]))
glm_occ=glm(f, dfd[ (dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1) , model_vars ] , mocc.dist, mocc.lnk)
checksingularity(genFmula(m.y_var,vars,m.logvar), dfd)

glm(f,  dfd[convert(Array{Symbol},vcat(f.lhs,f.rhs.args[3:end]))]  , dist, lnk )
featureSelection(dfd[(dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1),:], mocc)



# ---- 2 -----
#dev
poolit!(dfd,cfg[:random_campaigns])
m=mocc
for c in vcat([:group],cfg[:random_campaigns])
    f=genFmula(m.y_var,vcat(m.finalvars,[c]),m.logvar)
    g=glm(f, dfd[ (dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1) , convert(Array{Symbol},vcat(f.lhs,f.rhs.args[3:end])) ] , m.dist, m.lnk)
    d1=drop1(g, [c])
    d_g = d1.cols[find( x->(x == "P[> Χ²]"), d1.colnms)[1]][find( x->(x == string(c)), d1.rownms)[1]]
    println("----> ",d_g)
end




f=genFmula(mocc.y_var,vcat(mocc.finalvars,[:group]),mocc.logvar)
f=genFmula(mocc.y_var,vcat(mocc.finalvars,[:group],cfg[:random_campaigns]),mocc.logvar)

g=glm(f, dfd[ (dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1) , model_vars ] , mocc.dist, mocc.lnk)

drop1(g)

y.cols[end][end]



----------
find( x->(x == "P[> Χ²]"), y.colnms)
# time: 2016-09-26 21:05:07 UTC
# mode: julia
        find( x->(x == "P[> Χ²]"), y.colnms)[1]
# time: 2016-09-26 21:05:28 UTC
# mode: julia
        y.cols[find( x->(x == "P[> Χ²]"), y.colnms)[1]]   [find( x->(x == "group"), y.rownms)[1]]
# time: 2016-09-26 21:05:32 UTC
# mode: julia
        y.cols[find( x->(x == "P[> Χ²]"), y.colnms)[1]][find( x->(x == "group"), y.rownms)[1]]

-----------




-------------- Campels--------
using DataStructures, DataFrames, StatsFuns, GLM , JuMP,NLopt, HDF5, JLD, Distributions, MixedModels, RCall, StatsBase, xCommon
 R""" #load("/media/u01/analytics/Campbell_Soup_785/Victor/Glen_finaldata_dolocc.RDS")
dfd <- readRDS("/media/u01/analytics/Glen_finaldata_dolocc.RDS")
         ls()
     """
df_in = rcopy("dfd")


#Campels
    cfg[:all_mandatory_vars] = []

cfg[:all_mandatory_vars] = [:experian_id, cfg[:dolocc_y_var],cfg[:dolocc_logvar],
                            :hh_age,:estimated_hh_income,:number_of_children_in_living_Un,:person_1_gender]


------------------------------








function genRFmula(y::Symbol, logvar::Symbol, fvars::Array{Symbol}, rvars::Array{Symbol} )
    xvars=setdiff(fvars,[y,logvar]) # " ~ 1"
    vout = string(y)*" ~ 1"* reduce(*, [ " + "*  string(c) for c in xvars ] )
    f1 = replace(vout," ~ 1 + "," ~ ") 
    f1 = f1 * " + offset(log("*string(logvar)*"+1)) + group" * reduce(*, [ " + (0+group|"*string(c)*")" for c in rvars ] )
end
f = genRFmula(mocc.y_var, mocc.logvar, mocc.finalvars, cfg[:random_campaigns] )

#include("/home/rmadmin/g/StatStack/src/RCode.jl")




function runRglmm(dfd::DataFrame,f::String, modelName::String)
    R"""
    rm(list=ls())
    Start.Time <- Sys.time()
    list.of.packages <- c("data.table","bit64","optimx","lme4",'languageR','lmerTest','glmnet','lsmeans','car','RDS','DMwR','stringr','nloptr','ggplot2','reshape2','plyr','viridis','gmodels')
    new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
    if(length(new.packages)) install.packages(new.packages)
    lapply(list.of.packages,require,character.only=TRUE)
    
    defaultControl <- list(algorithm="NLOPT_LN_BOBYQA",xtol_rel=1e-6,maxeval=1e5)
    nloptwrap2 <- function(fn,par,lower,upper,control=list(),...) 
       {
         for (n in names(defaultControl)) 
             if (is.null(control[[n]])) control[[n]] <- defaultControl[[n]]
                 res <- nloptr(x0=par,eval_f=fn,lb=lower,ub=upper,opts=control,...)
             with(res,list(par=solution,
                    fval=objective,
                    feval=iterations,
                    conv=if (status>0) 0 else status,
                    message=message))
       }
    """
    
    dfname="/home/rmadmin/glmm_dataset_"*modelName*".dat"
    writetable(dfname, dfd, header = true)
    @rput dfname
    @rput f
    @rput modelName
    
    R2fact = vcat(cfg[:random_campaigns],[:group])
    R2fact = map(x-> string(x),R2fact)
    @rput R2fact
    R2num = vcat(m.finalvars,[cfg[:occ_y_var],cfg[:occ_logvar],cfg[:dolocc_y_var],cfg[:dolocc_logvar],cfg[:pen_y_var],cfg[:pen_logvar]] )
    R2num = map(x-> string(x),R2num)
    @rput R2num
    
    R""" dfd <- fread(   dfname  ,colClasses = "as.character",header=T) 
    dfd$trps_pre_p1
    
    for(i in noquote(R2fact)){ data[i] <- lapply(data[i],function(x) as.factor(x))  } 

    
    
    
    if (modelName=="occ") 
       { cat("occ")
         a1$Trps_POS_P1 <- as.factor(a1$Trps_POS_P1)
    
    dfd$targeting_fct1 <- as.factor(dfd$targeting_fct1)
    dfd$publisher_fct1 <- as.factor(dfd$publisher_fct1)
         occ_model <- glmer(f,data=dfd,family=poisson(link="log"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
         
       } 
    else if (modelName=="dolocc") 
       { cat("dolocc") 
         dolocc_model <- glmer(f,data=dfd,family=Gamma(link="log"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
       } 
    else 
       {  cat("pen")
          pen_model <- glmer(f,data=dfd,family=binomial(link="logit"),control = glmerControl(calc.derivs = FALSE, optimizer="nloptwrap2"))
       }
         
    """

end

m = mocc


runRglmm(dfd[(dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1),:] ,f, m.modelName)


for(i in R2fact)
    { eval(parse(text=dfd[i])) ]


        
        
        
        
        
        
# -----------------------------------------------------------------------------------------------------------------
#    OLD
# -----------------------------------------------------------------------------------------------------------------
"""

using DataStructures, DataFrames, StatsFuns, GLM, NLopt, HDF5, JLD, Distributions, MixedModels, RCall, StatsBase, xCommon

function loadDF()
    #cd("/media/u01/analytics/scoring/Healthy/modeling5/")
    #df_data = readtable("csv_final_healthychoice1_I3.csv",header=true); #lowercase(df_data)
    #df_h = readtable("Headers_healthy_choice3.csv",header=false); #lowercase(df_data)
    #names!(df_data, convert(Array{Symbol}, df_h[:x1]) ) 
    cd("/media/u01/analytics/scoring/CDW5_792/")
    df_data = readtable("csv_final_cdw5_792.csv",header=false);
    df_h = readtable("Headers_cdw5.csv",header=false);
    names!(df_data, convert(Array{Symbol}, df_h[:x1]) )
end
df_in=loadDF()



function isValid(df_data::DataFrame,cfg::OrderedDict)
    function checkValid(iarr::Array{Symbol})  length(setdiff(iarr, names(df_data))) > 0 ? false : true end
    !checkValid(cfg[:all_mandatory_vars]) ? error("ERROR: Not all mandatory_vars in dataset ") : println("VALID : mandatory_vars")
    !checkValid(cfg[:scoring_vars]) ? error("ERROR: Not all scoring_vars in dataset ") : println("VALID : scoring_vars")
end


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
                        ,:all_mandatory_vars => [:Buyer_Pos_P1,:Buyer_Pre_P1,:Trps_POS_P1,:Trps_PRE_P1,:Dol_per_Trip_POS_P1,:Dol_per_Trip_PRE_P1,
                                                 :Nonbuyer_Pre_P1,:hh_age,:estimated_hh_income,:number_of_children_in_living_Un,
                                                 :person_1_gender,:experian_id]
                       )


function getCFG(df_in::DataFrame)
    cfg=loadCFG(cfgDefaults, "app.cfg")
    cfg[:exposed_flag_var] = :exposed_flag_new                  # Go In app.cfg
    cfg[:random_campaigns] = [:Publisher_Fct1,:Targeting_Fct1]  # Go In app.cfg

    cfg[:allrandoms] = vcat(cfg[:random_demos],cfg[:random_campaigns])
    #cfg[:ProScore] = grep1("MODEL",names(df_in))
    ps = filter(x->contains(string(x), "MODEL"), names(df_in)) # Set the ProScore variable in the dataset   #push!(cfg[:all_mandatory_vars], cfg[:ProScore])
    cfg[:ProScore] = length(ps) > 0 ? ps[1] : :MISSING_MODEL_VARIABLE_IN_DATA
    cfg[:num_products] = length( grep("Buyer_Pre_P",names(df_in)))-1  # get number of products in the data
    #rework vars after loading cfg
    cfg[:random_campaigns] = intersect(cfg[:random_campaigns],names(df_in))


    cfg[:occ_y_var] = :trps_pos_p1
    cfg[:occ_logvar] = :trps_pre_p1
    cfg[:dolocc_y_var] = :dol_per_trip_pos_p1
    cfg[:dolocc_logvar] = :dol_per_trip_pre_p1
    cfg[:pen_y_var] = :buyer_pos_p1
    cfg[:pen_logvar] = :buyer_pre_p1


    return cfg
end
cfg=getCFG(df_in)


isValid(df_in,cfg)




function reworkCFG!(df_in::DataFrame,cfg::OrderedDict)
    xflags=[cfg[:exposed_flag_var],:exposed_flag]
    features = setdiff(names(df_in),xflags)

    cfg[:xVarsDemos]=vcat([:experian_id, :banner, :Prd_0_Qty_PRE],features[findfirst(features, :state):findfirst(features, :Mosaic)])   # exclude demos between....
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

function data_Prep(df_data::DataFrame, cfg::OrderedDict)
    df_data[:isO]=false
    df_data[:whyO]=""

    df_data[df_data[:number_of_children_in_living_Un].>=4,:number_of_children_in_living_Un] = 4  # aggregate #of children for 4+ L_114
    #vars = getvars()   # :GROUP
    #vars = DataFrame(names=names(df_data),eltypes=eltypes(df_data))
    if typeof(df_data[:group]) in [DataArray{String,1}]
        df_data[ findin(df_data[:group],["//N","\\N"]), :group] = "0"
        df_data[DataArrays.isna(df_data[:group]), :group]="0"
        df_data[:group] = [parse(Int64,s) for s = df_data[:group]]
    else
        df_data[ DataArrays.isnan(df_data[:group]), :group] = 0
    end

    vars = DataFrame(names=names(df_data),eltypes=eltypes(df_data))
    for c in setdiff(vars[findin(vars[:eltypes],[String]),:names],cfg[:allrandoms]) # set variables as.numeric and replace NA's with zero
        print("Convert String: String->Numeric: ",c)
        try df_data[c] = map( x -> DataArrays.isna(x) ?  NaN : convert(Float64, x)  , df_data[c]) catch e println("  (failed)") end  #NA to NaN
        try df_data[c] = convert(Array{Float64}, df_data[c]) catch e end   # To Float64
    end
    vars = DataFrame(names=names(df_data),eltypes=eltypes(df_data))
    #vars = getvars()
    for c in setdiff(vars[findin(vars[:eltypes],[Float64]),:names],cfg[:allrandoms])  # replace NaN's with zero for numeric variables
        println("Replace Float64 NaN (0.0): ",c)
        df_data[ DataArrays.isnan(df_data[c]), c] = 0.0
    end
    for c in  setdiff(vars[findin(vars[:eltypes],[Int64]),:names],cfg[:allrandoms])
        println("Replace Int64 NaN (0) : ",c)
        df_data[ DataArrays.isnan(df_data[c]), c] = 0
    end


    # create factor variables -  Need to understand how this impacts the data passed to R - might need to do this in R
    #pool!(df_data, cfg[:allrandoms])
    #pool!(df_data, [,:group])

    # NOTE : Do QCs here
    df_data[df_data[:person_1_gender].=="U",:isO] = true   # remove HHs with no gender info
    df_data[df_data[:person_1_gender].=="U",:whyO] = "person_1_gender=U; remove HHs with no gender info"
    df_data[findin(df_data[:estimated_hh_income],["U","L"]),:estimated_hh_income]="L" # aggregate U and L levels of hh income

    for r in cfg[:random_campaigns]    # check and drop exposed HHs with no publisher info or non-exposed HHs with publisher info
        df_data[findin(df_data[r],["\\N","NULL","0","NONE"])  ,r] ="none"
        df_data[ (df_data[:isO].==false) & (df_data[r].!="none") & (df_data[:group].==0) ,:whyO] = "non exposed HHs with publisher info"
        df_data[ (df_data[:isO].==false) & (df_data[r].!="none") & (df_data[:group].==0) ,:isO] = true
        println(r," non exposed HHs with publisher info : ",nrow(df_data[df_data[:whyO].=="non exposed HHs with publisher info",:] )   )
        df_data[ (df_data[:isO].==false) & (df_data[r].=="none") & (df_data[:group].!=0) ,:whyO] = "exposed HHs with no publisher info"
        df_data[ (df_data[:isO].==false) & (df_data[r].=="none") & (df_data[:group].!=0) ,:isO] = true
        println(r," exposed HHs with no publisher info : ",nrow(df_data[df_data[:whyO].=="exposed HHs with no publisher info",:] )   )
    end

    # segments for outliers detection
    df_data[:data_NB_NE_B] = false
    df_data[ (df_data[:Buyer_Pre_P1].==0 ) & (df_data[:group].==0 ) & (df_data[:Buyer_Pos_P1].==1 ) ,:data_NB_NE_B] = true
    df_data[:data_B_E_NB] = false
    df_data[ (df_data[:Buyer_Pre_P1].==1 ) & (df_data[:group].==0 ) & (df_data[:Buyer_Pos_P1].==0 ) ,:data_B_E_NB] = true
    df_data[:pen_reduction] = false
    df_data[ (df_data[:Buyer_Pre_P1].==1) & (df_data[:group].==0) & (df_data[:Buyer_Pos_P1].==0 )  ,:pen_reduction] = true
    df_data[:occ_reduction] = false
    df_data[ (df_data[:group].==0) & (df_data[:Buyer_Pos_P1].==1) & (df_data[:Trps_POS_P1].< df_data[:Trps_PRE_P1] ) ,:occ_reduction] = true
    df_data[:dolocc_reduction] = false
    df_data[  (df_data[:group].==0) & (df_data[:Buyer_Pos_P1].==1) & (df_data[:Dol_per_Trip_POS_P1].< df_data[:Dol_per_Trip_PRE_P1] )  , :dolocc_reduction] = true

    return df_data[df_data[:isO].==false, : ]   #[setdiff(names(df_data),[:isO,:whyO])]
end

dfd = data_Prep(dfd, cfg);




function MatchMe(df_data::DataFrame,cfg::OrderedDict)
    df=df_data[df_data[:isO].==false,:]
    df_exp     = df[df[:group].==1,:]
    df_unexp   = df[df[:group].==0,:]
    df_exp_dim   = nrow(df_exp)
    df_unexp_dim = nrow(df_unexp)
    new_unexp_dim = df_unexp_dim*(df_unexp_dim>2000000 ? 0.3 : df_unexp_dim>1000000 ? 0.4 : df_unexp_dim>750000 ? 0.6 : 0.7)
    #df_sample=DataFrame
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
        df_data_sample  = vcat(df_exp,new_df_unexp_1,new_df_unexp_0)
    elseif length(string(cfg[:ProScore])) > 0
        sample_control_data=similar(df_unexp, 0)
        for (key, value) in countmap(df_exp[cfg[:ProScore]])
            sample_dim=round(Int64,new_unexp_dim*(value/df_exp_dim))
            temp_data = df_unexp[df_unexp[cfg[:ProScore]].==key,:]
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
            df_data_sample = vcat(sample_df_exp,new_df_unexp_1,sample_df_unexp_0)
        else
            new_df_unexp_0 = sample_df_unexp_0[sample(1:sample_df_unexp_0_dim, dim_sampleB, replace=false ),:]
            df_data_sample = vcat(sample_df_exp,sample_df_unexp_1,new_df_unexp_0)
        end
    end


    rows2remove = setdiff(df_data[df_data[:isO].==false, :panid],df_data_sample[:panid])
    df_data[findin(df_data[:panid],rows2remove),:whyO]="NoMatch"
    df_data[findin(df_data[:panid],rows2remove),:isO]=true

    return df_data[df_data[:isO].==false, : ]  #[setdiff(names(df_data),[:isO,:whyO])]
end

dfd = MatchMe(dfd,cfg)

lowercase!(dfd)
cfg=lowercase(cfg)

#Not Used
dfd=dfd[setdiff(names(dfd),[:nonbuyer_pre_p1])]



######################################
#------------MODEL OBJECTS-----------#
######################################
abstract MModel

#!!! Multicollinearity, removing :[:fea_or_dis_trps_shr_dpp_p1,:fea_or_dis_trps_shr_dpp_p2,:fea_or_dis_trps_shr_dpp_p3,:fea_or_dis_trps_shr_dpp_p4]
type xGLM
    vfactors::Vector{Symbol}
    fmula::DataFrames.Formula
    xvars::Vector{Symbol}
    model::Any #DataFrameRegressionModel
    sdf::DataFrame
    wasSuccessful::Bool

    function xGLM(dfd::DataFrame, dist::Distribution, y_var::Symbol, logvar::Symbol , lnk::Link , vfactors::Array{Symbol} )
        this=new()
        this.wasSuccessful=false
        this.xvars=Symbol[]
        this.vfactors=setdiff(vfactors, [y_var,logvar])
        for l in 1:30
            #log_colname=Symbol("LOG_"*string(logvar))
            #this.vfactors=vcat(setdiff(this.vfactors,this.xvars),[log_colname])
            this.fmula = genFmula(y_var, this.vfactors, logvar  )
            try
                #this.model = glm(this.fmula,  dfd[vcat(this.vfactors,[y_var])], dist, LogLink(), offset=log(Array(dfd[logvar]  + 1)) )
                #dfd[log_colname]=log(Array(dfd[logvar]+1))
                f=this.fmula
                ##this.model = glm(this.fmula,  dfd[vcat(this.vfactors,[y_var,log_colname])], dist, m.lnk )   # LogLink()
                this.model = glm(f,  dfd[convert(Array{Symbol},vcat(f.lhs,f.rhs.args[3:end]))]  , dist, lnk )

                this.sdf = DataFrame(vars=vcat([:intercept],this.model.mf.terms.terms), coef=coef(this.model), se=stderr(this.model), zval=coef(this.model)./stderr(this.model) ,pval= ccdf(FDist(1, df_residual(this.model)), abs2(coef(this.model)./stderr(this.model))))
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




type MDolOcc <: MModel
    vars::Vector{Symbol}
    finalvars::Vector{Symbol}
    y_var::Symbol
    dist::Distribution
    lnk::Link
    mandatory_vars::Vector{Symbol}
    exclude_vars::Vector{Symbol}
    removed_SingleLevelVars::Vector{Symbol}
    singularity_x::Vector{Symbol}
    glm1_pvals::xGLM
    glm1_pvals_x::Vector{Symbol}
    glm2_ZnVIF::xGLM
    glm2_ZnVIF_x::Vector{Symbol}
    glm3_PnSigns::xGLM
    glm3_PnSigns_x::Vector{Symbol}
    glm4_PnSignsClean::xGLM
    glm4_PnSignsClean_x::Vector{Symbol}
    glm5::xGLM
    glm5_Z_x::Vector{Symbol}
    corrvars_x::Vector{Symbol}
    glm6_final::xGLM
    Buyer_Pos_P1_is1::Bool
    modelName::String
    logvar::Symbol
    function MDolOcc(dfd::DataFrame,cfg::OrderedDict=Dict())
        this=new(); this.modelName="dolocc"; this.logvar=cfg[:dolocc_logvar]; this.y_var=cfg[:dolocc_y_var]; this.dist=Gamma()
        this.lnk=LogLink()
        this.mandatory_vars=Symbol[this.y_var]
        this.removed_SingleLevelVars=Symbol[]
        this.glm1_pvals_x=Symbol[]
        this.glm2_ZnVIF_x=Symbol[]
        this.glm3_PnSigns_x=Symbol[]
        this.glm4_PnSignsClean_x=Symbol[]
        this.corrvars_x=Symbol[]
        this.exclude_vars= Symbol[ cfg[:occ_y_var],cfg[:occ_logvar],cfg[:pen_y_var],cfg[:pen_logvar] ]
        this.Buyer_Pos_P1_is1=true
        if this.Buyer_Pos_P1_is1  this.exclude_vars = vcat(this.exclude_vars, [:buyer_pos_p1] )  end
        this.vars=setdiff(names(dfd),vcat(this.exclude_vars,[:iso,:whyo,:data_nb_ne_b, :data_b_e_nb ,:pen_reduction,:occ_reduction,:dolocc_reduction]))
        return this
    end
end
mdolocc = MDolOcc(dfd,cfg)





type MOcc <: MModel
    vars::Vector{Symbol}
    finalvars::Vector{Symbol}
    y_var::Symbol
    dist::Distribution
    lnk::Link
    mandatory_vars::Vector{Symbol}
    exclude_vars::Vector{Symbol}
    singularity_x::Vector{Symbol}
    removed_SingleLevelVars::Vector{Symbol}
    glm1_pvals::xGLM
    glm1_pvals_x::Vector{Symbol}
    glm2_ZnVIF::xGLM
    glm2_ZnVIF_x::Vector{Symbol}
    glm3_PnSigns::xGLM
    glm3_PnSigns_x::Vector{Symbol}
    glm4_PnSignsClean::xGLM
    glm4_PnSignsClean_x::Vector{Symbol}
    glm5::xGLM
    glm5_Z_x::Vector{Symbol}
    corrvars_x::Vector{Symbol}
    glm6_final::xGLM
    Buyer_Pos_P1_is1::Bool
    modelName::String
    logvar::Symbol
    function MOcc(dfd::DataFrame,cfg::OrderedDict=Dict())
        this=new(); this.modelName="occ"; this.logvar=cfg[:occ_logvar]; this.y_var=cfg[:occ_y_var]; this.dist=Poisson()
        this.lnk=LogLink()
        this.mandatory_vars=Symbol[this.y_var]
        this.removed_SingleLevelVars=Symbol[]
        this.glm1_pvals_x=Symbol[]
        this.glm2_ZnVIF_x=Symbol[]
        this.glm3_PnSigns_x=Symbol[]
        this.glm4_PnSignsClean_x=Symbol[]
        this.corrvars_x=Symbol[]
        this.exclude_vars=Symbol[cfg[:dolocc_y_var],cfg[:dolocc_logvar],cfg[:pen_y_var],cfg[:pen_logvar] ]
        this.Buyer_Pos_P1_is1=true
        if this.Buyer_Pos_P1_is1  this.exclude_vars = vcat(this.exclude_vars, [:buyer_pos_p1] )  end
        this.vars=setdiff(names(dfd),vcat(this.exclude_vars,[:iso,:whyo,:data_nb_ne_b, :data_b_e_nb ,:pen_reduction,:occ_reduction,:dolocc_reduction]))
        return this
    end
end
mocc = MOcc(dfd,cfg)



type MPen <: MModel
    vars::Vector{Symbol}
    finalvars::Vector{Symbol}
    y_var::Symbol
    dist::Distribution
    lnk::Link
    mandatory_vars::Vector{Symbol}
    exclude_vars::Vector{Symbol}
    singularity_x::Vector{Symbol}
    removed_SingleLevelVars::Vector{Symbol}
    glm1_pvals::xGLM
    glm1_pvals_x::Vector{Symbol}
    glm2_ZnVIF::xGLM
    glm2_ZnVIF_x::Vector{Symbol}
    glm3_PnSigns::xGLM
    glm3_PnSigns_x::Vector{Symbol}
    glm4_PnSignsClean::xGLM
    glm4_PnSignsClean_x::Vector{Symbol}
    glm5::xGLM
    glm5_Z_x::Vector{Symbol}
    corrvars_x::Vector{Symbol}
    glm6_final::xGLM
    Buyer_Pos_P1_is1::Bool
    modelName::String
    logvar::Symbol
    function MPen(dfd::DataFrame,cfg::OrderedDict=Dict())
        this=new(); this.modelName="pen"; this.logvar=cfg[:pen_logvar]; this.y_var=cfg[:pen_y_var]; this.dist=Bernoulli() #Binomial()
        this.lnk=LogitLink()
        this.mandatory_vars=Symbol[this.y_var]
        this.removed_SingleLevelVars=Symbol[]
        this.glm1_pvals_x=Symbol[]
        this.glm2_ZnVIF_x=Symbol[]
        this.glm3_PnSigns_x=Symbol[]
        this.glm4_PnSignsClean_x=Symbol[]
        this.corrvars_x=Symbol[]
        this.exclude_vars=Symbol[cfg[:occ_y_var],cfg[:occ_logvar],cfg[:dolocc_y_var],cfg[:dolocc_logvar] ]
        this.Buyer_Pos_P1_is1=false
        if this.Buyer_Pos_P1_is1  this.exclude_vars = vcat(this.exclude_vars, [:buyer_pos_p1] )  end
        this.vars=setdiff(names(dfd),vcat(this.exclude_vars,[:iso,:whyo,:data_nb_ne_b, :data_b_e_nb ,:pen_reduction,:occ_reduction,:dolocc_reduction]))
        return this
    end
end
mpen = MPen(dfd,cfg)



    include("/home/rmadmin/.julia/v0.4/RegTools/src/diagnostics.jl")
    include("/home/rmadmin/.julia/v0.4/RegTools/src/misc.jl")
    include("/home/rmadmin/.julia/v0.4/RegTools/src/modsel.jl")

#include("/media/u01/analytics/RegTools/diagnostics.jl")
#include("/media/u01/analytics/RegTools/misc.jl")
#include("/media/u01/analytics/RegTools/modsel.jl")


function vif!(g::xGLM)
    vdf=vif(g.model)
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
# -- test --
#m=mocc
#custom_vars=[:dolocc_reduction,:occ_reduction,:pen_reduction,:data_b_e_nb,:data_nb_ne_b,:whyo,:iso]
#required_vars=vcat([m.y_var,:panid,m.logvar],cfg[:random_demos],cfg[:random_campaigns],cfg[:scoring_vars])
#vars=setdiff(m.vars,vcat(required_vars,custom_vars))
#m.removed_SingleLevelVars=FS_singleLevel(dfd,vars)
#vars = rmVars(m.removed_SingleLevelVars)
#f=genFmula(m.y_var,vars,m.logvar)

#checksingularity(f, dfd)
# -- test end ---



# =======================================================================================
# =======================================================================================

using IRImodels

factor_cols=vcat( [ cfg[:proscore], :group, :panid], cfg[:allrandoms] )

function featureSelection(dfd::DataFrame, m::MModel)
    function rmVars(v::Array{Symbol})
        v=setdiff(v,[:group])
        return setdiff(vars,v)
    end


    log_colname=Symbol("LOG_"*string(m.logvar))
    dfd[log_colname]=log(Array(dfd[m.logvar]+1))
    #this.vfactors=vcat(setdiff(this.vfactors,this.xvars),[log_colname])
    #this.fmula = genFmula(y_var, this.vfactors, logvar  )



    custom_vars=[:dolocc_reduction,:occ_reduction,:pen_reduction,:data_b_e_nb,:data_nb_ne_b,:whyo,:iso]
    required_vars=vcat([m.y_var,:panid,m.logvar],cfg[:random_demos],cfg[:random_campaigns],cfg[:scoring_vars])
    vars=setdiff(vcat(m.vars,[log_colname]),vcat(required_vars,custom_vars))

    #SingleValue
    m.removed_SingleLevelVars=FS_singleLevel(dfd,vars)
    vars = rmVars(m.removed_SingleLevelVars)

    # Singularity
    m.singularity_x = checksingularity(genFmula(m.y_var,vars,m.logvar), dfd)
    vars = rmVars(m.singularity_x)

    #PVals
    m.glm1_pvals = xGLM(dfd, m.dist, m.y_var, m.logvar, m.lnk , vars  )
    g1=m.glm1_pvals
    m.glm1_pvals_x=g1.sdf[g1.sdf[:pval].>0.7,:vars]
    vars = rmVars(vcat(m.glm1_pvals_x, g1.xvars ) )

    #Z & Vif
    m.glm2_ZnVIF = xGLM(dfd, m.dist, m.y_var, m.logvar, m.lnk ,vars  )
    g2=m.glm2_ZnVIF
    vif!(g2)
    z = g2.sdf[abs(g2.sdf[:zval]).<1.96,:vars]
    v = g2.sdf[ !DataArrays.isna(g2.sdf[:vif])&(g2.sdf[:vif].>15),:vars]
    m.glm2_ZnVIF_x =intersect(z,v)
    vars = rmVars(vcat(m.glm2_ZnVIF_x, g2.xvars) )


    function chkSigns(m::MModel, vars::Array{Symbol}, dfd::DataFrame, cfg::OrderedDict)  # Pvalue & Signs
        g = xGLM(dfd, m.dist, m.y_var, m.logvar, m.lnk , vars  )
        neutralvars = setdiff(vars,vcat(cfg[:negativevars],cfg[:positivevars]))
        neg=intersect(cfg[:negativevars],g.sdf[g.sdf[:coef].<0,:vars])
        pos=intersect(cfg[:positivevars],g.sdf[g.sdf[:coef].>0,:vars])
        varstokeep = intersect(vcat(neutralvars, pos,neg) ,  g.sdf[ g.sdf[:pval].<cfg[:pvalue_lvl] ,:vars] )
        return g, varstokeep
    end

    (m.glm3_PnSigns, initialvars) = chkSigns(m, vars, dfd, cfg)
    (m.glm4_PnSignsClean, vars_2) = chkSigns(m, convert(Array{Symbol},initialvars) , dfd, cfg)


    function getCorrVars(dfd::DataFrame, vars_2::Array{Symbol})
        rm_lst=Symbol[]
        if (length(vars_2) > 1) & (   length(getColswithType("num", dfd, convert(Array{Symbol},vars_2) ) ) > 1  )
            stackdf = corrDF(dfd,vars_2)
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

    m.corrvars_x = getCorrVars(dfd,convert(Array{Symbol},setdiff(vars_2,factor_cols)))
    vars_2 = setdiff(vars_2,m.corrvars_x)
    (m.glm5, m.finalvars) =  chkSigns(m, convert(Array{Symbol},vars_2), dfd, cfg)

    # Final Review:
    m.glm6_final = xGLM(dfd, m.dist, m.y_var, m.logvar, m.lnk , convert(Array{Symbol},vcat(m.finalvars,[:group]))  )
end



featureSelection(dfd[(dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1),:], mocc)
featureSelection(dfd[(dfd[:iso].==false)&(dfd[:buyer_pos_p1].==1),:], mdolocc)
featureSelection(dfd[(dfd[:iso].==false) ,:], mpen)

"""
        
        
        
        
        
        
        
        