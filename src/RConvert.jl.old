# ==============================  LOAD DATA =========================================
#save("data.jld", "data", df_data, compress=true)
#load("data.jld")["data"]

using DataStructures, DataFrames, StatsFuns, GLM , JuMP,NLopt, StatStack, HDF5, JLD, Distributions, MixedModels, RCall, StatsBase, xCommon

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


function getCFG(df_in::DataFrame)
    cfgDefaults=OrderedDict( :P2_Competitor => true
                        ,:pvalue_lvl => 0.20  #pvalue_lvl = 0.20 
                        ,:excludedBreaks => AbstractString[]    #["estimated_hh_income","hh_age","number_of_children_in_living_un","person_1_gender"]
                        ,:excludedLevels => ["none"]
                        ,:excludedKeys => AbstractString[]
                        ,:exposed_flag_var => :exposed_flag
                        ,:sigLevel => "0.2"
                        ,:random_demos => [:estimated_hh_income,:hh_age,:number_of_children_in_living_Un,:person_1_gender]
                        ,:random_campaigns => []
                        ,:dropvars => [:exposed_flag]
                        ,:scoring_vars => [:Prd_1_Net_Pr_PRE,:Prd_1_Net_Pr_POS,:Buyer_Pos_P0,:Buyer_Pre_P0]
                        ,:all_mandatory_vars => [:Buyer_Pos_P1,:Buyer_Pre_P1,:Trps_POS_P1,:Trps_PRE_P1,:Dol_per_Trip_POS_P1,:Dol_per_Trip_PRE_P1,:Nonbuyer_Pre_P1,:hh_age,:estimated_hh_income,:number_of_children_in_living_Un,:person_1_gender,:experian_id]
                       )
    cfg=xCommon.loadCFG(cfgDefaults, pwd()*"/app.cfg")
    
    cfg[:exposed_flag_var] = :exposed_flag_new                  # Go In app.cfg
    cfg[:random_campaigns] = [:Publisher_Fct1,:Targeting_Fct1]  # Go In app.cfg

    cfg[:allrandoms] = vcat(cfg[:random_demos],cfg[:random_campaigns])
    cfg[:ProScore] = grep1("MODEL",names(df_in))   # Set the ProScore variable in the dataset   #push!(cfg[:all_mandatory_vars], cfg[:ProScore])
    cfg[:ProScore] = cfg[:ProScore]==nothing ? :MISSING_MODEL_VARIABLE_IN_DATA : cfg[:ProScore]
    cfg[:num_products] = length( grep("Buyer_Pre_P",names(df_in)))-1  # get number of products in the data
    #rework vars after loading cfg
    cfg[:random_campaigns] = intersect(cfg[:random_campaigns],names(df_in))
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

df_data = reworkCFG!(df_in,cfg)



#######################################
#--------- Data Manipulation ---------#
#######################################

function data_Prep(df_data::DataFrame, cfg::OrderedDict)
    df_data[:isO]=false
    df_data[:whyO]="" 
    
    df_data[df_data[:number_of_children_in_living_Un].>=4,:number_of_children_in_living_Un] = 4  # aggregate #of children for 4+ L_114
    #vars = getvars()   # :GROUP
    #vars = DataFrame(names=names(df_data),eltypes=eltypes(df_data))
    if typeof(df_data[:group]) in [DataArray{UTF8String,1},DataArray{ASCIIString,1}] 
        df_data[ findin(df_data[:group],["//N","\\N"]), :group] = "0" 
        df_data[isna(df_data[:group]), :group]="0"
        df_data[:group] = [parse(Int64,s) for s = df_data[:group]]
    else
        df_data[ isnan(df_data[:group]), :group] = 0
    end
    
    vars = DataFrame(names=names(df_data),eltypes=eltypes(df_data))
    for c in setdiff(vars[findin(vars[:eltypes],[UTF8String]),:names],cfg[:allrandoms]) # set variables as.numeric and replace NA's with zero
        print("Convert String: String->Numeric: ",c)
        try df_data[c] = map( x -> isna(x) ?  NaN : convert(Float64, x)  , df_data[c]) catch e println("  (failed)") end  #NA to NaN
        try df_data[c] = convert(Array{Float64}, df_data[c]) catch e end   # To Float64
    end
    vars = DataFrame(names=names(df_data),eltypes=eltypes(df_data))
    #vars = getvars()
    for c in setdiff(vars[findin(vars[:eltypes],[Float64]),:names],cfg[:allrandoms])  # replace NaN's with zero for numeric variables 
        println("Replace Float64 NaN (0.0): ",c)
        df_data[ isnan(df_data[c]), c] = 0.0
    end
    for c in  setdiff(vars[findin(vars[:eltypes],[Int64]),:names],cfg[:allrandoms])    
        println("Replace Int64 NaN (0) : ",c)
        df_data[ isnan(df_data[c]), c] = 0
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

df_data = data_Prep(df_data, cfg);




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
    
    
    #Generate Report!!!! ---   still need to remove :isO
    function dict2df(d::Dict,kc::AbstractString,c::AbstractString)
        dfo=DataFrame(k=Int64[],v=Int64[])
        for (key, value) in d
            push!(dfo,[key, value])
        end
        rename!(dfo,:k,symbol(kc))
        rename!(dfo,:v,symbol(c))
        return dfo
    end
    ds0=dict2df(countmap(df_data_sample[ df_data_sample[:group].==0,cfg[:ProScore]]),"ProScore","out_unexp")
    ds1=dict2df(countmap(df_data_sample[ df_data_sample[:group].==1,cfg[:ProScore]]),"ProScore","out_exp")
    do0=dict2df(countmap(df[ df[:group].==0,cfg[:ProScore]]),"ProScore","in_unexp")
    do1=dict2df(countmap(df[ df[:group].==1,cfg[:ProScore]]),"ProScore","in_exp")
    dfo=join(join(join(do0, do1, on = :ProScore,kind=:outer), ds0, on = :ProScore,kind=:outer),ds1, on = :ProScore,kind=:outer)
    for c in [:out_unexp, :out_exp, :in_unexp, :in_exp]
        dfo[isna(dfo[c]), c] = 0
    end
    b_ds0=dict2df(countmap(df_data_sample[ df_data_sample[:group].==0,:Buyer_Pre_P1]),"Buyer_Pre_P1","out_unexp")
    b_ds1=dict2df(countmap(df_data_sample[ df_data_sample[:group].==1,:Buyer_Pre_P1]),"Buyer_Pre_P1","out_exp")
    b_do0=dict2df(countmap(df_data[ df_data[:group].==0,:Buyer_Pre_P1]),"Buyer_Pre_P1","in_unexp")
    b_do1=dict2df(countmap(df_data[ df_data[:group].==1,:Buyer_Pre_P1]),"Buyer_Pre_P1","in_exp")
    dfx=DataFrame(Buyer_Pre_P1=[0,1],Buyer_Pre_Period=["No","Yes"])
    b_dfo1=join(b_do0, b_do1, on = :Buyer_Pre_P1,kind=:outer)
    b_dfo1[:in_proportion] = b_dfo1[:in_exp]./sum(b_dfo1[:in_exp])
    b_dfo2=join(b_ds0, b_ds1, on = :Buyer_Pre_P1,kind=:outer)
    b_dfo2[:out_proportion] =b_dfo2[:out_exp]./sum(b_dfo2[:out_exp])
    b_dfo=join(dfx,join(b_dfo1, b_dfo2, on = :Buyer_Pre_P1,kind=:outer), on = :Buyer_Pre_P1,kind=:outer)
    delete!(b_dfo, :Buyer_Pre_P1 )
    #still need to Write QC to file
    #dfo  b_dfo
    
    rows2remove = setdiff(df_data[df_data[:isO].==false, :panid],df_data_sample[:panid])
    df_data[findin(df_data[:panid],rows2remove),:whyO]="NoMatch"
    df_data[findin(df_data[:panid],rows2remove),:isO]=true 
    
    return df_data[df_data[:isO].==false, : ][setdiff(names(df_data),[:isO,:whyO])] 
end

df_data = MatchMe(df_data,cfg)

lowercase!(df_data)
cfg=lowercase(cfg)


abstract MModel 




df_data[:iso]=false
#df_data=df_data[(df_data[:iso].==false)&(df_data[:buyer_pos_p1].==1),:]

######################################
#--------------OCCASION--------------#
######################################

#!!! Multicollinearity, removing :[:fea_or_dis_trps_shr_dpp_p1,:fea_or_dis_trps_shr_dpp_p2,:fea_or_dis_trps_shr_dpp_p3,:fea_or_dis_trps_shr_dpp_p4]
type xGLM_OLD
    vfactors::Vector{Symbol}
    fmula::DataFrames.Formula
    xvars::Vector{Symbol}   
    model::Any #DataFrameRegressionModel
    sdf::DataFrame
    wasSuccessful::Bool
    xFromGLM::Vector{Symbol} 
 
    function xGLM(df_data::DataFrame, m::MModel, cfg::OrderedDict) 
        this=new()
        this.wasSuccessful=false
        this.xvars=Symbol[]
        this.xFromGLM = vcat([m.y_var, :panid, m.logvar],cfg[:random_demos],cfg[:random_campaigns],cfg[:scoring_vars])
        this.vfactors=unique(setdiff(m.vars, this.xFromGLM)) 
        for l in 1:30
            this.vfactors=setdiff(this.vfactors,this.xvars)
            this.fmula = genFmula(m.y_var, this.vfactors, m.logvar  )
            try
                #this.model = glm(this.fmula, df_data[  (df_data[:iso].==false)&(df_data[:buyer_pos_p1].==1)  ,vcat(this.vfactors,[m.y_var,m.logvar])], m.dist, LogLink(), offset=log(Array(df_data[(df_data[:iso].==false)&(df_data[:buyer_pos_p1].==1),m.logvar]  + 1)) ) 
                this.model = glm(this.fmula,  df_data[vcat(m.vars,[m.y_var,m.logvar])], m.dist, LogLink(), offset=log(Array(df_data[m.logvar]  + 1)) )
                
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


type xGLM
    vfactors::Vector{Symbol}
    fmula::DataFrames.Formula
    xvars::Vector{Symbol}   
    model::Any #DataFrameRegressionModel
    sdf::DataFrame
    wasSuccessful::Bool
    #xFromGLM::Vector{Symbol} 
 
    function xGLM(dfd::DataFrame, dist::Distribution, y_var::Symbol, logvar::Symbol , vfactors::Array{Symbol} )  
        this=new()
        this.wasSuccessful=false
        this.xvars=Symbol[]
        this.vfactors=setdiff(vfactors, [y_var,logvar])
        for l in 1:30
            this.vfactors=setdiff(this.vfactors,this.xvars)
            this.fmula = genFmula(y_var, this.vfactors, logvar  )
            try
                this.model = glm(this.fmula,  dfd[vcat(this.vfactors,[y_var])], dist, LogLink(), offset=log(Array(dfd[logvar]  + 1)) )
                
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



type MOcc <: MModel
    vars::Vector{Symbol}
    y_var::Symbol
    dist::Distribution
    mandatory_vars::Vector{Symbol}
    exclude_vars::Vector{Symbol}
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
    
    Buyer_Pos_P1_is1::Bool
    
    modelName::AbstractString
    logvar::Symbol
    
    function MOcc(dfd::DataFrame,cfg::OrderedDict=Dict()) 
        this=new(); this.modelName="occ"; this.logvar=:trps_pre_p1; this.y_var=:trps_pos_p1; this.dist=Poisson()
        this.mandatory_vars=Symbol[this.y_var]
        this.removed_SingleLevelVars=Symbol[]
        this.glm1_pvals_x=Symbol[]
        this.glm2_ZnVIF_x=Symbol[]
        this.glm3_PnSigns_x=Symbol[]
        this.glm4_PnSignsClean_x=Symbol[]
        
        this.exclude_vars=Symbol[:dol_per_trip_pos_p1,:nonbuyer_pre_p1,:buyer_pre_p1,:dol_per_trip_pre_p1]
        this.Buyer_Pos_P1_is1=true
        if this.Buyer_Pos_P1_is1  this.exclude_vars = vcat(this.exclude_vars, [:buyer_pos_p1] )  end 
        this.vars=setdiff(names(dfd),vcat(this.exclude_vars,[:iso,:whyo,:data_nb_ne_b, :data_b_e_nb ,:pen_reduction,:occ_reduction,:dolocc_reduction]))
    
        return this 
    end

end
mocc = MOcc(df_data,cfg)




#function rmSingleLevelVar!(dfd::DataFrame,m::MModel)
#    slv=FS_singleLevel(dfd,m.vars)
#    m.removed_SingleLevelVars = slv
#    m.vars=setdiff(m.vars,slv)
#end





function xVIF(g::xGLM)
    include("/home/rmadmin/.julia/v0.4/RegTools/src/diagnostics.jl")
    include("/home/rmadmin/.julia/v0.4/RegTools/src/misc.jl")
    include("/home/rmadmin/.julia/v0.4/RegTools/src/modsel.jl")
    vdf=vif(g.model)
    vdf[:vars] = convert(Array{Symbol}, vdf[:variable])
    g.sdf = join(g.sdf,vdf[[:vars,:vif]], on = :vars, kind=:outer)
end


function modelme(m::MModel,dfd::DataFrame) 
    required_vars=vcat([m.y_var,:panid,m.logvar],cfg[:random_demos],cfg[:random_campaigns],cfg[:scoring_vars])
    vars=setdiff(m.vars,required_vars)
    function rmVars(v::Array{Symbol})
        return setdiff(vars,v)
    end

    #SingleValue
    m.removed_SingleLevelVars=FS_singleLevel(dfd,vars)
    vars = rmVars(m.removed_SingleLevelVars)
    
    #PVals
    m.glm1_pvals = xGLM(dfd, m.dist, m.y_var, m.logvar, vars  )  
    g1=m.glm1_pvals
    m.glm1_pvals_x=g1.sdf[g1.sdf[:pval].>0.7,:vars]
    vars = rmVars(vcat(m.glm1_pvals_x, g1.xvars ) )
    
    #Z & Vif
    m.glm2_ZnVIF = xGLM(dfd, m.dist, m.y_var, m.logvar, vars  ) 
    g2=m.glm2_ZnVIF
    xVIF(g2)
    z = g2.sdf[g2.sdf[:zval].<1.96,:vars]
    v = g2.sdf[ !isna(g2.sdf[:vif])&(g2.sdf[:vif].>15),:vars]
    m.glm2_ZnVIF_x =intersect(z,v)
    vars = rmVars(vcat(m.glm2_ZnVIF_x, g2.xvars) )
    
    # Pvalue & Signs
    m.glm3_PnSigns = xGLM(dfd, m.dist, m.y_var, m.logvar, vars  )  # Pvalue & Signs
    g3=m.glm3_PnSigns
    neutralvars = setdiff(vars,vcat(cfg[:negativevars],cfg[:positivevars])) 
    neg=intersect(cfg[:negativevars],g3.sdf[ g3.sdf[:coef].<0,:vars])
    pos=intersect(cfg[:positivevars],g3.sdf[ g3.sdf[:coef].>0,:vars])
    varstokeep = intersect(vcat(neutralvars, pos,neg) ,  g3.sdf[ g3.sdf[:pval].<cfg[:pvalue_lvl] ,:vars] )
    m.glm3_PnSigns_x= setdiff(vars, varstokeep )
    vars = rmVars(vcat(m.glm3_PnSigns_x, g3.xvars) )
    
    
    # Clean Up Pvalue & Signs
    m.glm4_PnSignsClean = xGLM(dfd, m.dist, m.y_var, m.logvar, vars  )
    g4=m.glm4_PnSignsClean
    neutralvars = setdiff(vars,vcat(cfg[:negativevars],cfg[:positivevars])) 
    neg=intersect(cfg[:negativevars],g4.sdf[ g4.sdf[:coef].<0,:vars])
    pos=intersect(cfg[:positivevars],g4.sdf[ g4.sdf[:coef].>0,:vars])
    varstokeep = intersect(vcat(neutralvars, pos,neg) ,  g4.sdf[ g4.sdf[:pval].<cfg[:pvalue_lvl] ,:vars] )
    
    g4.sdf[ g4.sdf[:pval].<cfg[:pvalue_lvl] ,:vars]
    m.glm4_PnSignsClean_x = 
    #m.glm4 = xGLM(dfd, m, cfg)
    
    #m.vars=vcat(vars,required_vars)
end

modelme(mocc, df_data[(df_data[:iso].==false)&(df_data[:buyer_pos_p1].==1),:] )   

m=mocc
dfd=df_data[(df_data[:iso].==false)&(df_data[:buyer_pos_p1].==1),:]


# ==============================================================
"""

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










modelme(mocc,df_data)
#function glm2df(mdl::DataFrames.DataFrameRegressionModel)
#    return DataFrame(vars=vcat([:intercept],mdl.mf.terms.terms), coef=coef(mdl), se=stderr(mdl), zval=coef(mdl)./stderr(mdl) ,pval= ccdf(FDist(1, df_residual(mdl)), abs2(coef(mdl)./stderr(mdl))))
#end

#g.sdf=glm2df(g.model)









#model_glm_total <- glm(Formula_fixed,data=finaldata_occ,family=poisson(link="log"))


vars = unique(vcat( setdiff(m.vars, m.xFromGLM),[m.y_var, m.logvar]) )
logvar= m.logvar
y_var=m.y_var
runGLM(m.y_var, unique(vcat( setdiff(m.vars, m.xFromGLM),[m.y_var, m.logvar]) ) , m.logvar)
function runGLM(y_var::Symbol, vars::Array{Symbol}, logvar::Symbol=:Missing)
    #*"offset(log("*string(m.logvar)*"+1))"))
    #     vars = setdiff(vars,[:fea_trps_shr_dpp_p1 ,:fea_trps_shr_dpp_p2,:fea_trps_shr_dpp_p3,:fea_trps_shr_dpp_p4
        #,:fea_or_dis_trps_shr_dpp_p1,:fea_or_dis_trps_shr_dpp_p2,:fea_or_dis_trps_shr_dpp_p3,:fea_or_dis_trps_shr_dpp_p4] )    
    #cor(df_data[:fea_trps_shr_dpp_p1],df_data[:fea_or_dis_trps_shr_dpp_p1]) #cor(df_data[:fea_trps_shr_dpp_p2],df_data[:fea_or_dis_trps_shr_dpp_p2])
    #cor(df_data[:fea_trps_shr_dpp_p3],df_data[:fea_or_dis_trps_shr_dpp_p3]) #cor(df_data[:fea_trps_shr_dpp_p4],df_data[:fea_or_dis_trps_shr_dpp_p4])
       
    #vars=setdiff(vars,[:fea_trps_shr_dpp_p3,:fea_or_dis_trps_shr_dpp_p3,:fea_trps_shr_dpp_p4,:fea_or_dis_trps_shr_dpp_p4])
    xvars=Symbol[]
    gmx=nothing
    for l in 1:30
        vars=setdiff(vars,xvars)
        fmla = genFmula(y_var, vars, logvar  )
        try
            if logvar!=:Missing
                #fmla = genFmula(y_var, vars, logvar  )
                gmx = glm(fmla, df_data[vars], Poisson(), LogLink(), offset=log(Array(df_data[logvar]  + 1)) )
            else
                #fmla = genFmula(y_var, vars)
                gmx = glm(fmla, df_data[vars], Poisson() )
            end
            break
        catch e
            if isa(e, Base.LinAlg.PosDefException)
                v=vars[e.info-1]
                push!(xvars,v)
                println("!!! Multicollinearity, removing :",xvars,"~~~",e.info)
            else
                println("....",e)
                break
            end
        end
    end
    return gmx,xvars
end






function modelme(m::MOcc,df_data::DataFrame)    
    #Testing
    #vars=unique(vcat( setdiff(m.vars, m.xFromGLM),[m.y_var, m.logvar]) )
    #vars=setdiff(unique(vcat( setdiff(m.vars, m.xFromGLM),[m.y_var, m.logvar]) ) ,  [     :fea_or_dis_trps_shr_dpp_p1,:fea_or_dis_trps_shr_dpp_p2,:fea_or_dis_trps_shr_dpp_p3,:fea_or_dis_trps_shr_dpp_p4 ]  )
    
    #vars=setdiff(unique(vcat( setdiff(m.vars, m.xFromGLM),[m.y_var, m.logvar]) ) ,  [:fea_trps_shr_dpp_p1 ,:fea_trps_shr_dpp_p2,:fea_trps_shr_dpp_p3,:fea_trps_shr_dpp_p4 ]  )
    #Pkg.clone("https://github.com/joemliang/RegTools.jl")
#    /home/rmadmin/.julia/v0.4/RegTools/src
#    diagnostics.jl  misc.jl  modsel.jl
    
    
    #fmla = genFmula(m.y_var, vars, m.logvar  )
    # gmx = glm(fmla, df_data[vars], Poisson(), LogLink(), offset=log(Array(df_data[m.logvar]  + 1)) )
    # df_data[ df_data[:fea_trps_shr_dpp_p1].!=0 ,[:fea_trps_shr_dpp_p1, :fea_or_dis_trps_shr_dpp_p1] ]
     
    
    # 1***. drop variables with only one level - cater for buyer_pos_p1
    if m.Buyer_Pos_P1_is1
        m.singleLevel_vars = [c for c in filter(x -> length(unique( df_data[(df_data[:iso].==false)&(df_data[:buyer_pos_p1].==1),x] ))==1, m.vars)]
    else
        m.singleLevel_vars = [c for c in filter(x -> length(unique( df_data[(df_data[:iso].==false),x] ))==1, m.vars)]
    end
    m.vars = setdiff(m.vars,m.singleLevel_vars)
    mdl,xvars = runGLM(m.y_var, unique(vcat( setdiff(m.vars, m.xFromGLM),[m.y_var, m.logvar]) ) , m.logvar)
    m.GLM1_xvars=xvars
    p = glm2df(mdl)
    m.xpvals=vcat(m.xpvals,p[p[:pval].>0.7,:vars])
    
    
    include("/home/rmadmin/.julia/v0.4/RegTools/src/diagnostics.jl")
    include("/home/rmadmin/.julia/v0.4/RegTools/src/misc.jl")
    include("/home/rmadmin/.julia/v0.4/RegTools/src/modsel.jl")
    vif(mdl)
    
end




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