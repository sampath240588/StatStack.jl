#--------------------- MODEL ----------------------------------------

function readFixedFile(fname::AbstractString)
    if isfile(fname)
        idf = readtable(fname, header=true)
        dfnames = [:x,:Estimate,:Std_Error,:z_value,:Pr_z_]
        dfnames2 = [:x,:Estimate,:Std_Error,:t_value,:Pr_z_]
        dfnames3 = [:x,:Estimate,:Std_Error,:t_value,:Pr_t_]
        if (names(idf) == dfnames)|(names(idf) == dfnames2)|(names(idf) == dfnames3)
            idf[:x] = map(x-> lowercase(replace(replace(x,"(",""),")","")), idf[:x])
            names!(idf,[:x,:estimate,:std_error,:z_value,:pr_z_])
        else
            error("Invalid Headers in Fixed Src File :",fname)
        end
    else
        error("Can't find file: ",fname)
    end
    return idf
end


function readRandFile(fname::AbstractString, cfg::OrderedDict)
    defaultDF=DataFrame(class=Any[],row=Int64[],level=UTF8String[],B0=Float64[],B1=Float64[],levelse=Int64[],SE0=Float64[],SE1=Float64[],exposed_orig=Int64[],exposed=Any[],key=UTF8String[])

    if isfile(fname)
        idf = readtable(fname, header=true)
        dfnames = [:class,:row,:level,:group0,:group1,:levelSE,:SE_group0,:SE_group1,:exposed]
        dfnames2  = [:class,:row,:level,:group0,:group1,:levelSE,:SE_group0,:SE_group1,:exposed]   ## Same for now
        if (names(idf) == dfnames)|(names(idf) == dfnames2)
            idf[:class] = map(x-> lowercase(x), idf[:class])
            names!(idf,[:class,:row,:level,:B0,:B1,:levelse,:SE0,:SE1,:exposed_orig])
            idf[:exposed] = map(x-> x .==1 ? true:false, idf[:exposed_orig])
            # --- GROUP1 -----
#            idf[:Z1] = idf[:B1] ./ idf[:SE1]                         
#	        idf[:P1] = 2.0 * ccdf(Normal(), abs(idf[:Z1]))                      
#            idf[:SIG1] = map( x -> ((isnan(x))|(x > 0.2)) ?  false : true, idf[:P1] )        
#            # --- GROUP0 -----
#            idf[:Z0] = idf[:B0] ./ idf[:SE0]                          
#            idf[:P0] = 2.0 * ccdf(Normal(), abs(idf[:Z0]))                       
#            idf[:SIG0] = map( x -> (isnan(x)|(x > 0.2)) ?  false : true, idf[:P0] )  
            idf[:key] = map((c,l) ->  rkey(c,l), idf[:class],idf[:level])
            
            x=get(cfg,:excludedKeys, NA)
            if typeof(x) != NAtype
                for xv in x
                    idf=idf[idf[:key].!=xv,:]
                end
            end
            x=get(cfg,:excludedBreaks, NA)
            if typeof(x) != NAtype
                for xv in x
                    idf=idf[idf[:class].!=xv,:]
                end
            end
            x=get(cfg,:excludedLevels, NA)
            if typeof(x) != NAtype
                for xv in x
                    idf=idf[idf[:level].!=xv,:]
                end
            end
            
            
        else
            error("Invalid Headers in Random Src File",fname)
        end
    else
        #error("Can't find file: ",fname)
        println("Can't find file: ",fname)
        idf=defaultDF
    end
    return idf
end


abstract MModel 


function genScoreTotals(m::MModel, df_data::DataFrame, totalOnly::Bool)  # Still need to implement totalOnly -- cureently checks for reff_xxxx
    #println(methods(exp))
    if m.modelName == "occ"
        #df_data[:pre_occ_score0]=map(Float64,df_data[:pre_occ_score0])
        #df_data[:pre_occ_score1]=map(Float64,df_data[:pre_occ_score1])
        if :reff_occ in names(df_data)
            #df_data[:reff_occ] = map(Float64, df_data[:reff_occ])
            println("using reff : ",typeof(df_data[:pre_occ_score0]))
            df_data[:occ_score0] = exp(df_data[:pre_occ_score0] + df_data[:reff_occ]  ) 
            df_data[:occ_score1] = exp(df_data[:pre_occ_score1] + df_data[:reff_occ] ) 
        else
            df_data[:occ_score0] = exp(df_data[:pre_occ_score0]) 
            df_data[:occ_score1] = exp(df_data[:pre_occ_score1])
        end
    end    
    #---------------------
        #DataArrays
    if m.modelName == "dolocc"
        if :reff_dolocc in names(df_data)
            println("using reff")
            df_data[:dolocc_score0] = exp(df_data[:pre_dolocc_score0] + df_data[:reff_dolocc]  ) 
            df_data[:dolocc_score1] = exp(df_data[:pre_dolocc_score1] + df_data[:reff_dolocc] )
        else
            df_data[:dolocc_score0] = exp(df_data[:pre_dolocc_score0])
            df_data[:dolocc_score1] = exp(df_data[:pre_dolocc_score1]) 
        end
    end
    #----------------------
    if m.modelName == "pen"
        if :reff_pen in names(df_data)
            println("using reff")
            df_data[:pen_score0] = exp((df_data[:pre_pen_score0] + df_data[:reff_pen])) ./ ( exp((df_data[:pre_pen_score0] + df_data[:reff_pen])) +1)
            df_data[:pen_score1] = exp((df_data[:pre_pen_score1] + df_data[:reff_pen])) ./ (exp((df_data[:pre_pen_score1] + df_data[:reff_pen])) +1) 
        else
            df_data[:pen_score0] =  exp(df_data[:pre_pen_score0]) ./ ( exp(df_data[:pre_pen_score0]) +1)
            df_data[:pen_score1] = exp(df_data[:pre_pen_score1]) ./ ( exp(df_data[:pre_pen_score1]) +1)
        end
    end    
end



type MOcc <: MModel
    hasBreaks::Bool 
    fList::Vector{Symbol}
    rList::Vector{Symbol}
    dList::Vector{Symbol}
    feff::FOcc
    reff::ROcc
    fmula::AbstractString
    modelName::AbstractString
    logvar::AbstractString
    function MOcc(df_data::DataFrame,cfg::OrderedDict) this=new(); this.modelName="occ"; if   cfg[:offset]  this.logvar="trps_pre_p1"; end#this.sdf=pushSDFrow!(deepcopy(SDF),"Total Campaign",this.v_model);
        this.fList=Symbol[]; this.rList=Symbol[]; this.dList=Symbol[]
        #this.hasBreaks=false
        this.fmula=""
        idf=readFixedFile("Occasion/Occ_fixed_effects.csv")
        this.feff=FOcc(df_data, idf,cfg)
        idf=readRandFile("Occasion/Occ_random_effects.csv",cfg)
        this.reff=ROcc(df_data, idf)
        this.hasBreaks = length(idf[1])==0 ? false : true
        genScoreTotals(this,df_data,cfg[:TotalModelsOnly])
        return this 
    end

end





type MDolOcc <: MModel
    hasBreaks::Bool 
    fList::Vector{Symbol}
    rList::Vector{Symbol}
    dList::Vector{Symbol}
    feff::FDolOcc
    reff::RDolOcc
    fmula::AbstractString
    modelName::AbstractString
    logvar::AbstractString
    function MDolOcc(df_data::DataFrame,cfg::OrderedDict) this=new(); this.modelName="dolocc"; if  ( cfg[:offset])  this.logvar="dol_per_trip_pre_p1"; end #this.sdf=pushSDFrow!(deepcopy(SDF),"Total Campaign",this.v_model);
        this.fList=Symbol[]; this.rList=Symbol[]; this.dList=Symbol[]
        this.fmula=""
        idf=readFixedFile("DollarsOccasion/DolOcc_fixed_effects.csv")
        this.feff=FDolOcc(df_data, idf, cfg)
        idf=readRandFile("DollarsOccasion/DolOcc_random_effects.csv",cfg)
        this.reff=RDolOcc(df_data, idf)
        this.hasBreaks = length(idf[1])==0 ? false : true
        genScoreTotals(this,df_data,cfg[:TotalModelsOnly])
        return this 
    end

end



type MPen <: MModel
    hasBreaks::Bool 
    fList::Vector{Symbol}
    rList::Vector{Symbol}
    dList::Vector{Symbol}
    feff::FPen
    reff::RPen
    fmula::AbstractString
    modelName::AbstractString
    logvar::AbstractString
    function MPen(df_data::DataFrame,cfg::OrderedDict) this=new(); this.modelName="pen";if  ( cfg[:offset])  this.logvar="buyer_pre_p1"; end#this.sdf=pushSDFrow!(deepcopy(SDF),"Total Campaign",this.v_model);
        this.fList=Symbol[]; this.rList=Symbol[]; this.dList=Symbol[]
        this.fmula=""
        idf=readFixedFile("Penetration/Pen_fixed_effects.csv")
        this.feff=FPen(df_data, idf,cfg)
        idf=readRandFile("Penetration/Pen_random_effects.csv",cfg)
        this.reff=RPen(df_data, idf)
        this.hasBreaks = length(idf[1])==0 ? false : true
        genScoreTotals(this,df_data,cfg[:TotalModelsOnly])
        return this 
    end

end




type MDolHH
    sdf::DataFrame
    mocc::MOcc
    mdolocc::MDolOcc
    mpen::MPen
    modelName::AbstractString
    function MDolHH(mocc::MOcc,mdolocc::MDolOcc,mpen::MPen) this=new(); this.modelName="dolhh";  
        this.mocc=mocc
        this.mdolocc=mdolocc
        this.mpen=mpen
        this.sdf=pushSDFrow!(deepcopy(SDF),"Total Campaign",this.modelName);
        #by(vcat(mocc.reff.src[[:key, :class, :level,:exposed]], mdolocc.reff.src[[:key, :class, :level,:exposed]], mpen.reff.src[[:key, :class, :level,:exposed]]), [:key,:class,:level], df ->  maximum(df[:exposed]) )
        
        occlst = mocc.hasBreaks ? mocc.reff.src[:key] : DataArray[]
        dolocclst = mocc.hasBreaks ? mocc.reff.src[:key] : DataArray[]
        penlst = mocc.hasBreaks ? mocc.reff.src[:key] : DataArray[]
        
        for k in unique(vcat(occlst,dolocclst,penlst))
            pushSDFrow!(this.sdf,k,this.modelName);
        end
        #for k in unique(vcat(mocc.reff.src[[:key]], mdolocc.reff.src[[:key]], mpen.reff.src[[:key]]))[1]
        #    pushSDFrow!(this.sdf,k,this.modelName);
        #end
        
        delete!(this.sdf, :mean_score0)
        delete!(this.sdf, :mean_score1)
        delete!(this.sdf, :B)
        delete!(this.sdf, :SE)
        delete!(this.sdf, :P)
        delete!(this.sdf, :M)
        delete!(this.sdf, :Mt)
        delete!(this.sdf, :Mc)
        
        return this  
    end
end

function Base.show(io::IO, mdolhh::MDolHH) 
    show(io,mdolhh.sdf)
end

function Base.show(io::IO, mmodel::MModel) 
    show(io,fieldnames(mmodel))
end




function genCnts(df_data::DataFrame, mocc::MOcc,mdolocc::MDolOcc,mpen::MPen,TotalModelsOnly::Bool)  
    tot=DataFrame(key=["Total Campaign"],class=[""],level=[""],exposed=[true], M=[0.0],Mt=[0.0],Mc=[0.0],N=[0.0],Nt=[0.0],Nc=[0.0])
    tot[:M] =  length(df_data[ df_data[:buyer_pos_p1] .== 1 ,1])
    tot[:Mt] = length(df_data[ (df_data[:group] .== 1) & (df_data[:buyer_pos_p1] .== 1 )  ,1])
    tot[:Mc] = length(df_data[ (df_data[:group] .== 0) & (df_data[:buyer_pos_p1] .== 1 ) ,1])
    tot[:N] = length(df_data[1])    
    tot[:Nc] = length(df_data[ df_data[:group] .== 0 ,1])
    tot[:Nt] = length(df_data[ df_data[:group] .== 1 ,1])
    
    if !TotalModelsOnly
        randlist = vcat(mocc.reff.src[[:key, :class, :level,:exposed]],mdolocc.reff.src[[:key, :class, :level,:exposed]],mpen.reff.src[[:key, :class, :level,:exposed]])
        if length(randlist[1]) > 0
            sdf = by(randlist, [:key,:class,:level], df ->  maximum(df[:exposed]) )
            #sdf = by(vcat(mocc.reff.src[[:key, :class, :level,:exposed]],mdolocc.reff.src[[:key, :class, :level,:exposed]],mpen.reff.src[[:key, :class, :level,:exposed]]), [:key,:class,:level], df ->  maximum(df[:exposed]) )
            rename!(sdf,:x1,:exposed)
            sdf[:M]=0.0
            sdf[:Mt]=0.0
            sdf[:Mc]=0.0
            sdf[:N]=0.0
            sdf[:Nt]=0.0
            sdf[:Nc]=0.0
            for row in eachrow(sdf)
                k = row[:key]
                ranfx = row[:class]
                v_level = row[:level]
                exposed = row[:exposed] 
                sdf[sdf[:key].==k, :Mt] = length(df_data[ (df_data[:group] .== 1) & (df_data[:buyer_pos_p1] .== 1 ) & (df_data[symbol(ranfx)] .== v_level) ,1])
                sdf[sdf[:key].==k, :M] = length(df_data[ (df_data[:buyer_pos_p1] .== 1) & (df_data[symbol(ranfx)] .== v_level) ,1])     
                sdf[sdf[:key].==k, :Nt] = length(df_data[ (df_data[:group] .== 1) & (df_data[symbol(ranfx)] .== v_level) ,1])
                sdf[sdf[:key].==k, :N] = length(df_data[ (df_data[symbol(ranfx)] .== v_level) ,1])     
                if exposed    #  If Exposed - default Nc & Mc to total -- else count by rndfx
                    sdf[sdf[:key].==k, :Mc] = length(df_data[ (df_data[:group] .== 0) & (df_data[:buyer_pos_p1] .== 1 ) ,1])
                    sdf[sdf[:key].==k, :Nc] = length(df_data[ (df_data[:group] .== 0) ,1])
                    sdf[sdf[:key].==k, :M] = sdf[sdf[:key].==k, :Mt] + sdf[sdf[:key].==k, :Mc] 
                    sdf[sdf[:key].==k, :N] = sdf[sdf[:key].==k, :Nt] + sdf[sdf[:key].==k, :Nc]
                else
                    sdf[sdf[:key].==k, :Mc] = length(df_data[ (df_data[:group] .== 0) & (df_data[:buyer_pos_p1] .== 1 ) & (df_data[symbol(ranfx)] .== v_level) ,1])
                    sdf[sdf[:key].==k, :Nc] = length(df_data[ (df_data[:group] .== 0) & (df_data[symbol(ranfx)] .== v_level) ,1])
                end
                println("-- ",ranfx,"-- ",v_level,"-- ", exposed )
            end
            return vcat(tot,sdf)
            
        else
            return tot
        end
    else
        return tot
    end
end
#genCnts(df_data, mocc,mdolocc,mpen)

type Cnts
    sdf::DataFrame
#    get::Function
    function Cnts(df_data::DataFrame, mocc::MOcc,mdolocc::MDolOcc,mpen::MPen,TotalModelsOnly::Bool)  
        this = new()
        this.sdf = genCnts(df_data, mocc,mdolocc,mpen,TotalModelsOnly) 
#        function getk(k::AbstractString) 
#            return this.sdf[this.sdf[:key].==k,:] 
#        end
#        function getk(k::AbstractString,s::Symbol) 
#            return this.sdf[this.sdf[:key].==k,s][1]
#        end
#        this.get = getk
        return this
    end
end



function hhcounts(cnts::Cnts)
    sdf=cnts.sdf 
    sdf[:hh]=0.0
    sdf[:impressions]=0.0
    sdf[:weight]=0.0
    df_hhc = readtable("hhcounts.csv",header=true); lowercase(df_hhc)
    df_hhc[:class] = map(x->lowercase(x)  ,df_hhc[:class])
    for row in eachrow(df_hhc[df_hhc[:class].!="total campaign",:])
        k=rkey(row)
        sdf[sdf[:key].==k,:hh] = row[:hh]
        sdf[sdf[:key].==k,:impressions] = row[:impressions]
    end
    #tot = sum(sdf[:hh])
    tot = by(sdf,:class, df -> sum(df[:hh]))[:x1][2]
    toti = sum(sdf[:impressions])
    sdf[sdf[:key].=="Total Campaign",:hh] = tot
    sdf[sdf[:key].=="Total Campaign",:impressions] = toti
    for row in eachrow(cnts.sdf[cnts.sdf[:key].!="Total Campaign",:])
        k=rkey(row)
        sdf[sdf[:key].==k,:weight] = row[:hh] / tot
        println(typeof(row[:hh]))
        println(row[:hh])
    end
end


function hhcounts_OLD(cnts::Cnts)
    sdf=cnts.sdf
    sdf[:hh]=0
    sdf[:impressions]=0
    sdf[:weight]=0.0
    df_hhc = readtable("hhcounts.csv",header=true); lowercase(df_hhc)
    df_hhc[:class] = map(x->lowercase(x)  ,df_hhc[:class])
    tot = df_hhc[df_hhc[:class].=="total campaign",:hh][1]
    toti = df_hhc[df_hhc[:class].=="total campaign",:impressions][1]
    sdf[sdf[:key].=="Total Campaign",:hh] = tot
    sdf[sdf[:key].=="Total Campaign",:impressions] = toti
    for row in eachrow(df_hhc[df_hhc[:class].!="total campaign",:])
        k=rkey(row)
        #println(k,"-",row[:level],"-",row[:hh])
        sdf[sdf[:key].==k,:hh] = row[:hh]
        sdf[sdf[:key].==k,:impressions] = row[:impressions]
        sdf[sdf[:key].==k,:weight] = row[:hh] / tot
        println(typeof(row[:hh]))
    end
    #sum(cnts.sdf[:hh])
    class=""
    for row in eachrow(cnts.sdf[cnts.sdf[:key].!="Total Campaign",:])
        if row[:class] == class
            
        else
            class=row[:class]
        end
        println(row[:hh])
    end
end


function getk(sdf::DataFrame,k::AbstractString)
    return sdf[sdf[:key].==k,:]
end
function getk(sdf::DataFrame,k::AbstractString,s::Symbol) 
    return sdf[sdf[:key].==k,s][1]
end

#function getk(cnts::Cnts,k::AbstractString)
#    return cnts.sdf[cnts.sdf[:key].==k,:]
#end
#function getk(cnts::Cnts,k::AbstractString,s::Symbol) 
#    return cnts.sdf[cnts.sdf[:key].==k,s][1]
#end
   
#function getk(mdf::MDF,k::AbstractString,cnts::Cnts) 
#    mdf.sdf = genMDF(cnts,this.mocc,this.mdolocc, this.mpen)
#    return this.sdf[this.sdf[:key].==k,:] 
#end


    

function genRndMDF(cnts::Cnts, mocc::MOcc, mdolocc::MDolOcc, mpen::MPen, withNA::Bool=true, includeRawData::Bool=false) 
    mdfcols=[:key,:B1,:SE1,:P1,:o_mean_score0,:o_mean_score1,:inOcc,:isSig1,:B1_orig,:SE1_orig,:o_adj_mean_cntrl_grp, :o_adj_mean_expsd_grp,
        :B2,:SE2,:P2,:y_mean_score0,:y_mean_score1,:inDolOcc,:isSig2,:B2_orig,:SE2_orig,:y_adj_mean_cntrl_grp, :y_adj_mean_expsd_grp,
             :B3,:SE3,:P3,:p_mean_score0,:p_mean_score1,:inPen,:isSig3,:B3_orig,:SE3_orig,:p_adj_mean_cntrl_grp, :p_adj_mean_expsd_grp,
        :M,:Mt,:Mc,:N,:Nt,:Nc,:o_B0,:o_SE0,:y_B0,:y_SE0,:p_B0,:p_SE0
            ]
    if mocc.hasBreaks & mdolocc.hasBreaks & mpen.hasBreaks & isdefined(mocc, :reff) & isdefined(mdolocc, :reff) & isdefined(mpen, :reff)
        #or=deepcopy(mocc.reff.sdf[[:key,:B1_combo,:SE1,:SE1_combo,:P1_combo,:mean_score0,:mean_score1]])
        or=deepcopy(mocc.reff.sdf[[:key,:B1_combo,:SE1_combo,:P1_combo,:mean_score0,:mean_score1,:B0,:SE0,:B1,:SE1,:adj_mean_cntrl_grp, :adj_mean_expsd_grp]])  #:adj_mean_cntrl_grp, :adj_mean_expsd_grp 
        rename!(or,:B1,:B1_orig)
        rename!(or,:SE1,:SE1_orig)
        
        rename!(or,:B1_combo,:B1)
        rename!(or,:P1_combo,:P1)    
        #rename!(or,:SE1,:SE1)
        rename!(or,:SE1_combo,:SE1)    
        rename!(or,:B0,:o_B0)
        rename!(or,:SE0,:o_SE0)

        rename!(or,:mean_score0,:o_mean_score0)
        rename!(or,:mean_score1,:o_mean_score1)
        
        rename!(or,:adj_mean_cntrl_grp,:o_adj_mean_cntrl_grp); rename!(or,:adj_mean_expsd_grp ,:o_adj_mean_expsd_grp)
        
        or[:inOcc] = true
        or[:isSig1] = true
    
        #yr=deepcopy(mdolocc.reff.sdf[[:key,:B1_combo,:SE1,:SE1_combo,:P1_combo,:mean_score0,:mean_score1]])
        yr=deepcopy(mdolocc.reff.sdf[[:key,:B1_combo,:SE1_combo,:P1_combo,:mean_score0,:mean_score1,:B0,:SE0,:B1,:SE1,:adj_mean_cntrl_grp, :adj_mean_expsd_grp]])
        rename!(yr,:B1,:B2_orig)
        rename!(yr,:SE1,:SE2_orig)
        
        rename!(yr,:B1_combo,:B2)
        rename!(yr,:P1_combo,:P2)    
        #rename!(yr,:SE1,:SE2)
        rename!(yr,:SE1_combo,:SE2)
        rename!(yr,:B0,:y_B0)
        rename!(yr,:SE0,:y_SE0)
        
        rename!(yr,:mean_score0,:y_mean_score0)
        rename!(yr,:mean_score1,:y_mean_score1)
        
        rename!(yr,:adj_mean_cntrl_grp,:y_adj_mean_cntrl_grp); rename!(yr, :adj_mean_expsd_grp ,:y_adj_mean_expsd_grp)
        
        yr[:inDolOcc] = true
        yr[:isSig2] = true
    
        #pr=deepcopy(mpen.reff.sdf[[:key,:B1_combo,:SE1,:SE1_combo,:P1_combo,:mean_score0,:mean_score1]])
        pr=deepcopy(mpen.reff.sdf[[:key,:B1_combo,:SE1_combo,:P1_combo,:mean_score0,:mean_score1,:B0,:SE0,:B1,:SE1,:adj_mean_cntrl_grp, :adj_mean_expsd_grp]])
        rename!(pr,:B1,:B3_orig)
        rename!(pr,:SE1,:SE3_orig)
        
        rename!(pr,:B1_combo,:B3)
        rename!(pr,:P1_combo,:P3)    
        #rename!(pr,:SE1,:SE3)
        rename!(pr,:SE1_combo,:SE3)
                
        rename!(pr,:B0,:p_B0)
        rename!(pr,:SE0,:p_SE0)
        
        rename!(pr,:mean_score0,:p_mean_score0)
        rename!(pr,:mean_score1,:p_mean_score1)
        
        rename!(pr,:adj_mean_cntrl_grp,:p_adj_mean_cntrl_grp); rename!(pr, :adj_mean_expsd_grp ,:p_adj_mean_expsd_grp)
        
        pr[:inPen] = true
        pr[:isSig3] = true  
        
        ztmp= join(join(or,yr, on = :key, kind = :left),pr, on = :key, kind = :outer)   #:left
        z=join(ztmp,cnts.sdf[[:key,:M,:Mt,:Mc,:N,:Nt,:Nc,]], on = :key)[mdfcols]
        #z= join(join(or,yr, on = :key, kind = :left),pr, on = :key, kind = :left)[mdfcols]
        #showcols(z)
        z[find(isna(z[:inOcc])), :inOcc] = false
        z[find(isna(z[:isSig1])), :isSig1] = false 
        z[find(isna(z[:inDolOcc])), :inDolOcc] = false   
        z[find(isna(z[:isSig2])), :isSig2] = false   
        z[find(isna(z[:inPen])), :inPen] = false  
        z[find(isna(z[:isSig3])), :isSig3] = false   
        #t=join(z,rcnts.sdf[[:key,:M,:Mt,:Mc,:N,:Nt,:Nc]], on = :key)
        #t[[:M,:M_1,:Mt,:Mt_1,:Mc,:Mc_1,:N,:N_1,:Nt,:Nt_1,:Nc,:Nc_1]]
        
        
        z[find(isna(z[:o_B0])), :o_B0] = 0.0
        z[find(isna(z[:o_SE0])), :o_SE0] = 0.0 
        z[find(isna(z[:y_B0])), :y_B0] = 0.0
        z[find(isna(z[:y_SE0])), :y_SE0] = 0.0 
        z[find(isna(z[:p_B0])), :p_B0] = 0.0
        z[find(isna(z[:p_SE0])), :p_SE0] = 0.0 
        
        xdf=z
        
        #if !withNA
            #for c in 2:length(z)
            #    r=xdf[1,c]
	        #    xdf[find(isna(xdf[c])), c]=r
            #end
            #for i in 1:length(xdf)
            #    o_fval=mocc.reff.sdf[names(mocc.reff.sdf)[i]][1]
            #    y_fval=mdolocc.reff.sdf[names(mdolocc.reff.sdf)[i]][1]
            #    p_fval=mpen.reff.sdf[names(mpen.reff.sdf)[i]][1]
            #    println(i," ~~ ",o_fval, " - " ,y_fval, " - " ,p_fval )
            #end
            
            #for i in mdfcols
            #    r=
            #    xdf[find(isna(xdf[c])), c]=r
            #end
            
            
            
        #end
    end
    xdf = join(xdf, cnts.sdf[[:key,:class,:level,:exposed]], on = :key)
    return xdf
end
#genRndMDF(cnts, mocc, mdolocc, mpen, false) 


function genFixedMDF(cnts::Cnts, mocc::MOcc, mdolocc::MDolOcc, mpen::MPen) 
    o=deepcopy(mocc.feff.sdf[[:key,:B,:SE,:P,:unadj_mean_score0,:unadj_mean_score1,:M,:Mt,:Mc,:adj_mean_expsd_grp,:adj_mean_cntrl_grp]])
    rename!(o,:B,:B1)
    rename!(o,:P,:P1)    
    rename!(o,:SE,:SE1)
    #rename!(o,:unadj_mean_score0,:o_mean_score0)
    #rename!(o,:unadj_mean_score1,:o_mean_score1)
    rename!(o,:adj_mean_cntrl_grp,:o_mean_score0)
    rename!(o,:adj_mean_expsd_grp,:o_mean_score1)
    o[:inOcc] = true
    o[:isSig1] = true
      o[:B1_orig]=o[:B1]
      o[:SE1_orig]=o[:SE1]
    
    y=deepcopy(mdolocc.feff.sdf[[:key,:B,:SE,:P,:unadj_mean_score0,:unadj_mean_score1,:adj_mean_expsd_grp,:adj_mean_cntrl_grp]])
    rename!(y,:B,:B2)
    rename!(y,:P,:P2)    
    rename!(y,:SE,:SE2)
    #rename!(y,:unadj_mean_score0,:y_mean_score0)
    #rename!(y,:unadj_mean_score1,:y_mean_score1)
    rename!(y,:adj_mean_cntrl_grp,:y_mean_score0)
    rename!(y,:adj_mean_expsd_grp,:y_mean_score1)
    y[:inDolOcc] = true
    y[:isSig2] = true
       y[:B2_orig]=y[:B2]
       y[:SE2_orig]=y[:SE2]
    
    p=deepcopy(mpen.feff.sdf[[:key,:B,:SE,:P,:unadj_mean_score0,:unadj_mean_score1,:M,:Mt,:Mc,:adj_mean_expsd_grp,:adj_mean_cntrl_grp]])
    rename!(p,:B,:B3)
    rename!(p,:P,:P3)    
    rename!(p,:SE,:SE3)
    #rename!(p,:unadj_mean_score0,:p_mean_score0)
    #rename!(p,:unadj_mean_score1,:p_mean_score1)
    rename!(p,:adj_mean_cntrl_grp,:p_mean_score0)
    rename!(p,:adj_mean_expsd_grp,:p_mean_score1)
    
    rename!(p,:M,:N)
    rename!(p,:Mt,:Nt)
    rename!(p,:Mc,:Nc)
    p[:inPen] = true
    p[:isSig3] = true
    p[:B3_orig]=p[:B3]
    p[:SE3_orig]=p[:SE3] 
    
    mdfcols=[:key,:B1,:SE1,:P1,:o_mean_score0,:o_mean_score1,:inOcc,:isSig1,:B1_orig,:SE1_orig,
             :B2,:SE2,:P2,:y_mean_score0,:y_mean_score1,:inDolOcc,:isSig2,:B2_orig,:SE2_orig,
             :B3,:SE3,:P3,:p_mean_score0,:p_mean_score1,:inPen,:isSig3,:B3_orig,:SE3_orig,
             :M,:Mt,:Mc,:N,:Nt,:Nc
            ]
    xdf= join(join(o,y, on = :key),p, on = :key)[mdfcols]
    xdf = join(xdf, cnts.sdf[[:key,:class,:level]], on = :key)
    return xdf
end
#genFixedMDF(cnts, mocc, mdolocc, mpen)



function genMDF(cnts::Cnts, mocc::MOcc, mdolocc::MDolOcc, mpen::MPen) 
    mdfcols=[:key,:B1,:SE1,:P1,:o_mean_score0,:o_mean_score1,:inOcc,:isSig1,:B1_orig,:SE1_orig,
             :B2,:SE2,:P2,:y_mean_score0,:y_mean_score1,:inDolOcc,:isSig2,:B2_orig,:SE2_orig,
             :B3,:SE3,:P3,:p_mean_score0,:p_mean_score1,:inPen,:isSig3,:B3_orig,:SE3_orig,
             :M,:Mt,:Mc,:N,:Nt,:Nc
            ]
    
    fmdf=genFixedMDF(cnts, mocc, mdolocc, mpen)
    if length(cnts.sdf[1]) > 1
        rmdf=genRndMDF(cnts, mocc, mdolocc, mpen)
        xdf = vcat(fmdf,rmdf)[mdfcols]
    else
        xdf = fmdf
    end

    #for i in [:B1_orig,:SE1_orig,:B2_orig,:SE2_orig,:B3_orig,:SE3_orig]
    #    xdf[find(isna(xdf[i])), i]=0.0
    #end
    
    for i in mdfcols   # default to total
        r=xdf[1,i]
        xdf[find(isna(xdf[i])), i]=r 
    end
    
    return xdf
end
#genMDF(cnts, mocc, mdolocc, mpen)


 
type MDF 
    sdf::DataFrame
    withNA::Bool
    mocc::MOcc
    mdolocc::MDolOcc
    mpen::MPen
    cnts::Cnts
#    get::Function
    function MDF(cnts::Cnts,mocc::MOcc,mdolocc::MDolOcc,mpen::MPen,withNA::Bool=true) 
        this=new() 
        this.cnts = cnts
        this.mocc = mocc
        this.mdolocc = mdolocc
        this.mpen = mpen
        this.withNA=withNA
#        function getk(k::AbstractString) 
#            #if !isdefined(this, :sdf)
#            #    this.sdf = genMDF(this.cnts,this.mocc,this.mdolocc, this.mpen, false)
#            #end
#            this.sdf = genMDF(this.cnts,this.mocc,this.mdolocc, this.mpen)
#            return this.sdf[this.sdf[:key].==k,:] 
#        end
                
#        this.get = getk
        return this
    end
end


function Base.show(io::IO, mdf::MDF)
    mdf.sdf = genMDF(mdf.cnts,mdf.mocc,mdf.mdolocc, mdf.mpen)
    show(io,mdf.sdf)
end




#function genRDF_OLD(mocc::MOcc,mdolocc::MDolOcc,mpen::MPen, mdolhh::MDolHH)
#    rdfFmt=[:key,:model,
#            :unadj_avg_expsd_hh_pre,:unadj_avg_cntrl_hh_pre,:unadj_avg_expsd_hh_pst,:unadj_avg_cntrl_hh_pst,:adj_mean_expsd_grp,:adj_mean_cntrl_grp,
#            :adj_dod_effct,:twotail_pval,:onetail_pval,
#            :onetail_80_pct_intrvl_lb,:onetail_80_pct_intrvl_ub,:twotail_80_pct_intrvl_lb,:twotail_80_pct_intrvl_ub,
#            :onetail_90_pct_intrvl_lb,:onetail_90_pct_intrvl_ub,:twotail_90_pct_intrvl_lb,:twotail_90_pct_intrvl_ub
#           ]
#    x=vcat(mocc.feff.sdf[rdfFmt], mdolocc.feff.sdf[rdfFmt],mpen.feff.sdf[rdfFmt], mdolhh.sdf[1,rdfFmt])
#    if isdefined(mocc, :reff)
#        #x=vcat(x, mocc.reff.sdf[rdfFmt], mdolocc.reff.sdf[rdfFmt], mpen.reff.sdf[rdfFmt], mdolhh.sdf[2:end,rdfFmt])
#        x=vcat(x,mocc.reff.sdf[map(x -> contains(x, "(none)") ? 1:0, mocc.reff.sdf[:key]) .== 0,:][rdfFmt],
#                 mdolocc.reff.sdf    [map(x -> contains(x, "(none)") ? 1:0, mdolocc.reff.sdf[:key]) .== 0,:]     [rdfFmt], 
#                 mpen.reff.sdf       [map(x -> contains(x, "(none)") ? 1:0, mpen.reff.sdf[:key]) .== 0,:]     [rdfFmt], 
#                 mdolhh.sdf          [map(x -> contains(x, "(none)") ? 1:0, mdolhh.sdf[:key]) .== 0,:]     [2:end,rdfFmt]
#              )
#    end
#    sdf=sort(x, cols = :key)
#end
##genRDF(mocc,mdolocc,mpen, mdolhh)


function genRDF(mocc::MOcc,mdolocc::MDolOcc,mpen::MPen, mdolhh::MDolHH,cfg::OrderedDict=OrderedDict(:TotalModelsOnly => false))
    rdfFmt=[:key,:model,
            :unadj_avg_expsd_hh_pre,:unadj_avg_cntrl_hh_pre,:unadj_avg_expsd_hh_pst,:unadj_avg_cntrl_hh_pst,:adj_mean_expsd_grp,:adj_mean_cntrl_grp,
            :adj_dod_effct,:twotail_pval,:onetail_pval,
            :onetail_80_pct_intrvl_lb,:onetail_80_pct_intrvl_ub,:twotail_80_pct_intrvl_lb,:twotail_80_pct_intrvl_ub,
            :onetail_90_pct_intrvl_lb,:onetail_90_pct_intrvl_ub,:twotail_90_pct_intrvl_lb,:twotail_90_pct_intrvl_ub
           ]
    x=vcat(mocc.feff.sdf[rdfFmt], mdolocc.feff.sdf[rdfFmt],mpen.feff.sdf[rdfFmt], mdolhh.sdf[1,rdfFmt])
    
    if !cfg[:TotalModelsOnly]
    #if isdefined(mocc, :reff)
        #x=vcat(x,mocc.reff.sdf[map(x -> contains(x, "(none)") ? 1:0, mocc.reff.sdf[:key]) .== 0,:][rdfFmt],
        #         mdolocc.reff.sdf[map(x -> contains(x, "(none)") ? 1:0, mdolocc.reff.sdf[:key]) .== 0,:][rdfFmt], 
        #         mpen.reff.sdf[map(x -> contains(x, "(none)") ? 1:0, mpen.reff.sdf[:key]) .== 0,:][rdfFmt], 
        #         mdolhh.sdf[map(x -> contains(x, "(none)") ? 1:0, mdolhh.sdf[:key]) .== 0,:][2:end,rdfFmt]
        #      )
        z=vcat(mocc.reff.sdf[rdfFmt], mdolocc.reff.sdf[rdfFmt], mpen.reff.sdf[rdfFmt],mdolhh.sdf[2:end,rdfFmt])
        if length(z[1]) > 0  x=vcat(x,z) end
    end
    x2 = similar(x,0)
    for k in unique(x[:key])
        for m in ["occ","dolocc","pen","dolhh"]
            len = length(x[ (x[:model].== m) & (x[:key].== k)  ,:key])
            if len == 1
                v = collect(values(df2dict(x[(x[:model].== m) & (x[:key].== k),:])))
                push!(x2, v)                       
            else
                push!(x2, [k;m; 
                           x[(x[:key].=="Total Campaign") & (x[:model].==m),:unadj_avg_expsd_hh_pre];
                           x[(x[:key].=="Total Campaign") & (x[:model].==m),:unadj_avg_cntrl_hh_pre];
                           x[(x[:key].=="Total Campaign") & (x[:model].==m),:unadj_avg_expsd_hh_pst];
                           x[(x[:key].=="Total Campaign") & (x[:model].==m),:unadj_avg_cntrl_hh_pst];
                           x[(x[:key].=="Total Campaign") & (x[:model].==m),:adj_mean_expsd_grp];
                           x[(x[:key].=="Total Campaign") & (x[:model].==m),:adj_mean_cntrl_grp];
                           x[(x[:key].=="Total Campaign") & (x[:model].==m),:adj_dod_effct];
                           x[(x[:key].=="Total Campaign") & (x[:model].==m),:twotail_pval];
                           x[(x[:key].=="Total Campaign") & (x[:model].==m),:onetail_pval];
                           NaN;  #:onetail_80_pct_intrvl_lb
                           NaN;  #:onetail_80_pct_intrvl_ub
                           NaN;  #:twotail_80_pct_intrvl_lb
                           NaN;  #:twotail_80_pct_intrvl_ub
                           NaN;  #:onetail_90_pct_intrvl_lb
                           NaN;  #:onetail_90_pct_intrvl_ub
                           NaN;  #:twotail_90_pct_intrvl_lb
                           NaN   #:twotail_90_pct_intrvl_ub
                          ]
                    )             
            end
        end    
    end    
    sdf=sort(x2, cols = :key)
end


type RDF 
    sdf::DataFrame
    mocc::MOcc
    mdolocc::MDolOcc
    mpen::MPen
    mdolhh::MDolHH
    function RDF(mocc::MOcc,mdolocc::MDolOcc,mpen::MPen,mdolhh::MDolHH) 
        this=new() 
        this.mocc = mocc
        this.mdolocc = mdolocc
        this.mpen = mpen
        this.mdolhh = mdolhh
        return this
    end
end

function Base.show(io::IO, rdf::RDF)
    #x=vcat(rdf.mocc.feff.sdf[rdfFmt], rdf.mdolocc.feff.sdf[rdfFmt],rdf.mpen.feff.sdf[rdfFmt],rdf.mdolhh.sdf[1,rdfFmt])
    #if isdefined(rdf.mocc, :reff)
    #    x=vcat(x, rdf.mocc.reff.sdf[rdfFmt], rdf.mdolocc.reff.sdf[rdfFmt], rdf.mpen.reff.sdf[rdfFmt], rdf.mdolhh.sdf[2:end,rdfFmt])
    #end
    #rdf.sdf=sort(x, cols = :key)
    rdf.sdf=genRDF(rdf.mocc,rdf.mdolocc,rdf.mpen, rdf.mdolhh)
    show(io,rdf.sdf)
end


"""
# scoreme file
function aggregate!(df_data::DataFrame, focc::FOcc, cnts::Cnts)
    sdf = focc.sdf
    sdf[:M] = cnts.get("Total Campaign")[:M][1]
    sdf[:Mt] = cnts.get("Total Campaign")[:Mt][1]
    sdf[:Mc] = cnts.get("Total Campaign")[:Mc][1]
#    if :reff_occ in names(df_data)
#        println("using reff")
#        df_data[:occ_score0] = exp(df_data[:pre_occ_score0] + df_data[:reff_occ]  ) 
#        df_data[:occ_score1] = exp(df_data[:pre_occ_score1] + df_data[:reff_occ] ) 
#    else
#        df_data[:occ_score0] = exp(df_data[:pre_occ_score0]) 
#        df_data[:occ_score1] = exp(df_data[:pre_occ_score1])
#    end
    sdf[:mean_score0]=mean( df_data[ df_data[:buyer_pos_p1] .== 1 , :occ_score0] )
    sdf[:mean_score1]=mean( df_data[ df_data[:buyer_pos_p1] .== 1 , :occ_score1] ) 
    
    sdf[:unadj_mean_score0]=mean(df_data[(df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 0),:occ_score0])
    sdf[:unadj_mean_score1]=mean(df_data[(df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 1),:occ_score1])

    sdf[:adj_mean_cntrl_grp ] = mean( df_data[ df_data[:buyer_pos_p1] .== 1 , :occ_score0] )                     
    sdf[:adj_mean_expsd_grp ] = mean( df_data[ df_data[:buyer_pos_p1] .== 1 , :occ_score1] )
    # -------  Occ Score
    sdf[:unadj_avg_cntrl_hh_pre] = mean(df_data[ (df_data[:buyer_pre_p1] .== 1) & (df_data[:group] .== 0), :trps_pre_p1] )  
    sdf[:unadj_avg_expsd_hh_pre] = mean(df_data[ (df_data[:buyer_pre_p1] .== 1) & (df_data[:group] .== 1), :trps_pre_p1] )
    sdf[:unadj_avg_cntrl_hh_pst] = mean(df_data[ (df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 0), :trps_pos_p1] )
    sdf[:unadj_avg_expsd_hh_pst] = mean(df_data[ (df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 1), :trps_pos_p1] )
end



function aggregate!(df_data::DataFrame, fdolocc::FDolOcc, cnts::Cnts)
    sdf = fdolocc.sdf
    sdf[:M] = cnts.get("Total Campaign")[:M][1]
    sdf[:Mt] = cnts.get("Total Campaign")[:Mt][1]
    sdf[:Mc] = cnts.get("Total Campaign")[:Mc][1]
#    if :reff_dolocc in names(df_data)
#        println("using reff")
#        df_data[:dolocc_score0] = exp(df_data[:pre_dolocc_score0] + df_data[:reff_dolocc]  ) 
#        df_data[:dolocc_score1] = exp(df_data[:pre_dolocc_score1] + df_data[:reff_dolocc] )
#    else
#        df_data[:dolocc_score0] = exp(df_data[:pre_dolocc_score0])
#        df_data[:dolocc_score1] = exp(df_data[:pre_dolocc_score1]) 
#    end
    sdf[:mean_score0]=mean( df_data[ df_data[:buyer_pos_p1] .== 1 , :dolocc_score0] )
    sdf[:mean_score1]=mean( df_data[ df_data[:buyer_pos_p1] .== 1 ,  :dolocc_score1] )
    
    sdf[:unadj_mean_score0]=mean(df_data[(df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 0),:dolocc_score0])
    sdf[:unadj_mean_score1]=mean(df_data[(df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 1),:dolocc_score1])
    
    sdf[:adj_mean_cntrl_grp ] =  mean( df_data[ df_data[:buyer_pos_p1] .== 1 , :dolocc_score0] )             
    sdf[:adj_mean_expsd_grp ] =  mean( df_data[ df_data[:buyer_pos_p1] .== 1 ,  :dolocc_score1] )            
    # -------  DolOcc Score
    sdf[:unadj_avg_cntrl_hh_pre] =  mean(df_data[ (df_data[:buyer_pre_p1] .== 1) & (df_data[:group] .== 0), :dol_per_trip_pre_p1] )
    sdf[:unadj_avg_expsd_hh_pre] =  mean(df_data[ (df_data[:buyer_pre_p1] .== 1) & (df_data[:group] .== 1), :dol_per_trip_pre_p1] )
    sdf[:unadj_avg_cntrl_hh_pst] =  mean(df_data[ (df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 0), :dol_per_trip_pos_p1] )
    sdf[:unadj_avg_expsd_hh_pst] =  mean(df_data[ (df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 1), :dol_per_trip_pos_p1] )
end



function aggregate!(df_data::DataFrame, fpen::FPen, cnts::Cnts)
    sdf = fpen.sdf
    sdf[:M] = cnts.get("Total Campaign")[:N][1]
    sdf[:Mt] = cnts.get("Total Campaign")[:Nt][1]
    sdf[:Mc] = cnts.get("Total Campaign")[:Nc][1]
#    if :reff_pen in names(df_data)
#        println("using reff")
#        df_data[:pen_score0] = exp((df_data[:pre_pen_score0] + df_data[:reff_pen])) ./ ( exp((df_data[:pre_pen_score0] + df_data[:reff_pen])) +1)
#        df_data[:pen_score1] = exp((df_data[:pre_pen_score1] + df_data[:reff_pen])) ./ (exp((df_data[:pre_pen_score1] + df_data[:reff_pen])) +1) 
#    else
#        df_data[:pen_score0] =  exp(df_data[:pre_pen_score0]) ./ ( exp(df_data[:pre_pen_score0]) +1)
#        df_data[:pen_score1] = exp(df_data[:pre_pen_score1]) ./ ( exp(df_data[:pre_pen_score1]) +1)
#    end
    sdf[:mean_score0]=mean( df_data[ :pen_score0] )
    sdf[:mean_score1]=mean( df_data[ :pen_score1] )
    sdf[:unadj_mean_score0]=mean(df_data[(df_data[:group] .== 0),:pen_score0])
    sdf[:unadj_mean_score1]=mean(df_data[(df_data[:group] .== 1),:pen_score1])
    
    sdf[:adj_mean_cntrl_grp ] = mean( df_data[ :pen_score0] )             
    sdf[:adj_mean_expsd_grp ] = mean( df_data[ :pen_score1] )            
    # -------  Pen Score   - buyer_pre_p1
    sdf[:unadj_avg_cntrl_hh_pre ] = mean(df_data[ (df_data[:group] .== 0)  , :buyer_pre_p1] )
    sdf[:unadj_avg_expsd_hh_pre ] = mean(df_data[ (df_data[:group] .== 1)  , :buyer_pre_p1] )
    sdf[:unadj_avg_cntrl_hh_pst ] = mean(df_data[ (df_data[:group] .== 0)  , :buyer_pos_p1] )
    sdf[:unadj_avg_expsd_hh_pst ] = mean(df_data[ (df_data[:group] .== 1)  , :buyer_pos_p1] )
end




function aggregateCommon!(fixedeff::FixedEffect)
    sdf = fixedeff.sdf
    #src = fixedeff.src
    sdf[:adj_dod_effct] = ((sdf[:adj_mean_expsd_grp] - sdf[:adj_mean_cntrl_grp])  ./ sdf[:adj_mean_cntrl_grp] ) * 100   
    sdf[:twotail_pval] = sdf[:P]   #twotail_pval      
    sdf[:onetail_pval] = 1 - (sdf[:twotail_pval] ./ 2)   #onetail_pval
    sdf[:twotail_pval] = 1-sdf[:P]    #redo :twotail_pval
    #For NON-SIG Models, update - adj_mean_expsd_grp (to = adj_mean_cntrl_grp) & adj_dod_effct (= 0)
    sdf[(sdf[:twotail_pval] .< 0.80) & (sdf[:twotail_pval] .!= 0),:adj_mean_expsd_grp]      =  sdf[(sdf[:twotail_pval] .< 0.80) & (sdf[:twotail_pval] .!= 0),:adj_mean_cntrl_grp]
end

function ConfidenceIntervals(feff::FixedEffect)
    sdf = feff.sdf
    for row in eachrow(sdf)
            k = row[:key]
            md = df2dict(row)    
            md[:mean_score0] = md[:unadj_mean_score0];
            md[:mean_score1] = md[:unadj_mean_score1];
            cis =  CIs(md, ZDict, 0)
            for (zscore_key,zscore) in  ZDict     
                    ubk = symbol(zscore_key*"_ub")
                    lbk = symbol(zscore_key*"_lb")
                    sdf[sdf[:key].==k, lbk] = md[lbk]*100
                    sdf[sdf[:key].==k, ubk] = md[ubk]*100
            end
    end
end


# === Random Effects ====
function init!(cnts::Cnts, m::MModel)  
        src=m.reff.src
        # --- GROUP1 -----
        src[:Z1] = src[:B1] ./ src[:SE1]
        src[:P1] = 2.0 * ccdf(Normal(), abs(src[:Z1]))
        src[:SIG1] = map( x -> ((isnan(x))|(x > 0.2)) ?  false : true, src[:P1] ) 
        # --- GROUP0 -----
        src[:Z0] = src[:B0] ./ src[:SE0]                          
        src[:P0] = 2.0 * ccdf(Normal(), abs(src[:Z0]))                       
        src[:SIG0] = map( x -> (isnan(x)|(x > 0.2)) ?  false : true, src[:P0] )  
        
        m.reff.sdf = join(m.reff.sdf,m.reff.src[[:class,:level,:B0,:B1,:SE0,:SE1,:exposed,:P1,:key]], on = :key)
        sdf=m.reff.sdf        
        sdf[:B_fixed] = m.feff.sdf[1,:B]
        sdf[:P_fixed] = m.feff.sdf[1,:P]
        sdf[:SE_fixed] = m.feff.sdf[1,:SE]
        #sdf[:B1_combo] = sdf[:B1] + sdf[:B_fixed] #- sdf[:B0]
        #sdf[:SE1_combo] = sqrt(sdf[:SE_fixed].^2+sdf[:SE1].^2)
        sdf[:B1_combo] = sdf[:B1] + sdf[:B_fixed] - sdf[:B0]
        sdf[:SE1_combo] = sqrt(sdf[:SE_fixed].^2+sdf[:SE1].^2+sdf[:SE0].^2)
        sdf[:P1_combo] = 2.0 * ccdf(Normal(), abs( (sdf[:B1_combo]) ./ sdf[:SE1_combo])) 
        
        if m.reff.v_model == "pen"
            for k in sdf[:key]     
                sdf[sdf[:key].==k ,:M] = cnts.get(k)[:N][1]
                sdf[sdf[:key].==k ,:Mt] = cnts.get(k)[:Nt][1]
                sdf[sdf[:key].==k ,:Mc] = cnts.get(k)[:Nc][1]
            end  
        else
            for k in sdf[:key]     
                sdf[sdf[:key].==k ,:M] = cnts.get(k)[:M][1]
                sdf[sdf[:key].==k ,:Mt] = cnts.get(k)[:Mt][1]
                sdf[sdf[:key].==k ,:Mc] = cnts.get(k)[:Mc][1]
            end   
        end 
end



function aggregate!(df_data::DataFrame, mocc::MOcc)
    sdf=mocc.reff.sdf
    for row = eachrow(mocc.reff.sdf)
        k = row[:key]
        ranfx = lowercase(row[:class])
        sranfx = symbol(ranfx)
        v_level = row[:level]
        exposed = row[:exposed]         

        dest_colname0=symbol(k*"_occ_0")
        dest_colname1=symbol(k*"_occ_1")
        xcol=symbol("reff_"*ranfx*"_occ")   # Column to exclude
        fmula0="df_data[:pre_occ_score0]+"*getRandomFormula(mocc.reff.rcols, xcol)*"+"*string(row[:B0])
        fmula0=replace(fmula0,"++","+")    #if only one col
        fmula1="df_data[:pre_occ_score1]+"*getRandomFormula(mocc.reff.rcols, xcol)*"+"*string(row[:B1])
        fmula1=replace(fmula1,"++","+")
        
        df_data[dest_colname0] = exp(eval(parse(fmula0)))
        df_data[dest_colname1] = exp(eval(parse(fmula1)))
        println("\n\n\n",xcol," ~~ ",fmula1)
        println("occ ~ ",k," --> ",dest_colname0)  

        mean_score0 = mean(df_data[df_data[:buyer_pos_p1] .== 1,  dest_colname0 ])
        mean_score1 = mean(df_data[ df_data[:buyer_pos_p1] .== 1 , dest_colname1 ])
        sdf[sdf[:key].==k ,:mean_score0] = mean_score0
        sdf[sdf[:key].==k ,:mean_score1] = mean_score1
        sdf[sdf[:key].==k ,:adj_mean_cntrl_grp] = mean_score0
        sdf[sdf[:key].==k ,:adj_mean_expsd_grp] = mean_score1
        
        
        #sdf[sdf[:key].==k ,:unadj_avg_expsd_hh_pre] = mean(df_data[ (df_data[:group] .== 1) & (df_data[:buyer_pre_p1] .== 1) & (df_data[sranfx] .== v_level) , :trps_pre_p1] )
        #sdf[sdf[:key].==k ,:unadj_avg_expsd_hh_pst] = mean(df_data[ (df_data[:group] .== 1) & (df_data[:buyer_pos_p1] .== 1) & (df_data[sranfx] .== v_level) , :trps_pos_p1] )
        sdf[sdf[:key].==k ,:unadj_avg_expsd_hh_pre] = mean(df_data[ (df_data[:group] .== 1) & (df_data[sranfx] .== v_level) , :trps_pre_p1] )
        sdf[sdf[:key].==k ,:unadj_avg_expsd_hh_pst] = mean(df_data[ (df_data[:group] .== 1) & (df_data[sranfx] .== v_level) , :trps_pos_p1] )        
        
        if exposed
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mocc.feff.sdf[1,:unadj_avg_cntrl_hh_pre]
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mocc.feff.sdf[1,:unadj_avg_cntrl_hh_pst]
        else
            #sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mean(df_data[ (df_data[:group] .== 0) & (df_data[:buyer_pre_p1] .== 1) & (df_data[sranfx] .== v_level) , :trps_pre_p1] )
            #sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mean(df_data[ (df_data[:group] .== 0) & (df_data[:buyer_pos_p1] .== 1) & (df_data[sranfx] .== v_level) , :trps_pos_p1] )
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mean(df_data[ (df_data[:group] .== 0) & (df_data[sranfx] .== v_level) , :trps_pre_p1] )
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mean(df_data[ (df_data[:group] .== 0) & (df_data[sranfx] .== v_level) , :trps_pos_p1] )
        end       
    end 
end




function aggregate!(df_data::DataFrame, mdolocc::MDolOcc)
    sdf=mdolocc.reff.sdf
    for row = eachrow(mdolocc.reff.sdf)
        k = row[:key]
        ranfx = lowercase(row[:class])
        sranfx = symbol(ranfx)
        v_level = row[:level]
        exposed = row[:exposed]         
        dest_colname0=symbol(k*"_dolocc_0")
        dest_colname1=symbol(k*"_dolocc_1")
        xcol=symbol("reff_"*ranfx*"_dolocc")
        fmula0="df_data[:pre_dolocc_score0]+"*getRandomFormula(mdolocc.reff.rcols, xcol)*"+"*string(row[:B0])
        fmula0=replace(fmula0,"++","+")    #if only one col
        fmula1="df_data[:pre_dolocc_score1]+"*getRandomFormula(mdolocc.reff.rcols, xcol)*"+"*string(row[:B1])
        fmula1=replace(fmula1,"++","+") 
                
        df_data[dest_colname0] = exp(eval(parse(fmula0)))
        df_data[dest_colname1] = exp(eval(parse(fmula1)))
        println("\n\n\n",xcol," ~~ ",fmula1)
        println("dolocc ~ ",k," --> ",dest_colname0)         
   
        mean_score0 = mean(df_data[df_data[:buyer_pos_p1] .== 1,  dest_colname0 ])
        mean_score1 = mean(df_data[ df_data[:buyer_pos_p1] .== 1 , dest_colname1 ])
        sdf[sdf[:key].==k ,:mean_score0] = mean_score0
        sdf[sdf[:key].==k ,:mean_score1] = mean_score1   
        sdf[sdf[:key].==k ,:adj_mean_cntrl_grp] = mean_score0
        sdf[sdf[:key].==k ,:adj_mean_expsd_grp] = mean_score1
        
        #sdf[sdf[:key].==k ,:unadj_avg_expsd_hh_pre] = mean(df_data[ (df_data[:group] .== 1) & (df_data[:buyer_pre_p1] .== 1) & (df_data[sranfx] .== v_level) , :dol_per_trip_pre_p1] )
        #sdf[sdf[:key].==k ,:unadj_avg_expsd_hh_pst] = mean(df_data[ (df_data[:group] .== 1) & (df_data[:buyer_pos_p1] .== 1) & (df_data[sranfx] .== v_level) , :dol_per_trip_pos_p1] )
        sdf[sdf[:key].==k ,:unadj_avg_expsd_hh_pre] = mean(df_data[ (df_data[:group] .== 1) & (df_data[sranfx] .== v_level) , :dol_per_trip_pre_p1] )
        sdf[sdf[:key].==k ,:unadj_avg_expsd_hh_pst] = mean(df_data[ (df_data[:group] .== 1) & (df_data[sranfx] .== v_level) , :dol_per_trip_pos_p1] )
        
        if exposed
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mdolocc.feff.sdf[1,:unadj_avg_cntrl_hh_pre]
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mdolocc.feff.sdf[1,:unadj_avg_cntrl_hh_pst]
        else
            #sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mean(df_data[ (df_data[:group] .== 0) & (df_data[:buyer_pre_p1] .== 1) & (df_data[sranfx] .== v_level) , :dol_per_trip_pre_p1] ) 
            #sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mean(df_data[ (df_data[:group] .== 0) & (df_data[:buyer_pos_p1] .== 1) & (df_data[sranfx] .== v_level) , :dol_per_trip_pos_p1] )
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mean(df_data[ (df_data[:group] .== 0) & (df_data[sranfx] .== v_level) , :dol_per_trip_pre_p1] ) 
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mean(df_data[ (df_data[:group] .== 0) & (df_data[sranfx] .== v_level) , :dol_per_trip_pos_p1] )
        end    
    end   
end





function aggregate!(df_data::DataFrame,mpen::MPen)
    sdf=mpen.reff.sdf
    for row = eachrow(mpen.reff.sdf)
        k = row[:key]
        ranfx = lowercase(row[:class])
        sranfx = symbol(ranfx)
        v_level = row[:level]
        exposed = row[:exposed]         
        dest_colname0=symbol(k*"_pen_0")
        dest_colname1=symbol(k*"_pen_1")
        xcol=symbol("reff_"*ranfx*"_pen")
        fmula0="df_data[:pre_pen_score0]+"*getRandomFormula(mpen.reff.rcols, xcol)*"+"*string(row[:B0])
        fmula0=replace(fmula0,"++","+")    #if only one col
        fmula1="df_data[:pre_pen_score1]+"*getRandomFormula(mpen.reff.rcols, xcol)*"+"*string(row[:B1])
        fmula1=replace(fmula1,"++","+") 
        

        df_data[dest_colname0] = exp(eval(parse(fmula0))) ./ (1+exp(eval(parse(fmula0))))
        df_data[dest_colname1] = exp(eval(parse(fmula1))) ./ (1+exp(eval(parse(fmula1))))    
        println("\n\n\n",xcol," ~~ ",fmula1)
        println("pen ~ ",k," --> ",dest_colname1)    
        mean_score0 = mean(df_data[dest_colname0])
        mean_score1 = mean(df_data[dest_colname1])
        
        println("..... ",mean_score0," ~ ",mean_score1)
        
        sdf[sdf[:key].==k ,:mean_score0] = mean_score0
        sdf[sdf[:key].==k ,:mean_score1] = mean_score1 
        sdf[sdf[:key].==k ,:adj_mean_cntrl_grp] = mean_score0
        sdf[sdf[:key].==k ,:adj_mean_expsd_grp] = mean_score1
  
        sdf[sdf[:key].==k ,:unadj_avg_expsd_hh_pre] = mean(df_data[ (df_data[:group] .== 1) & (df_data[symbol(ranfx)] .== v_level) , :buyer_pre_p1] )
        sdf[sdf[:key].==k ,:unadj_avg_expsd_hh_pst] = mean(df_data[ (df_data[:group] .== 1) & (df_data[symbol(ranfx)] .== v_level) , :buyer_pos_p1] )
        if exposed
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mpen.feff.sdf[1,:unadj_avg_cntrl_hh_pre]
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mpen.feff.sdf[1,:unadj_avg_cntrl_hh_pst]    
        else
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mean(df_data[ (df_data[:group] .== 0) & (df_data[symbol(ranfx)] .== v_level) , :buyer_pre_p1] )
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mean(df_data[ (df_data[:group] .== 0) & (df_data[symbol(ranfx)] .== v_level) , :buyer_pos_p1] )
        end         
    end      
end
"""
