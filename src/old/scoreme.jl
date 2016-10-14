

## rdf.sdf[[:key,:model ,:adj_dod_effct,:onetail_80_pct_intrvl_lb,:onetail_80_pct_intrvl_ub]]
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ SCORING -------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------

using DataStructures, DataFrames, StatsFuns, GLM , JuMP,NLopt, StatStack
cd("/media/u01/analytics/scoring/Healthy/modeling5/Scoring/")
df_data = readtable("final_data.csv",header=true); lowercase(df_data)

mocc = MOcc(df_data)
mdolocc = MDolOcc(df_data)
mpen = MPen(df_data)
mdolhh=MDolHH(mocc,mdolocc,mpen)


df_data[symbol("pre_occ_score0")]=eval(parse(mocc.feff.fmula0))         
df_data[symbol("pre_occ_score1")]=eval(parse(mocc.feff.fmula1)) 
df_data[symbol("pre_dolocc_score0")]=eval(parse(mdolocc.feff.fmula0))         
df_data[symbol("pre_dolocc_score1")]=eval(parse(mdolocc.feff.fmula1)) 
df_data[symbol("pre_pen_score0")]=eval(parse(mpen.feff.fmula0))         
df_data[symbol("pre_pen_score1")]=eval(parse(mpen.feff.fmula1)) 


"""
function xgenRandomTotals(reff::RanEffect, df_data::DataFrame)
    model=reff.v_model
    idf=reff.src
    rcols=Symbol[]
    for ranfx in unique(idf[:class])    # -- Loop through each random effect :: create random df_data columns - and add list to reff
        df_data[symbol(ranfx)] = map(x->string(x), df_data[symbol(ranfx)])    # Convert all ranfx to string ... since they are categorical anyway
        dest_colname=symbol("reff_"*lowercase(ranfx)*"_"*model)
        push!(rcols,dest_colname)
        df_data[dest_colname]=0.0
        for row in eachrow(idf[idf[:class].==ranfx,:])   # Loop through each level for outter random effect
            level=string(row[:level])
            println("populating : ",dest_colname," : for :: (",level,")")
            df_data[ (df_data[symbol(ranfx)].==level) & (df_data[:group].==0),dest_colname]=row[:B0]
            df_data[ (df_data[symbol(ranfx)].==level) & (df_data[:group].==1),dest_colname]=row[:B1]            
        end
    end    
    reff.rcols=rcols
    reff.fmula=getRandomFormula(rcols)
end
xgenRandomTotals(mocc.reff,df_data)
xgenRandomTotals(mdolocc.reff,df_data)
xgenRandomTotals(mpen.reff,df_data)
"""


df_data[:reff_occ]=eval(parse(mocc.reff.fmula)) 
df_data[:reff_dolocc]=eval(parse(mdolocc.reff.fmula)) 
df_data[:reff_pen]=eval(parse(mpen.reff.fmula)) 

"""
df_data[:reff_total_occ_0]=eval(parse(mocc.reff.fmula0)) 
df_data[:reff_total_occ_1]=eval(parse(mocc.reff.fmula1))
df_data[:reff_total_dolocc_0]=eval(parse(mdolocc.reff.fmula0)) 
df_data[:reff_total_dolocc_1]=eval(parse(mdolocc.reff.fmula1))
df_data[:reff_total_pen_0]=eval(parse(mpen.reff.fmula0)) 
df_data[:reff_total_pen_1]=eval(parse(mpen.reff.fmula1))
"""

#NEW
#df_data[:reff_occ]=0.0
#df_data[df_data[:group].==1,:reff_occ] = df_data[df_data[:group].==1,:reff_total_occ_1]
#df_data[df_data[:group].==0,:reff_occ] = df_data[df_data[:group].==0,:reff_total_occ_0]
##df_data[[:reff_total_occ,:group,:reff_total_occ_1,:reff_total_occ_0]]
#df_data[:reff_dolocc]=0.0
#df_data[df_data[:group].==1,:reff_dolocc] = df_data[df_data[:group].==1,:reff_total_dolocc_1]
#df_data[df_data[:group].==0,:reff_dolocc] = df_data[df_data[:group].==0,:reff_total_dolocc_0]
##df_data[[:reff_total_dolocc,:group,:reff_total_dolocc_1,:reff_total_dolocc_0]]
#df_data[:reff_pen]=0.0
#df_data[df_data[:group].==1,:reff_pen] = df_data[df_data[:group].==1,:reff_total_pen_1]
#df_data[df_data[:group].==0,:reff_pen] = df_data[df_data[:group].==0,:reff_total_pen_0]
##df_data[[:reff_total_pen,:group,:reff_total_pen_1,:reff_total_pen_0]]

#  773893077228 - Ian

cnts = Cnts(df_data, mocc,mdolocc,mpen)
mdf=MDF(cnts,mocc,mdolocc,mpen)
rdf=RDF(mocc,mdolocc,mpen,mdolhh)



function init!(cnts::Cnts, mocc::MOcc,mdolocc::MDolOcc,mpen::MPen)
    for m in [mocc,mdolocc,mpen]
        #rsrc=m.reff.src
        #rsdf=m.reff.sdf

        # --- GROUP1 -----
#        rsrc[:B1_combo] = rsrc[:B1] + b_fixed                                     
#        rsrc[:P1_combo] = 2.0 * ccdf(Normal(), abs( (rsrc[:B1_combo]) ./ rsrc[:SE1]))      
        
#        rsdf[:B] = rsdf[:B] + b_fixed  # Set sdf[:B] to B1_combo
        #mocc.reff.src[[:class,:level,:B0,:B1,:SE0,:SE1,:exposed,:P1,:key]]
        
        m.reff.sdf = join(m.reff.sdf,m.reff.src[[:class,:level,:B0,:B1,:SE0,:SE1,:exposed,:P1,:key]], on = :key)
        sdf=m.reff.sdf
        sdf[:B_fixed] = m.feff.sdf[1,:B]
        sdf[:P_fixed] = m.feff.sdf[1,:P]
        sdf[:SE_fixed] = m.feff.sdf[1,:SE]
        #sdf[:B1_combo] = sdf[:B1] + sdf[:B_fixed] - sdf[:B0] 
        sdf[:B1_combo] = sdf[:B1] + sdf[:B_fixed] #- sdf[:B0]
        #sdf[:SE1_combo] = sqrt(sdf[:SE_fixed]^2+sdf[:SE0]^2+sdf[:SE1]^2)
        #sdf[:SE1_combo] = sqrt(sdf[:SE_fixed].^2+sdf[:SE0].^2+sdf[:SE1].^2)
        sdf[:SE1_combo] = sqrt(sdf[:SE_fixed].^2+sdf[:SE1].^2)
        sdf[:P1_combo] = 2.0 * ccdf(Normal(), abs( (sdf[:B1_combo]) ./ sdf[:SE1_combo])) 
        
        #rsrc[:B_combo] = rsrc[:B1] + b_fixed - rsrc[:B0] 
        #rsrc[:SE1_combo] = SEsq=sqrt(se_fixed^2+rsrc[:SE0]^2+rsrc[:SE1]^2)
        #rsrc[:P1_combo] = 2.0 * ccdf(Normal(), abs( (rsrc[:B1_combo]) ./ rsrc[:SE1]))      
        
        #rsdf[:B_combo]=0.0
        #rsdf[:SE1_combo]=0.0
        
        #rsdf[:BR] = (rsdf[:B] + b_fixed)  # Set sdf[:B] to B1_combo
        
        
        #-- SET non-sig values -- 
        #sdf[:sig1_out]=sdf[:SIG1]
        #sdf[:SE1_out] = sdf[:SE1]
        #sdf[sdf[:SIG1] .== false  ,:SE1_out] = se_fixed
        #sdf[:P1_out] = sdf[:P1_combo]
        #sdf[sdf[:SIG1] .== false  ,:P1_out] = p_fixed
        #sdf[:B1_out] = sdf[:B1_combo]                                     
        #sdf[sdf[:SIG1] .== false , :B1_out] = b_fixed
        
        #sdf[:isRevertSEPB1_2Total] = false
        #sdf[sdf[:SIG1] .== 0  ,:isRevertSEPB1_2Total] = true
        
        # --- GROUP0 ----- to redo - based on variance consideration
        #-- SET non-sig values -- 
        #sdf[:B0_out] = sdf[:B0]
        #sdf[sdf[:SIG0] .== false, :B0_out] = 0  
        
        #raneff.sdf[:key] = map((c,l) ->  rkey(c,l), raneff.sdf[:class],raneff.sdf[:level])
        #raneff.sdf=sdf
        #raneff.idf=idf
            
        #sdf=m.reff.sdf
        #for k in sdf[:key]     
        #    sdf[sdf[:key].==k ,:M] = rcnts.get(k)[:M][1]
        #    sdf[sdf[:key].==k ,:Mt] = rcnts.get(k)[:Mt][1]
        #    sdf[sdf[:key].==k ,:Mc] = rcnts.get(k)[:Mc][1]
        #end
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
end

init!(cnts, mocc,mdolocc,mpen)




 
function aggregate!(df_data::DataFrame, focc::FOcc)
    sdf = focc.sdf
    sdf[:M] = cnts.get("Total Campaign")[:M][1]
    sdf[:Mt] = cnts.get("Total Campaign")[:Mt][1]
    sdf[:Mc] = cnts.get("Total Campaign")[:Mc][1]
    
    """
    if :reff_total_occ_0 in names(df_data)
        df_data[:occ_score0] = exp(df_data[:pre_occ_score0] + df_data[:reff_total_occ_0]  ) 
    else
        df_data[:occ_score0] = exp(df_data[:pre_occ_score0]) 
    end
    if :reff_total_occ_1 in names(df_data)
        df_data[:occ_score1] = exp(df_data[:pre_occ_score1] + df_data[:reff_total_occ_1] )    
    else
        df_data[:occ_score1] = exp(df_data[:pre_occ_score1])    
    end
    """
    if :reff_occ in names(df_data)
        println("using reff")
        df_data[:occ_score0] = exp(df_data[:pre_occ_score0] + df_data[:reff_occ]  ) 
        df_data[:occ_score1] = exp(df_data[:pre_occ_score1] + df_data[:reff_occ] ) 
    else
        df_data[:occ_score0] = exp(df_data[:pre_occ_score0]) 
        df_data[:occ_score1] = exp(df_data[:pre_occ_score1])
    end
    
        

    sdf[:mean_score0]=mean( df_data[ df_data[:buyer_pos_p1] .== 1 , :occ_score0] )
    sdf[:mean_score1]=mean( df_data[ df_data[:buyer_pos_p1] .== 1 , :occ_score1] ) 
    
    sdf[:unadj_mean_score0]=mean(df_data[(df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 0),:occ_score0])
    sdf[:unadj_mean_score1]=mean(df_data[(df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 1),:occ_score1])

    sdf[:adj_mean_cntrl_grp ] = mean( df_data[ df_data[:buyer_pos_p1] .== 1 , :occ_score0] )                     
    sdf[:adj_mean_expsd_grp ] = mean( df_data[ df_data[:buyer_pos_p1] .== 1 , :occ_score1] )
    # -------  Occ Score
#    sdf[:unadj_avg_cntrl_hh_pre] = mean(df_data[ (df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 0), :trps_pre_p1] )  
#    sdf[:unadj_avg_expsd_hh_pre] = mean(df_data[ (df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 1), :trps_pre_p1] )
    sdf[:unadj_avg_cntrl_hh_pre] = mean(df_data[ (df_data[:buyer_pre_p1] .== 1) & (df_data[:group] .== 0), :trps_pre_p1] )  
    sdf[:unadj_avg_expsd_hh_pre] = mean(df_data[ (df_data[:buyer_pre_p1] .== 1) & (df_data[:group] .== 1), :trps_pre_p1] )
    
    sdf[:unadj_avg_cntrl_hh_pst] = mean(df_data[ (df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 0), :trps_pos_p1] )
    sdf[:unadj_avg_expsd_hh_pst] = mean(df_data[ (df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 1), :trps_pos_p1] )
    


end
aggregate!(df_data, mocc.feff)



function aggregate!(df_data::DataFrame, fdolocc::FDolOcc)
    sdf = fdolocc.sdf
    sdf[:M] = cnts.get("Total Campaign")[:M][1]
    sdf[:Mt] = cnts.get("Total Campaign")[:Mt][1]
    sdf[:Mc] = cnts.get("Total Campaign")[:Mc][1]
  
    """
    if :reff_total_dolocc_0 in names(df_data)
        df_data[:dolocc_score0] = exp(df_data[:pre_dolocc_score0] + df_data[:reff_total_dolocc_0]  ) 
    else
        df_data[:dolocc_score0] = exp(df_data[:pre_dolocc_score0])
    end
    if :reff_total_dolocc_1 in names(df_data)
        df_data[:dolocc_score1] = exp(df_data[:pre_dolocc_score1] + df_data[:reff_total_dolocc_1] )    
    else
        df_data[:dolocc_score1] = exp(df_data[:pre_dolocc_score1])   
    end    
    """
    if :reff_dolocc in names(df_data)
        println("using reff")
        df_data[:dolocc_score0] = exp(df_data[:pre_dolocc_score0] + df_data[:reff_dolocc]  ) 
        df_data[:dolocc_score1] = exp(df_data[:pre_dolocc_score1] + df_data[:reff_dolocc] )
    else
        df_data[:dolocc_score0] = exp(df_data[:pre_dolocc_score0])
        df_data[:dolocc_score1] = exp(df_data[:pre_dolocc_score1]) 
    end
    
    
    
    sdf[:mean_score0]=mean( df_data[ df_data[:buyer_pos_p1] .== 1 , :dolocc_score0] )
    sdf[:mean_score1]=mean( df_data[ df_data[:buyer_pos_p1] .== 1 ,  :dolocc_score1] )
    
    sdf[:unadj_mean_score0]=mean(df_data[(df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 0),:dolocc_score0])
    sdf[:unadj_mean_score1]=mean(df_data[(df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 1),:dolocc_score1])
    
    sdf[:adj_mean_cntrl_grp ] =  mean( df_data[ df_data[:buyer_pos_p1] .== 1 , :dolocc_score0] )             
    sdf[:adj_mean_expsd_grp ] =  mean( df_data[ df_data[:buyer_pos_p1] .== 1 ,  :dolocc_score1] )            
    # -------  DolOcc Score
    #sdf[:unadj_avg_cntrl_hh_pre] =  mean(df_data[ (df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 0), :dol_per_trip_pre_p1] )
    #sdf[:unadj_avg_expsd_hh_pre] =  mean(df_data[ (df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 1), :dol_per_trip_pre_p1] )
    sdf[:unadj_avg_cntrl_hh_pre] =  mean(df_data[ (df_data[:buyer_pre_p1] .== 1) & (df_data[:group] .== 0), :dol_per_trip_pre_p1] )
    sdf[:unadj_avg_expsd_hh_pre] =  mean(df_data[ (df_data[:buyer_pre_p1] .== 1) & (df_data[:group] .== 1), :dol_per_trip_pre_p1] )
    sdf[:unadj_avg_cntrl_hh_pst] =  mean(df_data[ (df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 0), :dol_per_trip_pos_p1] )
    sdf[:unadj_avg_expsd_hh_pst] =  mean(df_data[ (df_data[:buyer_pos_p1] .== 1) & (df_data[:group] .== 1), :dol_per_trip_pos_p1] )
end
aggregate!(df_data, mdolocc.feff)


function aggregate!(df_data::DataFrame, fpen::FPen)
    sdf = fpen.sdf
    sdf[:M] = cnts.get("Total Campaign")[:N][1]
    sdf[:Mt] = cnts.get("Total Campaign")[:Nt][1]
    sdf[:Mc] = cnts.get("Total Campaign")[:Nc][1]
    """
    if :reff_total_pen_0 in names(df_data)
        df_data[:pen_score0] = exp((df_data[:pre_pen_score0] + df_data[:reff_total_pen_0])) ./ ( exp((df_data[:pre_pen_score0] + df_data[:reff_total_pen_0])) +1)
    else
        df_data[:pen_score0] =  exp(df_data[:pre_pen_score0]) ./ ( exp(df_data[:pre_pen_score0]) +1)
    end
    if :reff_total_pen_1 in names(df_data)
        df_data[:pen_score1] = exp((df_data[:pre_pen_score1] + df_data[:reff_total_pen_1])) ./ (exp((df_data[:pre_pen_score1] + df_data[:reff_total_pen_1])) +1)     
    else
        df_data[:pen_score1] = exp(df_data[:pre_pen_score1]) ./ ( exp(df_data[:pre_pen_score1]) +1)  
    end
    #df_data[:pen_score0] =  exp(df_data[:pre_pen_score0]) ./ ( exp(df_data[:pre_pen_score0]) +1 ) 
    #df_data[:pen_score1] = exp(df_data[:pre_pen_score1]) ./ ( exp(df_data[:pre_pen_score1]) +1 )
    """
    if :reff_pen in names(df_data)
        println("using reff")
        df_data[:pen_score0] = exp((df_data[:pre_pen_score0] + df_data[:reff_pen])) ./ ( exp((df_data[:pre_pen_score0] + df_data[:reff_pen])) +1)
        df_data[:pen_score1] = exp((df_data[:pre_pen_score1] + df_data[:reff_pen])) ./ (exp((df_data[:pre_pen_score1] + df_data[:reff_pen])) +1) 
    else
        df_data[:pen_score0] =  exp(df_data[:pre_pen_score0]) ./ ( exp(df_data[:pre_pen_score0]) +1)
        df_data[:pen_score1] = exp(df_data[:pre_pen_score1]) ./ ( exp(df_data[:pre_pen_score1]) +1)
    end

    
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
aggregate!(df_data, mpen.feff)



"""
function aggregate!(df_data::DataFrame, mdolhh::MDolHH )
    o_mean_score0 = mdolhh.mocc.feff.sdf[1,:adj_mean_cntrl_grp]
    o_mean_score1 = mdolhh.mocc.feff.sdf[1,:adj_mean_expsd_grp]
    y_mean_score0 = mdolhh.mdolocc.feff.sdf[1,:adj_mean_cntrl_grp]
    y_mean_score1 = mdolhh.mdolocc.feff.sdf[1,:adj_mean_expsd_grp]
    p_mean_score0 = mdolhh.mpen.feff.sdf[1,:adj_mean_cntrl_grp]
    p_mean_score1 = mdolhh.mpen.feff.sdf[1,:adj_mean_expsd_grp]
    sdf = mdolhh.sdf
    k="Total Campaign"

    adjctl= o_mean_score0 * y_mean_score0 * p_mean_score0
    adjexp= o_mean_score1 * y_mean_score1 * p_mean_score1    
    sdf[sdf[:key].==k,:adj_mean_cntrl_grp] = adjctl
    sdf[sdf[:key].==k,:adj_mean_expsd_grp] = adjexp
    # -------  DolHH Score   -- prd_1_net_pr_pre
    sdf[sdf[:key].==k,:unadj_avg_cntrl_hh_pre] =  mean(df_data[ (df_data[:group] .== 0)  , :prd_1_net_pr_pre] )
    sdf[sdf[:key].==k,:unadj_avg_expsd_hh_pre] =  mean(df_data[ (df_data[:group] .== 1)  , :prd_1_net_pr_pre] )
    sdf[sdf[:key].==k,:unadj_avg_cntrl_hh_pst] =  mean(df_data[ (df_data[:group] .== 0)  , :prd_1_net_pr_pos] )
    sdf[sdf[:key].==k,:unadj_avg_expsd_hh_pst] =  mean(df_data[ (df_data[:group] .== 1)  , :prd_1_net_pr_pos] )
    sdf[sdf[:key].==k,:adj_dod_effct] = ((adjexp - adjctl)  ./ adjctl ) * 100   
end
#aggregate!(df_data, mdolhh)
"""


function aggregateCommon!(fixedeff::FixedEffect)
    sdf = fixedeff.sdf
    src = fixedeff.src
    sdf[:adj_dod_effct] = ((sdf[:adj_mean_expsd_grp] - sdf[:adj_mean_cntrl_grp])  ./ sdf[:adj_mean_cntrl_grp] ) * 100   
    #sdf[:adj_dod_effct] = ((sdf[:unadj_mean_score1] - sdf[:unadj_mean_score0])  ./ sdf[:unadj_mean_score0] ) * 100
    #twotail_pval
    sdf[:twotail_pval] = sdf[:P]       
    #onetail_pval
    sdf[:onetail_pval] = 1 - (sdf[:twotail_pval] ./ 2)
    #redo :twotail_pval
    sdf[:twotail_pval] = 1-sdf[:P]
    #For NON-SIG Models, update - adj_mean_expsd_grp (to = adj_mean_cntrl_grp) & adj_dod_effct (= 0)
    sdf[(sdf[:twotail_pval] .< 0.80) & (sdf[:twotail_pval] .!= 0),:adj_mean_expsd_grp]      =  sdf[(sdf[:twotail_pval] .< 0.80) & (sdf[:twotail_pval] .!= 0),:adj_mean_cntrl_grp]
    #sdf[(sdf[:twotail_pval] .< 0.80) & (sdf[:twotail_pval] .!= 0),:adj_mean_expsd_grp]      =  sdf[(sdf[:twotail_pval] .< 0.80) & (sdf[:twotail_pval] .!= 0),:adj_mean_cntrl_grp]
end
aggregateCommon!(mocc.feff)
aggregateCommon!(mdolocc.feff)
aggregateCommon!(mpen.feff) 


# -------- Confidence Intervals - Total --------


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
                    sdf[sdf[:key].==k, lbk] = md[lbk]
                    sdf[sdf[:key].==k, ubk] = md[ubk]
            end
    end
end
ConfidenceIntervals(mocc.feff)
ConfidenceIntervals(mdolocc.feff)
ConfidenceIntervals(mpen.feff)



# -------- DolHH Confidence Intervals - Total --------
"""
function ConfidenceIntervals(mdolhh::MDolHH )
    osdf=mdolhh.mocc.feff.sdf
    dsdf=mdolhh.mdolocc.feff.sdf
    psdf=mdolhh.mpen.feff.sdf
    sdf = mdolhh.sdf
    md=OrderedDict()
    md[:M] = osdf[1,:M]
    md[:Mt] = osdf[1,:Mt]
    md[:Mc] = osdf[1,:Mc]
    md[:N] = psdf[1,:M]
    md[:Nt] = psdf[1,:Mt]
    md[:Nc] = psdf[1,:Mc]
    
    #md[:M] = cnts.get("Total Campaign")[:M][1]
    #md[:Mt] = cnts.get("Total Campaign")[:Mt][1]
    #md[:Mc] = cnts.get("Total Campaign")[:Mc][1]
    #md[:N] = cnts.get("Total Campaign")[:N][1]
    #md[:Nt] = cnts.get("Total Campaign")[:Nt][1]
    #md[:Nc] = cnts.get("Total Campaign")[:Nc][1]
    
    md[:B1] = osdf[1,:B]
    md[:B2] = dsdf[1,:B]
    md[:B3] = psdf[1,:B]
    md[:SE1] = osdf[1,:SE]
    md[:SE2] = dsdf[1,:SE]
    md[:SE3] = psdf[1,:SE]

    md[:o_mean_score0] = osdf[1,:unadj_mean_score0]
    md[:o_mean_score1] = osdf[1,:unadj_mean_score1]
    md[:y_mean_score0] = dsdf[1,:unadj_mean_score0]
    md[:y_mean_score1] = dsdf[1,:unadj_mean_score1]
    md[:p_mean_score0] = psdf[1,:unadj_mean_score0]
    md[:p_mean_score1] = psdf[1,:unadj_mean_score1]
    
    md[:metakey] = "total"
    calcPValue_Opt(md)
    calcCI_Opt(md)
    #println("\n",md)
    for k in [ :onetail_pval, :twotail_pval, :onetail_80_pct_intrvl_lb,:onetail_80_pct_intrvl_ub, :onetail_90_pct_intrvl_lb, :onetail_90_pct_intrvl_ub
              ,:twotail_80_pct_intrvl_lb, :twotail_80_pct_intrvl_ub, :twotail_90_pct_intrvl_lb, :twotail_90_pct_intrvl_ub    
             ]
        sdf[sdf[:key].=="Total Campaign", k] = md[k]
    end  
end
#ConfidenceIntervals(mdolhh)
"""




# --------------------------------------------------------
# ---------- BREAKS --------------------------------------
# --------------------------------------------------------



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
        #xcol0=symbol("reff_"*ranfx*"_occ_0")   # Column to exclude
        #xcol1=symbol("reff_"*ranfx*"_occ_1")
        #fmula0="df_data[:pre_occ_score0]+"*getRandomFormula(mocc.reff.rcols0, xcol0)*"+"*string(row[:B0])
        #fmula0=replace(fmula0,"++","+")    #if only one col
        #fmula1="df_data[:pre_occ_score1]+"*getRandomFormula(mocc.reff.rcols1, xcol1)*"+"*string(row[:B1])
        #fmula1=replace(fmula1,"++","+")
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
        
        sdf[sdf[:key].==k ,:unadj_avg_expsd_hh_pre] = mean(df_data[ (df_data[:group] .== 1) & (df_data[:buyer_pos_p1] .== 1) & (df_data[sranfx] .== v_level) , :trps_pre_p1] )
        sdf[sdf[:key].==k ,:unadj_avg_expsd_hh_pst] = mean(df_data[ (df_data[:group] .== 1) & (df_data[:buyer_pos_p1] .== 1) & (df_data[sranfx] .== v_level) , :trps_pos_p1] )
        if exposed
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mocc.feff.sdf[1,:unadj_avg_cntrl_hh_pre]
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mocc.feff.sdf[1,:unadj_avg_cntrl_hh_pst]
        else
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mean(df_data[ (df_data[:group] .== 0) & (df_data[:buyer_pos_p1] .== 1) & (df_data[sranfx] .== v_level) , :trps_pre_p1] )
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mean(df_data[ (df_data[:group] .== 0) & (df_data[:buyer_pos_p1] .== 1) & (df_data[sranfx] .== v_level) , :trps_pos_p1] )
        end
        
        #sdf[sdf[:key].==k ,:adj_dod_effct] = ((mean_score1 - mean_score0) / mean_score0) * 100
        sdf[sdf[:key].==k ,:adj_dod_effct] = ((    sdf[sdf[:key].==k ,:adj_mean_expsd_grp]    - sdf[sdf[:key].==k ,:adj_mean_cntrl_grp] )  ./ sdf[sdf[:key].==k ,:adj_mean_cntrl_grp] ) * 100   
        
        P1 = sdf[sdf[:key].==k ,:P1_combo][1]
        sdf[sdf[:key].==k ,:twotail_pval] = (1-P1)*100         #;if twotail_pval > 99 twotail_pval = 99 end
        sdf[sdf[:key].==k ,:onetail_pval] = (1-(P1/2))*100     #;if onetail_pval > 99 onetail_pval = 99 end        
         
    end 
end
aggregate!(df_data, mocc)
#mocc.reff.sdf[[:adj_dod_effct]]





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
        #xcol0=symbol("reff_"*ranfx*"_dolocc_0")
        #xcol1=symbol("reff_"*ranfx*"_dolocc_1")
        #fmula0="df_data[:pre_dolocc_score0]+"*getRandomFormula(mdolocc.reff.rcols0, xcol0)*"+"*string(row[:B0])
        #fmula0=replace(fmula0,"++","+")    #if only one col
        #fmula1="df_data[:pre_dolocc_score1]+"*getRandomFormula(mdolocc.reff.rcols1, xcol1)*"+"*string(row[:B1])
        #fmula1=replace(fmula1,"++","+")  
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
        
        sdf[sdf[:key].==k ,:unadj_avg_expsd_hh_pre] = mean(df_data[ (df_data[:group] .== 1) & (df_data[:buyer_pos_p1] .== 1) & (df_data[sranfx] .== v_level) , :dol_per_trip_pre_p1] )
        sdf[sdf[:key].==k ,:unadj_avg_expsd_hh_pst] = mean(df_data[ (df_data[:group] .== 1) & (df_data[:buyer_pos_p1] .== 1) & (df_data[sranfx] .== v_level) , :dol_per_trip_pos_p1] )
        if exposed
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mdolocc.feff.sdf[1,:unadj_avg_cntrl_hh_pre]
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mdolocc.feff.sdf[1,:unadj_avg_cntrl_hh_pst]
        else
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mean(df_data[ (df_data[:group] .== 0) & (df_data[:buyer_pos_p1] .== 1) & (df_data[sranfx] .== v_level) , :dol_per_trip_pre_p1] ) 
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mean(df_data[ (df_data[:group] .== 0) & (df_data[:buyer_pos_p1] .== 1) & (df_data[sranfx] .== v_level) , :dol_per_trip_pos_p1] )
        end    
        #sdf[sdf[:key].==k ,:adj_dod_effct] = ((mean_score1 - mean_score0) / mean_score0) * 100
        sdf[sdf[:key].==k ,:adj_dod_effct] = ((sdf[sdf[:key].==k ,:adj_mean_expsd_grp]    - sdf[sdf[:key].==k ,:adj_mean_cntrl_grp] )  ./ sdf[sdf[:key].==k ,:adj_mean_cntrl_grp] ) * 100  
        P1 = sdf[sdf[:key].==k ,:P1_combo][1]
        sdf[sdf[:key].==k ,:twotail_pval] = (1-P1)*100         #;if twotail_pval > 99 twotail_pval = 99 end
        sdf[sdf[:key].==k ,:onetail_pval] = (1-(P1/2))*100     #;if onetail_pval > 99 onetail_pval = 99 end      

    end   
end
aggregate!(df_data, mdolocc)






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
        #xcol0=symbol("reff_"*ranfx*"_pen_0")
        #xcol1=symbol("reff_"*ranfx*"_pen_1")
        #fmula0="df_data[:pre_pen_score0]+"*getRandomFormula(mpen.reff.rcols0, xcol0)*"+"*string(row[:B0])
        #fmula0=replace(fmula0,"++","+")    #if only one col
        #fmula1="df_data[:pre_pen_score1]+"*getRandomFormula(mpen.reff.rcols1, xcol1)*"+"*string(row[:B1])
        #fmula1=replace(fmula1,"++","+") 
        fmula0="df_data[:pre_pen_score0]+"*getRandomFormula(mpen.reff.rcols, xcol)*"+"*string(row[:B0])
        fmula0=replace(fmula0,"++","+")    #if only one col
        fmula1="df_data[:pre_pen_score1]+"*getRandomFormula(mpen.reff.rcols, xcol)*"+"*string(row[:B1])
        fmula1=replace(fmula1,"++","+") 
        
        #df_data[dest_colname0] = exp(eval(parse(fmula0)))
        #df_data[dest_colname1] = exp(eval(parse(fmula1)))
        df_data[dest_colname0] = exp(eval(parse(fmula0))) ./ (1+exp(eval(parse(fmula0))))
        df_data[dest_colname1] = exp(eval(parse(fmula1))) ./ (1+exp(eval(parse(fmula1))))
        #df_data[symbol(dest_colname*"_0")] = exp(df_data[:pre_pen_score0]+row[:B0_out]) ./ (1+exp(df_data[:pre_pen_score0]+row[:B0_out]))    ##ASK SAMPATH
        #df_data[symbol(dest_colname*"_1")] = exp(df_data[:pre_pen_score1]+row[:B1]) ./ (1+exp(df_data[:pre_pen_score1]+row[:B1]))  
        #df_data[symbol(dest_colname0)] = exp(df_data[:pre_pen_score0]+row[:B0]) ./ (1+exp(df_data[:pre_pen_score0]+row[:B0]))    ##ASK SAMPATH
        #df_data[symbol(dest_colname0)] = exp(df_data[:pre_pen_score1]+row[:B1]) ./ (1+exp(df_data[:pre_pen_score1]+row[:B1])) 
        
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
            
#            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = rdf[(rdf[:model_desc].=="Total Campaign") & (rdf[:restype] .== "pen"), :unadj_avg_cntrl_hh_pre][1]
#            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = rdf[(rdf[:model_desc].=="Total Campaign") & (rdf[:restype] .== "pen"), :unadj_avg_cntrl_hh_pst][1]
        else
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pre] = mean(df_data[ (df_data[:group] .== 0) & (df_data[symbol(ranfx)] .== v_level) , :buyer_pre_p1] )
            sdf[sdf[:key].==k ,:unadj_avg_cntrl_hh_pst] = mean(df_data[ (df_data[:group] .== 0) & (df_data[symbol(ranfx)] .== v_level) , :buyer_pos_p1] )
        end 
        #sdf[sdf[:key].==k ,:adj_dod_effct] = ((mean_score1 - mean_score0) / mean_score0) * 100
        sdf[sdf[:key].==k,:adj_dod_effct] = ((sdf[sdf[:key].==k,:adj_mean_expsd_grp] - sdf[sdf[:key].==k,:adj_mean_cntrl_grp] )  ./ sdf[sdf[:key].==k,:adj_mean_cntrl_grp] ) * 100  
        P1 = sdf[sdf[:key].==k ,:P1_combo][1]
        sdf[sdf[:key].==k ,:twotail_pval] = (1-P1)*100         #;if twotail_pval > 99 twotail_pval = 99 end
        sdf[sdf[:key].==k ,:onetail_pval] = (1-(P1/2))*100     #;if onetail_pval > 99 onetail_pval = 99 end      
        
    end      
end
aggregate!(df_data, mpen)
     


"""
function aggregate!(mdolhh::MDolHH,mdf::MDF)
    sdf=mdolhh.sdf
    xsdf = join(mdolhh.sdf, cnts.sdf[[:key,:class,:level,:exposed]], on = :key)
    for row = eachrow(xsdf)
        k = row[:key]
        ranfx = row[:class]
        v_level = row[:level]
        exposed = row[:exposed]
        if k != "Total Campaign"
            println("getting : ",k)
            md = df2dict(mdf.get(k))
            adj_mean_cntrl_grp = md[:o_mean_score0] * md[:y_mean_score0] * md[:p_mean_score0]
            adj_mean_expsd_grp = md[:o_mean_score1] * md[:y_mean_score1] * md[:p_mean_score1]
            println(k," ~~ ",adj_mean_cntrl_grp,"~",adj_mean_expsd_grp) #,md)            
            sdf[sdf[:key].== k,:adj_mean_cntrl_grp] = adj_mean_cntrl_grp
            sdf[sdf[:key].== k,:adj_mean_expsd_grp] = adj_mean_expsd_grp
            sdf[sdf[:key].== k,:unadj_avg_expsd_hh_pre] = mean(df_data[ (df_data[:group] .== 1) & (df_data[symbol(lowercase(ranfx))] .== v_level) , :prd_1_net_pr_pre] )
            sdf[sdf[:key].== k,:unadj_avg_expsd_hh_pst] = mean(df_data[ (df_data[:group] .== 1) & (df_data[symbol(lowercase(ranfx))] .== v_level) , :prd_1_net_pr_pos] )
            sdf[sdf[:key].== k,:adj_dod_effct] = ((adj_mean_expsd_grp - adj_mean_cntrl_grp) / adj_mean_cntrl_grp) * 100
            if exposed
                sdf[sdf[:key].== k,:unadj_avg_cntrl_hh_pre] = sdf[1,:unadj_avg_cntrl_hh_pre] 
                sdf[sdf[:key].== k,:unadj_avg_cntrl_hh_pst] = sdf[1,:unadj_avg_cntrl_hh_pst]     
            else
                sdf[sdf[:key].== k,:unadj_avg_cntrl_hh_pre] = mean(df_data[ (df_data[:group] .== 0) & (df_data[symbol(lowercase(ranfx))] .== v_level) , :prd_1_net_pr_pre] )
                sdf[sdf[:key].== k,:unadj_avg_cntrl_hh_pst] = mean(df_data[ (df_data[:group] .== 0) & (df_data[symbol(lowercase(ranfx))] .== v_level) , :prd_1_net_pr_pos] ) 
            end            
        end
    end
end
#aggregate!(mdolhh,mdf)
"""






# -------- Confidence Intervals - Random Effects --------


#cio = CIs(   df2dict(mocc.reff.sdf[mocc.reff.sdf[:key].=="estimated_hh_income (A)",[:B1_combo,:SE1_combo,:mean_score0,:mean_score1,:M,:Mt,:Mc]])    , ZDict)
function ConfidenceIntervals(cnts::Cnts,reff::RanEffect)
    sdf = reff.sdf
    for row in eachrow(sdf)
        k = row[:key]
        md = df2dict(row)    #[:B1_combo,:SE1_combo,:mean_score0,:mean_score1,:M,:Mt,:Mc]
        md[:B] =  md[:B1_combo] # don't really need these...need to clean up
        md[:SE] =  md[:SE1_combo] # ditto
            
        md[:M_F] = cnts.sdf[1,:M]
        md[:Mt_F] = cnts.sdf[1,:Mt]
        md[:Mc_F] = cnts.sdf[1,:Mc]
        if row[:exposed]
            cis =  CIs(md, ZDict, 1)
        else
            cis =  CIs(md, ZDict, 2)
        end
        #cis =  CIs(md, ZDict, 1)
        for (zscore_key,zscore) in  ZDict     
                ubk = symbol(zscore_key*"_ub")
                lbk=symbol(zscore_key*"_lb")
                sdf[sdf[:key].==k, lbk] = md[lbk]
                sdf[sdf[:key].==k, ubk] = md[ubk]
        end
    end
end
#function ConfidenceIntervals(cnts::Cnts,rocc::ROcc, rdolocc::RDolOcc, rpen::RPen)
#    for reff in [rocc, rdolocc, rpen]
#        ConfidenceIntervals(cnts,reff)
#    end
#end
ConfidenceIntervals(cnts, mocc.reff)
ConfidenceIntervals(cnts, mdolocc.reff)
ConfidenceIntervals(cnts, mpen.reff)


#function ConfidenceIntervals(reff::RanEffect)
#    sdf = reff.sdf
#    for row in eachrow(sdf)
#            k = row[:key]
#            md = df2dict(row)
#            md[:B] =  md[:B1_combo]
#            md[:SE] =  md[:SE1_combo]
#            for (zscore_key,zscore) in  ZDict     
#                #if md[:inOcc] & md[:isSig1] & (md[:M] > 0)           
#                    lb, ub = calcCI(md, zscore)
#                    #println(md,"\nCIs : ",lb," ~ ",ub)
#                    sdf[sdf[:key].==k, symbol(zscore_key*"_lb")] = lb
#                    sdf[sdf[:key].==k, symbol(zscore_key*"_ub")] = ub
#                #end
#            end
#        end
#end
#
#function ConfidenceIntervals(rocc::ROcc, rdolocc::RDolOcc, rpen::RPen)
#    for reff in [rocc, rdolocc, rpen]
#        ConfidenceIntervals(reff)
#    end
#end
#ConfidenceIntervals(mocc.reff,mdolocc.reff,mpen.reff)
##sdf[sdf[:P].> 0.2, [:key,:adj_dod_effct,:onetail_80_pct_intrvl_lb,:onetail_80_pct_intrvl_ub,:twotail_pval,:onetail_pval,:P]]


function aggregate!(df_data::DataFrame, mdolhh::MDolHH,mdf::MDF)
    sdf=mdolhh.sdf
    xsdf = join(mdolhh.sdf, cnts.sdf[[:key,:class,:level,:exposed]], on = :key)
    for row = eachrow(xsdf)
        k = row[:key]
        ranfx = row[:class]
        v_level = row[:level]
        exposed = row[:exposed]
        println("getting : ",k)
        md = df2dict(mdf.get(k))
        adjctl = md[:o_mean_score0] * md[:y_mean_score0] * md[:p_mean_score0]
        adjexp = md[:o_mean_score1] * md[:y_mean_score1] * md[:p_mean_score1]
        println(k," ~~ ",adjctl,"~",adjexp) #,md)            
        sdf[sdf[:key].==k,:adj_mean_cntrl_grp] = adjctl
        sdf[sdf[:key].==k,:adj_mean_expsd_grp] = adjexp       
        sdf[sdf[:key].==k,:adj_dod_effct] = ((adjexp - adjctl) ./ adjctl ) * 100
        if k == "Total Campaign"
            sdf[sdf[:key].==k,:unadj_avg_cntrl_hh_pre] =  mean(df_data[ (df_data[:group] .== 0)  , :prd_1_net_pr_pre] )
            sdf[sdf[:key].==k,:unadj_avg_expsd_hh_pre] =  mean(df_data[ (df_data[:group] .== 1)  , :prd_1_net_pr_pre] )
            sdf[sdf[:key].==k,:unadj_avg_cntrl_hh_pst] =  mean(df_data[ (df_data[:group] .== 0)  , :prd_1_net_pr_pos] )
            sdf[sdf[:key].==k,:unadj_avg_expsd_hh_pst] =  mean(df_data[ (df_data[:group] .== 1)  , :prd_1_net_pr_pos] )
        else
            sdf[sdf[:key].== k,:unadj_avg_expsd_hh_pre] = mean(df_data[ (df_data[:group] .== 1) & (df_data[symbol(lowercase(ranfx))] .== v_level) , :prd_1_net_pr_pre] )
            sdf[sdf[:key].== k,:unadj_avg_expsd_hh_pst] = mean(df_data[ (df_data[:group] .== 1) & (df_data[symbol(lowercase(ranfx))] .== v_level) , :prd_1_net_pr_pos] )
            if exposed
                sdf[sdf[:key].== k,:unadj_avg_cntrl_hh_pre] = sdf[1,:unadj_avg_cntrl_hh_pre] 
                sdf[sdf[:key].== k,:unadj_avg_cntrl_hh_pst] = sdf[1,:unadj_avg_cntrl_hh_pst]     
            else
                sdf[sdf[:key].== k,:unadj_avg_cntrl_hh_pre] = mean(df_data[ (df_data[:group] .== 0) & (df_data[symbol(lowercase(ranfx))] .== v_level) , :prd_1_net_pr_pre] )
                sdf[sdf[:key].== k,:unadj_avg_cntrl_hh_pst] = mean(df_data[ (df_data[:group] .== 0) & (df_data[symbol(lowercase(ranfx))] .== v_level) , :prd_1_net_pr_pos] ) 
            end            
        end
    end
end
aggregate!(df_data,mdolhh,mdf)


# -------- DolHH Confidence Intervals - Total --------
"""
function ConfidenceIntervals_REff(mdolhh::MDolHH,mdf::MDF)
    sdf=mdolhh.sdf
    genMDF(cnts,mocc,mdolocc,mpen)
    #genMDF(mdf.cnts,mdf.mocc,mdf.mdolocc, mdf.mpen, false)
    #xsdf = join(mdolhh.sdf, cnts.sdf[[:key,:class,:level,:exposed]], on = :key)
    for row = eachrow(sdf)
        k = row[:key]
        #ranfx = row[:class]
        #v_level = row[:level]
        #exposed = row[:exposed]
        #println("ranfx:",ranfx," ~ ",v_level)
        if k != "Total Campaign"
            println("getting : ",k)
            md = df2dict(mdf.get(k))
            md[:metakey] = k
            #if (md[:isSig1] | md[:isSig2] | md[:isSig3] ) & ( md[:Mt] .>= 1) & ( md[:Nt] .>= 1 )    ###########NOTE:!!!! the 2 last conditions are because these are 0 for none, and causing
            calcPValue_Opt(md)
            calcCI_Opt(md) 
            for zk in [ :onetail_pval, :twotail_pval, :onetail_80_pct_intrvl_lb,:onetail_80_pct_intrvl_ub, :onetail_90_pct_intrvl_lb, :onetail_90_pct_intrvl_ub
                       ,:twotail_80_pct_intrvl_lb, :twotail_80_pct_intrvl_ub, :twotail_90_pct_intrvl_lb, :twotail_90_pct_intrvl_ub    
                      ]
                sdf[ sdf[:key].==k, zk]= md[zk]
            end    
            #end
        end
    end
end
#ConfidenceIntervals_REff(mdolhh,mdf)
"""

# -------- DolHH Confidence Intervals - Total --------
function ConfidenceIntervals(mdolhh::MDolHH,mdf::MDF)
    sdf=mdolhh.sdf
    genMDF(cnts,mocc,mdolocc,mpen)
    for row = eachrow(sdf)
        k = row[:key]
        println("getting : ",k)
        md = df2dict(mdf.get(k))
        md[:metakey] = k
        calcPValue_Opt(md)
        calcCI_Opt(md) 
        for zk in [ :onetail_pval, :twotail_pval, :onetail_80_pct_intrvl_lb,:onetail_80_pct_intrvl_ub, :onetail_90_pct_intrvl_lb, :onetail_90_pct_intrvl_ub
                   ,:twotail_80_pct_intrvl_lb, :twotail_80_pct_intrvl_ub, :twotail_90_pct_intrvl_lb, :twotail_90_pct_intrvl_ub    
                  ]
            sdf[ sdf[:key].==k, zk]= md[zk]
        end    
    end
end
ConfidenceIntervals(mdolhh,mdf)


write2disk(genMDF(cnts, mocc, mdolocc, mpen, false) ,pwd()*"/mdf.csv")
write2disk(genRDF(mocc,mdolocc,mpen, mdolhh),pwd()*"/rdf.csv")

#temp
write2disk(genMDF(cnts, mocc, mdolocc, mpen, false) ,"/home/rmadmin/g/StatStack/src/mdf.csv")
write2disk(genRDF(mocc,mdolocc,mpen, mdolhh),"/home/rmadmin/g/StatStack/src/rdf.csv")

# summary(fitted(occ_final_model)~finaldata_occ$Creative_Groups)
# summary(finaldata_occ$Trps_POS_P1 ~ finaldata_occ$Creative_Groups)










