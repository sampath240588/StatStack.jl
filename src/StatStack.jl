isdefined(Base, :__precompile__) && __precompile__()

#__precompile__()



"""

spreadsheet
 dolhh und_ctrlpst __ same as tot
Bx1 := estimate
Bx0 := ? intercept
 Bx0 _ Bx1
 unexp _ exp

CIs
O_O is observed

NOTES:   =:= Regression: intercept+[fixed*coef]+[raneff*coef].... +group1 coef for exposed
============================= TOTAL ===============================
--- OCC ---
adj_mean_cntrl_grp = mean(regression0)  ::  buyer_pos_p1=1      #include group=0 for CIs
adj_mean_expsd_grp = mean(regression1)  ::  buyer_pos_p1=1      #include group=1 for CIs
unadj_avg_cntrl_hh_pre = mean(trps_pre_p1)  ::  buyer_pre_p1=1 & group=0
unadj_avg_expsd_hh_pre = mean(trps_pre_p1)  ::  buyer_pre_p1=1 & group=1
unadj_avg_cntrl_hh_pst = mean(trps_pos_p1)  ::  buyer_pos_p1=1 & group=0
unadj_avg_expsd_hh_pst = mean(trps_pos_p1)  ::  buyer_pos_p1=1 & group=1
--- DOLOCC ---
adj_mean_cntrl_grp = mean(regression0)  ::  buyer_pos_p1=1      #include group=0 for CIs
adj_mean_expsd_grp = mean(regression1)  ::  buyer_pos_p1=1      #include group=1 for CIs
unadj_avg_cntrl_hh_pre = mean(dol_per_trip_pre_p1)  ::  buyer_pre_p1=1 & group=0
unadj_avg_expsd_hh_pre = mean(dol_per_trip_pre_p1)  ::  buyer_pre_p1=1 & group=1
unadj_avg_cntrl_hh_pst = mean(dol_per_trip_pos_p1)  ::  buyer_pos_p1=1 & group=0
unadj_avg_expsd_hh_pst = mean(dol_per_trip_pos_p1)  ::  buyer_pos_p1=1 & group=1
--- PEN ---
adj_mean_cntrl_grp = mean(regression0)  ::    #include group=0 for CIs
adj_mean_expsd_grp = mean(regression1)  ::    #include group=1 for CIs
unadj_avg_cntrl_hh_pre = mean(buyer_pre_p1)  ::  group=0
unadj_avg_expsd_hh_pre = mean(buyer_pre_p1)  ::  group=1
unadj_avg_cntrl_hh_pst = mean(buyer_pos_p1)  ::  group=0
unadj_avg_expsd_hh_pst = mean(buyer_pos_p1)  ::  group=1

============================= BREAKS ===============================
--- OCC ---
(for each Break)
key_occ_0
        adj_mean_cntrl_grp = mean(key_occ_0) :: buyer_pos_p1=1
        adj_mean_expsd_grp = mean(key_occ_1) :: buyer_pos_p1=1
        unadj_avg_expsd_hh_pre = mean(trps_pre_p1) :: group=1 & break=level
        unadj_avg_expsd_hh_pst = mean(trps_pos_p1) :: group=1 & break=level
        if exposed
            unadj_avg_cntrl_hh_pre = total campaign value
            unadj_avg_cntrl_hh_pst = total campaign value
        else for demos
            unadj_avg_cntrl_hh_pre = mean(trps_pre_p1) :: group=0 & break=level 
            unadj_avg_cntrl_hh_pst = mean(trps_pos_p1) :: group=0 & break=level  
        end      
--- DOLOCC ---
(for each Break)
        adj_mean_cntrl_grp = mean(key_dolocc_0) :: buyer_pos_p1=1
        adj_mean_expsd_grp = mean(key_dolocc_1) :: buyer_pos_p1=1
        unadj_avg_expsd_hh_pre = mean(dol_per_trip_pre_p1) :: group=1 & break=level
        unadj_avg_expsd_hh_pst = mean(dol_per_trip_pos_p1) :: group=1 & break=level
        if exposed
            unadj_avg_cntrl_hh_pre = total campaign value
            unadj_avg_cntrl_hh_pst = total campaign value
        else for demos
            unadj_avg_cntrl_hh_pre = mean(dol_per_trip_pre_p1) :: group=0 & break=level 
            unadj_avg_cntrl_hh_pst = mean(dol_per_trip_pos_p1) :: group=0 & break=level 
        end   
--- PEN ---
(for each Break)
        adj_mean_cntrl_grp = mean(key_pen_0) ::
        adj_mean_expsd_grp = mean(key_pen_1) ::
        unadj_avg_expsd_hh_pre = mean(buyer_pre_p1) :: group=1 & break=level
        unadj_avg_expsd_hh_pst = mean(buyer_pos_p1) :: group=1 & break=level
        if exposed
            unadj_avg_cntrl_hh_pre = total campaign value
            unadj_avg_cntrl_hh_pst = total campaign value
        else
            unadj_avg_cntrl_hh_pre = mean(buyer_pre_p1) :: group=0 & break=level 
            unadj_avg_cntrl_hh_pst = mean(buyer_pos_p1) :: group=0 & break=level 
        end  

======================== Regression =======================
genFixedTotals
  1.	Create fixed pre_model_score0/1  (e.g. df_data[:pre_occ_score0])
       	sum[Coef*val]+intercept+logvar ……… +group1 for exposed

genRandomTotals
loop through raneff.src[effects]
  1.	create coef cols := reff_ranfx_model   (e.g. df_data[:reff_creative_nm_occ] )
       	Set to 0.0
       	For each level in effect : update reff_ranfx_model with coef :: col=level group=?
  2.	Create total col reff_model (e.g. df_data[:reff_occ])
       	Sum(reff_ranfx_model cols for all breaks)
  3.	Create key_model_0/1    (e.g. df_data[symbol("placement_nm (Desktop_Display)_occ_1")])
       	Sum all cols in 1 except this col
       	Add coef for this level := score as whole model with this break = this level
       	This is mean_score1/0, and mean adj col

genScoreTotals
  1.	create model_score0   (e.g. occ_score1)
       	exp(fixed + random totals)
       	exp(df_data[:pre_occ_score0] + df_data[:reff_occ]  )
       	e.g. exp(df_data[:pre_occ_score0] + df_data[:reff_occ]  )
       	or pen : df_data[:pen_score0] = exp((df_data[:pre_pen_score0] + df_data[:reff_pen])) ./ ( exp((df_data[:pre_pen_score0] + df_data[:reff_pen])) +1)
       	this is used for fixed mean_score0/1 

"""



module StatStack

using Compat, DataArrays, GLM, DataFrames, Distributions, NLopt, Showoff, StatsBase, DataStructures, StatsFuns, JuMP, DBAPI, JavaCall, JDBC
#using JSON, Requests, HttpParser

import DataArrays
import DBAPI, JavaCall, JDBC
import StatsBase: coef, coeftable, df, deviance, fit!, fitted, loglikelihood, model_response, nobs, vcov
import Base: cond, std
import Distributions: Bernoulli, Binomial, Poisson, Gamma
import GLM: LogitLink, LogLink, InverseLink
import DataFrames: @~

export
       @~,
       calcPValue_Opt,
       calcCI_LB_Opt,
       calcCI_UB_Opt,
       calcCI_Opt,
       #calcCI,
       CIs,
       ZDict,
       v_ttl,
       ##Base.lowercase
       write2disk,
       rdfFmt,
       SDF,
       pushSDFrow!,
       ModelEffect,
       FixedEffect,
       FOcc,
       FDolOcc,
       FPen,
       FDolHH,
       RanEffect,
       ROcc,
       RDolOcc,
       RPen,
       RDolHH,
       MModel,
       MOcc, 
       MDolOcc, 
       MPen,
       MDolHH,
       MDF,
       RDF,
       rkey,
       getRandomFormula,
       genRandomTotals,
       genFixedTotals,
       genScoreTotals,
       Cnts,
       getk,
       genMDF,
       genRDF,
       genRndMDF,
       genFixedMDF,
       CIs_O,
       genCnts,
       hhcounts,
       df2dict,
       saveDBora,
       loadCFG,
       tst2

#abstract ModelEffect
#abstract FixedEffect <: ModelEffect

#abstract MixedModel <: RegressionModel # model with fixed and random effects

#  include("/home/rmadmin/zstat.jl/src/StatStack.jl")

import Base: ==, *

#type tst2
#    function tst2(df_data::DataFrame)
#        #eval(parse("df_data[:reff_creative_groups_pen_1]"))
#        p=parse("df_data[:Y] = df_data[:reff_creative_groups_pen_1]")
#        @eval $p
#    end
#end

eval(x) = Core.eval(StatStack, x)
eval(m,x) = Core.eval(m, x)

function tst2(df_data::DataFrame)
        #eval(parse("df_data[:reff_creative_groups_pen_1]"))
        #p=parse("df_data[:Y] = df_data[:reff_creative_groups_pen_1]")
        #@eval $p
        #t = :(sum(df_data[:group]))
        eval(parse("sum(df_data[:group])"))
end

include("CIfuncs.jl")
include("TypesFixed.jl")
include("TypesRand.jl")
include("common.jl")
include("TypesModel.jl")
include("db.jl")





end # module
