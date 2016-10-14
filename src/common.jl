# ------------------- GLOBAL ----------------------------
rdfFmt=[:key,:model,
        :unadj_avg_expsd_hh_pre,:unadj_avg_cntrl_hh_pre,:unadj_avg_expsd_hh_pst,:unadj_avg_cntrl_hh_pst,:adj_mean_expsd_grp,:adj_mean_cntrl_grp,
        :adj_dod_effct,:twotail_pval,:onetail_pval,
        :onetail_80_pct_intrvl_lb,:onetail_80_pct_intrvl_ub,:twotail_80_pct_intrvl_lb,:twotail_80_pct_intrvl_ub,
        :onetail_90_pct_intrvl_lb,:onetail_90_pct_intrvl_ub,:twotail_90_pct_intrvl_lb,:twotail_90_pct_intrvl_ub
       ]

SDF = DataFrame(key=ASCIIString[],
                model=ASCIIString[],
                unadj_avg_expsd_hh_pre=Float64[],
                unadj_avg_cntrl_hh_pre=Float64[],
                unadj_avg_expsd_hh_pst=Float64[],
                unadj_avg_cntrl_hh_pst=Float64[],
                adj_mean_expsd_grp=Float64[],
                adj_mean_cntrl_grp=Float64[],
                adj_dod_effct=Float64[],
                twotail_pval=Float64[],
                onetail_pval=Float64[],
                onetail_80_pct_intrvl_lb=Float64[],
                onetail_80_pct_intrvl_ub=Float64[],
                twotail_80_pct_intrvl_lb=Float64[],
                twotail_80_pct_intrvl_ub=Float64[],
                onetail_90_pct_intrvl_lb=Float64[],
                onetail_90_pct_intrvl_ub=Float64[],
                twotail_90_pct_intrvl_lb=Float64[],
                twotail_90_pct_intrvl_ub=Float64[],
                mean_score0=Float64[],
                mean_score1=Float64[],
                B=Float64[],
                SE=Float64[],
                P=Float64[],
                M=Float64[],
                Mt=Float64[],
                Mc=Float64[]
              )

function pushSDFrow!(sdf::DataFrame, key::AbstractString, v_model::AbstractString)
    push!(sdf, [key; v_model; 0.0; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA])
    idx=length(sdf[1])
    sdf[idx,:unadj_avg_expsd_hh_pre] = NA 
    return sdf
end

function pushSDFrow!(sdf::DataFrame, iDict::OrderedDict)
    key=get(iDict, :key, NA)
    model=get(iDict, :model, NA)
    if isna(key) | isna(model)
        error("Error: pushSDFrow! - missing key or model values")
    else
        B=get(iDict, :B, NA)
        SE=get(iDict, :SE, NA)
        P=get(iDict, :P, NA)
        M=get(iDict, :M, NA)
        Mt=get(iDict, :Mt, NA)
        Mc=get(iDict, :Mc, NA)
        push!(sdf, [key; model; 0.0; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; B; SE; P; M; Mt; Mc])

        #push!(sdf, [key; v_model; 0.0; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA])
        idx=length(sdf[1])
        sdf[idx,:unadj_avg_expsd_hh_pre] = NA
    end
    return sdf
end


#this.sdf=pushSDFrow!(deepcopy(SDF),"Total Campaign","occ")

#push!(SDF, ["Total Campaign"; ""; 0.0; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA; NA])
#SDF[:unadj_avg_expsd_hh_pre]=NA
#SDF[:model]=NA



function df2dict(idf::DataFrame)
    dout=OrderedDict()
    for n in names(idf)
       dout[n] = idf[1,symbol(n)]
    end
    return dout
end

function df2dict(row::DataFrameRow)
    dout=OrderedDict()
    for n in names(row)
       dout[n] = row[symbol(n)]
    end
    return dout
end



function write2disk(idf::DataFrame,fname::AbstractString, repvalues::Bool=true)
    ofile=fname
    if repvalues
        writetable(ofile*"tmp", idf, separator = ',', header = true)
        outfile = open(ofile, "w")
        open(ofile*"tmp") do filehandle
            for line in eachline(filehandle)
                write(outfile, replace(line,"NA",""))
            end
        end
        close(outfile)
        rm(ofile*"tmp")    
    else
        writetable(ofile, idf, separator = ',', header = true)
    end
    println("Complete !!!! - output written to ",ofile)
end

function Base.lowercase(df::DataFrame)
    for (i,v) in enumerate(names(df))   if v != symbol(lowercase(string(v)))  rename!(df,v,symbol(lowercase(string(v)))) end end 
end




"""
app.cfg
P2_Competitor = true
pvalue_lvl = 0.20
excludedKeys = key1,key2,key3
exposed_flag_var = new_exposed_flag
sigLevel = 0.2
random_demos =
random_campaigns =
dropvars = goop,snop
scoring_vars =
TotalModelsOnly=false
"""

function loadCFG(d::OrderedDict,fn::AbstractString)
    function cvt(v::AbstractString, t::DataType)
        if t in [Bool]
            return lowercase(v) =="true" ? true : lowercase(v) =="false" ? false : NA
        elseif t in [nothing]
            return NA
        elseif t in [Array{AbstractString,1}]
            #return AbstractString[string(x) split(v,",")]
            return AbstractString[string(x) for x in split(replace(v,"\"",""),",")]
        elseif t in [Array{Symbol,1}]
            return Symbol[symbol(x) for x in split(replace(v,"\"",""),",")]
        elseif t in [Symbol]
            return symbol(v)
        elseif t in [ASCIIString]
            return string(v)
        elseif t in [Float64]
            return parse(Float64,v)
        else
            return "__ooops__"
        end
    end
    if isfile(fn)
        f = open(fn)
        for ln in eachline(f)
            ln=replace(ln,"\n","")
            sidx=search(ln,"=").start-1
            eidx=search(ln,"=").stop+1
            if sidx>0
                k=strip(ln[1:sidx])
                v=strip(ln[eidx:end])
                if (length(k)>0) & (length(v)>0)
                    #println("LINE:: ~"*k*"~"*v*"~")
                    #println("DEF::       ______ ",d[symbol(k)],"  ----> ",typeof(d[symbol(k)]) )
                    newk=symbol(k)
                    newv = cvt(v,typeof(d[symbol(k)]))
                    #println("IN::        ______ ",symbol(k),"  ----> ",ct, " ....... ", typeof(ct))
                    #println( d[symbol(k)],"|",v," {",typeof(d[symbol(k)]),"}  ----->  ",newk,"|",newv,"  {",typeof(newv),"}")
                    d[symbol(k)] = newv 
                else
                    #println("CFG ERROR :: invalid length : "*"~"*k*"~"*v*"~",typeof(d[symbol(k)]))
                end
            else
                if length(ln)>0   println("CFG ERROR :: invalid cfg parameter : ~",ln,"~") end
            end
        end
        close(f)
    else
        println("Using Default configuration")
    end
    return d
end
