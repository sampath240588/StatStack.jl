# -------------------- FIXED EFFECTS --------------------
abstract ModelEffect
abstract FixedEffect <: ModelEffect

function getFixedFormula(idf::DataFrame, logvar::AbstractString="", df_name::AbstractString="df_data")
    v_intercept= idf[idf[:x] .== "intercept" , :estimate][1]
    v_group1 = idf[idf[:x] .== "group1" , :estimate][1]
    v_out = string(v_intercept)
    for fixedeff in setdiff(idf[:x], ["intercept","group1"])
        v_coef=  idf[idf[:x] .== fixedeff, :estimate][1]
        v_out *= " + ("*string(v_coef)*"*"*df_name*"[:"*fixedeff*"])"
    end
    if length(logvar) > 0
        v_out*="+log( "*df_name*"[:"*logvar*"] + 1) "
    end
    v_out2=v_out*"+"*string(v_group1)
    return v_out, v_out2
end



function genFixedTotals(feff::FixedEffect, df_data::DataFrame)
    model=feff.v_model
    src=feff.src
    logvar=feff.logvar
    intercept= src[src[:x] .== "intercept" , :estimate][1]
    group1 = src[src[:x] .== "group1" , :estimate][1]
    coefs=Float64[]
    cols=Symbol[]
    for fixedeff in setdiff(src[:x], ["intercept","group1"])
        v_coef =  src[src[:x] .== fixedeff, :estimate][1]
        push!(coefs,v_coef)
        push!(cols,symbol(fixedeff))
    end
    if length(logvar) > 0
        push!(cols,symbol(logvar))
    end
    println(coefs)
    println(cols)

    function calc_feff(row::DataFrameRow, ftype::Int64)
        tot=intercept
        ccnt=length(row)
        if length(logvar) > 0
            ccnt=ccnt-1
        end
        for x in 1:ccnt
            tot=tot+(row[x]*coefs[x])    
        end
        if length(logvar) > 0
            tot=tot+log(row[symbol(logvar)]+1)
        end
        if ftype == 1
            tot=tot+group1
        end
        return tot
    end
    df_data[symbol("pre_"*model*"_score0")] = map(x->calc_feff(x,0), eachrow(df_data[cols]) ) 
    df_data[symbol("pre_"*model*"_score1")] = map(x->calc_feff(x,1), eachrow(df_data[cols]) ) 
    
    df_data[symbol("pre_"*model*"_score0")]=map(Float64,df_data[symbol("pre_"*model*"_score0")])
    df_data[symbol("pre_"*model*"_score1")]=map(Float64,df_data[symbol("pre_"*model*"_score1")])
    #df_data[symbol("pre_occ_score0")]=eval(parse(mocc.feff.fmula0))         
    #df_data[symbol("pre_occ_score1")]=eval(parse(mocc.feff.fmula1))
end
#genFixedTotals(mocc.feff,df_data)
#df_data[[:pre_occ_score1,:xpre_occ_score1]]


type FOcc <: FixedEffect
    sdf::DataFrame
    src::DataFrame
    fmula0::AbstractString
    fmula1::AbstractString
    v_model::AbstractString
    logvar::AbstractString
    function FOcc(df_data::DataFrame, idf::DataFrame) this=new(); this.v_model="occ"; this.logvar="trps_pre_p1"; 
        this.sdf=pushSDFrow!(deepcopy(SDF),"Total Campaign",this.v_model);
        this.src = idf
        this.fmula0, this.fmula1 = getFixedFormula(this.src,this.logvar)
        #println(this.fmula0,"\n\n",this.fmula1)
        #df_data[symbol("pre_"*this.v_model*"_score0")]=eval(parse(this.fmula0))
        #df_data[symbol("pre_"*this.v_model*"_score1")]=eval(parse(this.fmula1))
        this.sdf[:B] = idf[idf[:x] .== "group1", :estimate][1]
        this.sdf[:SE] = idf[idf[:x] .== "group1", :std_error][1]
        this.sdf[:P] = idf[idf[:x] .== "group1", :pr_z_][1]
        genFixedTotals(this,df_data)
        return this 
    end
end



type FDolOcc   <: FixedEffect
    sdf::DataFrame
    src::DataFrame
    fmula0::AbstractString
    fmula1::AbstractString
    v_model::AbstractString
    logvar::AbstractString
    function FDolOcc(df_data::DataFrame, idf::DataFrame) this=new(); this.v_model="dolocc"; this.logvar="dol_per_trip_pre_p1"; 
        this.sdf=pushSDFrow!(deepcopy(SDF),"Total Campaign",this.v_model);
        this.src = idf
        this.fmula0, this.fmula1 = getFixedFormula(this.src,this.logvar)
        #println(this.fmula0,"\n\n",this.fmula1)
        #df_data[symbol("pre_"*this.v_model*"_score0")]=eval(parse(this.fmula0))         
        #df_data[symbol("pre_"*this.v_model*"_score1")]=eval(parse(this.fmula1))
        this.sdf[:B] = idf[idf[:x] .== "group1", :estimate][1]
        this.sdf[:SE] = idf[idf[:x] .== "group1", :std_error][1]
        this.sdf[:P] = idf[idf[:x] .== "group1", :pr_z_][1]
        genFixedTotals(this,df_data)
        return this 
    end
end


type FPen   <: FixedEffect
    sdf::DataFrame
    src::DataFrame
    fmula0::AbstractString
    fmula1::AbstractString
    v_model::AbstractString
    logvar::AbstractString
    function FPen(df_data::DataFrame, idf::DataFrame) this=new(); this.v_model="pen"; this.logvar="buyer_pre_p1"; 
        this.sdf=pushSDFrow!(deepcopy(SDF),"Total Campaign",this.v_model);
        this.src = idf
        this.fmula0, this.fmula1 = getFixedFormula(this.src,this.logvar)
        #println(this.fmula0,"\n\n",this.fmula1)
        #df_data[symbol("pre_"*this.v_model*"_score0")]=eval(parse(this.fmula0))
        #df_data[symbol("pre_"*this.v_model*"_score1")]=eval(parse(this.fmula1))
        this.sdf[:B] = idf[idf[:x] .== "group1", :estimate][1]
        this.sdf[:SE] = idf[idf[:x] .== "group1", :std_error][1]
        this.sdf[:P] = idf[idf[:x] .== "group1", :pr_z_][1]
        genFixedTotals(this,df_data)
        return this 
    end
end

type FDolHH
    sdf::DataFrame
    v_model::AbstractString
    function FDolHH() this=new(); this.v_model="dolhh"; this.sdf=pushSDFrow!(deepcopy(SDF),"Total Campaign",this.v_model); return this  end
end


function Base.show(io::IO, fdolhh::FDolHH) 
    show(io,fdolhh.sdf)
end
function Base.show(io::IO, fixedeffect::FixedEffect) 
    show(io,fixedeffect.sdf)
end


