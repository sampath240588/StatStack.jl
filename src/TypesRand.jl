# -------------------- RANDOM EFFECTS ------------------------------

function rkey(key::AbstractString, ilevel::AbstractString="") ilevel != "" ? key*" ("*ilevel*")" : key  end
function rkey(row::DataFrameRow) return rkey(row[:class], row[:level]) end


abstract RanEffect <: ModelEffect


function getRandomFormula(rlist::Vector{Symbol}, excludeCol::Symbol=:NA, df_name::AbstractString="df_data")
    fmula=""
    for s in setdiff(rlist, [excludeCol])
        fmula=fmula*"+"*df_name*"[:"*string(s)*"]"
    end
    return fmula[2:end]
end   


function genRandomTotals(reff::RanEffect, df_data::DataFrame)
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
    
    function calc_reff(row::DataFrameRow)
        tot=0
        for x in 1:length(row)
            tot=tot+row[x]    
        end
        return tot
    end
    df_data[symbol("reff_"*model)] = map(x->calc_reff(x),   eachrow(df_data[rcols]) )
    df_data[symbol("reff_"*model)] = map(Float64,df_data[symbol("reff_"*model)])      #convert to float 
    
    
    # --- Random Effect Mean Scor cols ---
    for row = eachrow(reff.src)
        k = row[:key]
        #needs review
        dest_col0=symbol(k*"_"*model*"_0")
        dest_col1=symbol(k*"_"*model*"_1")
        rcols = setdiff(reff.rcols, [symbol("reff_"*lowercase(row[:class])*"_"*model)]) 
        println("generating : ",k*"_"*model*"_X := sum(",rcols,", :pre_"*model*"_scoreX,+ BX) ")
        function tot_reff(row::DataFrameRow, B::Float64)
            tot=0
            for x in 1:length(row)
                tot=tot+row[x]    
            end
            tot = exp(tot+B)
            if model== "pen"
                return tot = tot / (1+tot)  #exp(eval(parse(fmula0))) ./ (1+exp(eval(parse(fmula0))))
            end
            return tot
        end
        #model
        df_data[dest_col0] = map(x->tot_reff(x, Float64(row[:B0]) ),   eachrow(df_data[vcat(rcols,symbol("pre_"*model*"_score0"))]) )
        df_data[dest_col1] = map(x->tot_reff(x, row[:B1] ),   eachrow(df_data[vcat(rcols,symbol("pre_"*model*"_score1"))]) )
        df_data[dest_col0] = map(Float64,df_data[dest_col0])      #convert to float 
        df_data[dest_col1] = map(Float64,df_data[dest_col1])      #convert to float 
    end    
    
end



type ROcc <: RanEffect 
    sdf::DataFrame
    src::DataFrame
    rcols::Vector{Symbol}
    fmula::AbstractString
    v_model::AbstractString
    function ROcc(df_data::DataFrame, idf::DataFrame)  this=new(); this.v_model="occ"; this.sdf=deepcopy(SDF);
         this.src = idf
         for row in eachrow(idf)
             pushSDFrow!(this.sdf,OrderedDict(:key => row[:key], :model => this.v_model))
         end
        delete!(this.sdf, :B)
        delete!(this.sdf, :SE)
        delete!(this.sdf, :P)
        if length(this.sdf[1]) > 0 genRandomTotals(this,df_data)  end
        return this
    end
end


type RDolOcc <: RanEffect 
    sdf::DataFrame
    src::DataFrame
    rcols::Vector{Symbol}
    fmula::AbstractString
    v_model::AbstractString
    function RDolOcc(df_data::DataFrame, idf::DataFrame)  this=new(); this.v_model="dolocc"; this.sdf=deepcopy(SDF);
         this.src = idf
         for row in eachrow(idf)
             pushSDFrow!(this.sdf,OrderedDict(:key => row[:key], :model => this.v_model))
         end
        delete!(this.sdf, :B)
        delete!(this.sdf, :SE)
        delete!(this.sdf, :P)
        if length(this.sdf[1]) > 0  genRandomTotals(this,df_data) end
        return this
    end
end


type RPen <: RanEffect 
    sdf::DataFrame
    src::DataFrame
    rcols::Vector{Symbol}
    fmula::AbstractString
    v_model::AbstractString
    function RPen(df_data::DataFrame, idf::DataFrame)  this=new(); this.v_model="pen"; this.sdf=deepcopy(SDF);
         this.src = idf
         for row in eachrow(idf)
             pushSDFrow!(this.sdf,OrderedDict(:key => row[:key], :model => this.v_model))
         end
        delete!(this.sdf, :B)
        delete!(this.sdf, :SE)
        delete!(this.sdf, :P)
        if length(this.sdf[1]) > 0 genRandomTotals(this,df_data)  end
        return this
    end
end


function Base.show(io::IO, raneffect::RanEffect) 
    show(io,raneffect.sdf)
end




