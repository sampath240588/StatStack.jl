
save(list = ls(all.names = TRUE), file = "/media/u01/analytics/scoring/CDW5_792/RConvert/renv.RData", envir = .GlobalEnv)



# =========== JULIA ===========


    R""" load("/media/u01/analytics/scoring/CDW5_792/RConvert/renv.RData")  
         ls()
     """

# ------- CFG -------
#function fcmp(l1:)
#end
chk=OrderedDict()
        chk[:P2_Competitor] = cfg[:P2_Competitor] == rcopy("P2_Competitor")
        chk[:pvalue_lvl] = cfg[:pvalue_lvl] == rcopy("pvalue_lvl") 
        chk[:exposed_flag_var] = string(cfg[:exposed_flag_var]) == rcopy("exposed_flag_var")
        chk[:random_demos] = sort(ASCIIString[string(x) for x in cfg[:random_demos]]) == sort(rcopy("random_demos"))
        chk[:random_campaigns] = sort(ASCIIString[string(x) for x in cfg[:random_campaigns]]) == sort(rcopy("random_campaigns"))
        #sort(ASCIIString[string(x) for x in cfg[:random_campaigns]]) == sort(rcopy("dropvars"))
        if (typeof(rcopy("dropvars")) == ASCIIString) & (length(cfg[:dropvars]) == 1)
            chk[:dropvars] = string(cfg[:dropvars][1]) == rcopy("dropvars") 
        else
            chk[:dropvars] = sort(ASCIIString[string(x) for x in cfg[:dropvars]]) == sort(rcopy("dropvars")) 
        end
        chk[:scoring_vars] = sort(ASCIIString[string(x) for x in cfg[:scoring_vars]]) == sort(rcopy("scoring_vars")) 
        chk[:all_mandatory_vars] = sort(ASCIIString[string(x) for x in cfg[:all_mandatory_vars]]) == sort(rcopy("all_mandatory_vars")) 
        chk[:allrandoms] = sort(ASCIIString[string(x) for x in cfg[:allrandoms]]) == sort(rcopy("allrandoms")) 
        chk[:ProScore] = string(cfg[:ProScore]) == rcopy("ProScore") 
        if rcopy("exists('random_demos_factor')")
            chk[:random_demos_factor] = sort(ASCIIString[string(x) for x in cfg[:random_demos_factor]]) == sort(rcopy("random_demos_factor")) 
        else
            chk[:random_demos_factor] =  "random_demos_factor doesn't exist"
        end
        chk[:num_products] = cfg[:num_products] == rcopy("num_products") 
        #addedvars=[:isO,:whyO,:data_NB_NE_B,:data_B_E_NB,:pen_reduction,:occ_reduction,:dolocc_reduction]
        chk[:xVarsDemos] =  sort(ASCIIString[string(x) for x in cfg[:xVarsDemos]]) == sort(rcopy("xVarsDemos")) 
        #setdiff(sort(rcopy("xVarsDemos")),sort(ASCIIString[string(x) for x in cfg[:xVarsDemos]]))
        chk[:xVarsPost] =  sort(ASCIIString[string(x) for x in cfg[:xVarsPost]]) == sort(rcopy("initial_vars_to_exclude"))
        #setdiff(sort(rcopy("initial_vars_to_exclude")),sort(ASCIIString[string(x) for x in cfg[:xVarsPost]]))  
        chk[:iVarsPREPOS] = sort(ASCIIString[string(x) for x in cfg[:iVarsPREPOS]]) == sort(rcopy("PREPOS_vars_to_include"))



        chk[:xVarsP0] =  sort(ASCIIString[string(x) for x in cfg[:xVarsP0]]) == sort(rcopy("P0_vars_to_exclude"))
        #setdiff(sort(rcopy("P0_vars_to_exclude")),sort(ASCIIString[string(x) for x in cfg[:xVarsP0]]))  
        #setdiff(sort(ASCIIString[string(x) for x in cfg[:xVarsP0]]), sort(rcopy("P0_vars_to_exclude")))     


        chk[:xVarsReports] = sort(ASCIIString[string(x) for x in cfg[:xVarsReports]]) == sort(rcopy("vars_to_exclude"))
        #setdiff(sort(rcopy("vars_to_exclude")),sort(ASCIIString[string(x) for x in cfg[:xVarsReports]]))
        #setdiff(sort(ASCIIString[string(x) for x in cfg[:xVarsReports]]),sort(rcopy("vars_to_exclude")))

        #chk[:xvars] = sort(ASCIIString[string(x) for x in cfg[:xvars]]) == sort(rcopy("ALL_vars_to_exclude")) 
        ##setdiff(sort(rcopy("ALL_vars_to_exclude")),sort(ASCIIString[string(x) for x in cfg[:xvars]]))
        ##setdiff(sort(ASCIIString[string(x) for x in cfg[:xvars]]),sort(rcopy("ALL_vars_to_exclude")))
        #:xvars
        #:ivars
        #chk[:ALL_vars_to_exclude] = sort(ASCIIString[string(x) for x in cfg[:ALL_vars_to_exclude]]) == sort(rcopy("ALL_vars_to_exclude"))
        ##setdiff(sort(rcopy("ALL_vars_to_exclude")),sort(ASCIIString[string(x) for x in cfg[:ALL_vars_to_exclude]]))
        ##setdiff(sort(ASCIIString[string(x) for x in cfg[:ALL_vars_to_exclude]]),sort(rcopy("ALL_vars_to_exclude")))


        chk[:negativevars] =  sort(ASCIIString[string(x) for x in cfg[:negativevars]]) == sort(rcopy("negativevars"))
        #setdiff(sort(rcopy("negativevars")),sort(ASCIIString[string(x) for x in cfg[:negativevars]]))  
        #setdiff(sort(ASCIIString[string(x) for x in cfg[:negativevars]]), sort(rcopy("negativevars")))  


        chk[:positivevars] =  sort(ASCIIString[string(x) for x in cfg[:positivevars]]) == sort(rcopy("positivevars"))
        #setdiff(sort(rcopy("positivevars")),sort(ASCIIString[string(x) for x in cfg[:positivevars]]))  
        #setdiff(sort(ASCIIString[string(x) for x in cfg[:positivevars]]), sort(rcopy("positivevars")))  
        #:ALL_vars_to_exclude
        #:negativevars
        #:positivevars
        initial_data=rcopy("initial_data")
        chk[:features] = sort(names(df_data)) == sort(names(initial_data))
        #setdiff(sort(names(initial_data)),sort(names(df_data)))
        #setdiff(sort(names(df_data)) , sort(names(initial_data))  )
       

    
# ---- DF_DATA Check -----
save(list = ls(all.names = TRUE), file = "/media/u01/analytics/scoring/CDW5_792/RConvert/renv.RData", envir = .GlobalEnv)

R""" load("/media/u01/analytics/scoring/CDW5_792/RConvert/renv.RData")  
    ls()
 """
greparr=names(df_in)
function grep(v::AbstractString,xvars::Array{Symbol}=greparr)   #function grep(v::AbstractString) filter(x->ismatch(Regex("$(v)"), string(x)), vars) end
     filter(x->ismatch(Regex("$(v)"), string(x)), xvars)
end
function grep(vi::Symbol,xvars::Array{Symbol}=greparr)   #function grep(v::AbstractString) filter(x->ismatch(Regex("$(v)"), string(x)), vars) end
    v = string(vi) 
    filter(x->ismatch(Regex("$(v)"), string(x)), xvars)
end
finaldata = rcopy("finaldata")
initial_data = finaldata
initial_data = rcopy("initial_data")

# ------------------------------------------------------------------------------------------------------
#@rget finaldata;
jlst = sort(setdiff(names(df_data),[:data_B_E_NB,:data_NB_NE_B,:occ_reduction,:dolocc_reduction ,:pen_reduction,]))

setdiff(names(df_data),names(finaldata))
setdiff(names(df_data),names(initial_data))
#setdiff(names(df_in),names(rcopy("finaldata")))
setdiff(names(df_in),names(finaldata)) 
grep(:exposed_flag,names(df_in))
grep(:exposed_flag,names(finaldata))

println("Only In df_in:::\n",setdiff(names(df_in),names(finaldata))  )
println("Only In finaldata:::\n",setdiff(names(finaldata), names(df_in)))   

# --- Data ----
println("df_data : ",nrow(df_data),"  ---->  initial_data : ",nrow(initial_data))
#df_data[df_data[:number_of_children_in_living_Un].>=4,:number_of_children_in_living_Un] = 4  # aggregate #of children for 4+

#NOTTE: remove observations;
#            291 - remove person_1_gender
#            298 - remove HH exposed/non-exposed
#            look for dropleveles - not sureif has impact
#df_data=df_in
# :number_of_children_in_living_Un  # 4+ values
#  df_data[df_data[:number_of_children_in_living_Un].>=4,:number_of_children_in_living_Un] = 4
function dict2str(di::Dict)
    od=Dict()
    for (k, v) in di  
        od[string(k)] = string(v)
    end
    return od
end
function cm()
    di=dict2str(countmap(df_in[:number_of_children_in_living_Un])) 
    dd=dict2str(countmap(df_data[:number_of_children_in_living_Un]))
    fd=dict2str(countmap(finaldata[:number_of_children_in_living_Un]))
    id=dict2str(countmap(initial_data[:number_of_children_in_living_Un]))
    #for k in sort(unique([replace(string(x),"+","") for x in union(keys(dd),keys(fd),keys(id))]))

    dfo=DataFrame(key=AbstractString[],df_in=AbstractString[],df_data=AbstractString[],finaldata=AbstractString[],initial_data=AbstractString[])
    for k in sort(unique([string(x) for x in union(keys(di),keys(dd),keys(fd),keys(id))]))
        iv=AbstractString[k,get(di,k,""),get(dd,k,""),get(fd,k,""),get(id,k,"")] 
        push!(dfo,iv)
        #push!(df, [3  6])
        #push!(dfo,[get(dd,k,""),get(fd,k,""),get(id,k,"")]) 
        #println(":::",iv)
    end
    return dfo
end
cm()



#   :group   # values and counts/proportions
function cm()
    di=dict2str(countmap(df_in[ df_data[:isO].==false, :group]))
    fd=dict2str( countmap(finaldata[:group]) )
    dfo=DataFrame(key=AbstractString[],df_in=AbstractString[],finaldata=AbstractString[])
    for k in sort(unique([string(x) for x in union(keys(di),keys(fd),)]))
        iv=AbstractString[k,get(di,k,""),get(fd,k,"")] 
        push!(dfo,iv)
        #push!(df, [3  6])
        #push!(dfo,[get(dd,k,""),get(fd,k,""),get(id,k,"")]) 
        #println(":::",iv)
    end
    return dfo
end
cm()


# L_290 remove HHs with no gender info  #finaldata <- finaldata[!person_1_gender=="U",]
function cm()
    di=dict2str(countmap(df_in[  df_data[:isO].==false, :person_1_gender]))
    fd=dict2str( countmap(finaldata[:person_1_gender]) )
    dfo=DataFrame(key=AbstractString[],df_in=AbstractString[],finaldata=AbstractString[])
    for k in sort(unique([string(x) for x in union(keys(di),keys(fd),)]))
        iv=AbstractString[k,get(di,k,""),get(fd,k,"")] 
        push!(dfo,iv)
        #push!(df, [3  6])
        #push!(dfo,[get(dd,k,""),get(fd,k,""),get(id,k,"")]) 
        #println(":::",iv)
    end
    return dfo
end
cm()





# # aggregate U and L levels of hh income  #finaldata <- finaldata[estimated_hh_income=="U",estimated_hh_income:="L"]
function cm()
    di=dict2str(countmap(df_in[  df_data[:isO].==false, :estimated_hh_income]))
    fd=dict2str( countmap(finaldata[:estimated_hh_income]) )
    dfo=DataFrame(key=AbstractString[],df_in=AbstractString[],finaldata=AbstractString[])
    for k in sort(unique([string(x) for x in union(keys(di),keys(fd),)]))
        iv=AbstractString[k,get(di,k,""),get(fd,k,"")] 
        push!(dfo,iv)
        #push!(df, [3  6])
        #push!(dfo,[get(dd,k,""),get(fd,k,""),get(id,k,"")]) 
        #println(":::",iv)
    end
    return dfo
end
cm()


function cm()
    o=Dict()
    for r in cfg[:random_campaigns]
        #di = countmap(df_data[df_data[:isO].==false, r])
        di = dict2str(countmap(df_data[df_data[:isO].==false, r]) )
        fd = dict2str( countmap(finaldata[r]) )
        dfo=DataFrame(key=AbstractString[],df_data=AbstractString[],finaldata=AbstractString[])
        for k in sort(unique([string(x) for x in union(keys(di),keys(fd),)]))
            iv=AbstractString[k,get(di,k,""),get(fd,k,"")] 
            push!(dfo,iv)
        end
        o[r] = dfo
    end
    return o
end
o=cm()



nrow(df_data[df_data[:isO].==false, : ])
nrow(initial_data)
