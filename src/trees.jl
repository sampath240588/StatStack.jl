""" 
# DataSet
drop table if exists jennie.jennie6_breaks_RF;
CREATE EXTERNAL TABLE IF NOT EXISTS jennie.jennie6_breaks_RF
  ( household_id string,
    iri_week string,
    date1 string,
    creative_id int,
    placement_id int,
    gross_imps bigint,
    val_imps bigint,
    placement_nm string,
    creative_nm string,
    publisher string
   )
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY ‘\n’ STORED AS TEXTFILE LOCATION '/mapr/mapr04p/analytics0001/analytic_users/Models/trees/tables/';



# show create table jennie.jennie6_breaks_RF;

Insert Overwrite Table jennie.jennie6_breaks_RF
select distinct
b.household_id ,
a.iri_week ,
a.date1 ,
a.creative_id ,
a.placement_id ,
a.gross_imps ,
a.val_imps ,
c.placement_nm,
d.creative_nm, 
c.publisher
from daily.daily_unioned a
left join jennie.placements6 c on a.placement_id = c.id
left join jennie.creatives6 d on a.creative_id = d.id
inner JOIN WH_SUPPLMENTAL.HOUSEHOLD_DEPENDENT_ID_MAP b on a.lr_id = b.dependent_id 
where b.dependent_type = 'LVRAMP_ID' 
      and b.household_type = 'EXPERIAN' 
      and b.retailer = 'COMSCORE' 
      and b.data_supplier = 'EXPERIAN' 
      and b.current_state = 'MATCHED'
      and a.clientid in('21884504') 
      and iri_week < 1934
order by household_id, date1;

=============================================================================

select distinct
b.household_id ,
a.iri_week ,
a.date1 ,
a.creative_id ,
a.placement_id ,
a.gross_imps ,
a.val_imps 
from daily.daily_unioned a,
WH_SUPPLMENTAL.HOUSEHOLD_DEPENDENT_ID_MAP b 
where a.lr_id = b.dependent_id 
      and b.dependent_type = 'LVRAMP_ID' 
      and b.household_type = 'EXPERIAN' 
      and b.retailer = 'COMSCORE' 
      and b.data_supplier = 'COMSCORE' 
      and b.current_state = 'MATCHED'
      and a.clientid in('21884504') 
      and iri_week < 1934
order by household_id, date1;

select count(*) 
from daily.daily_unioned a, 
     WH_SUPPLMENTAL.HOUSEHOLD_DEPENDENT_ID_MAP b 
where a.lr_id = b.dependent_id
      and a.clientid in('21884504')
      and iri_week < 1934
      and b.current_state = 'MATCHED'
      and b.retailer = 'COMSCORE' 
      and b.data_supplier = 'EXPERIAN'
;

select count(*) 
from WH_SUPPLMENTAL.HOUSEHOLD_DEPENDENT_ID_MAP b 
where b.retailer = 'COMSCORE' 
      and b.data_supplier = 'COMSCORE'
;

select distinct(retailer) from WH_SUPPLMENTAL.HOUSEHOLD_DEPENDENT_ID_MAP;

select distinct(data_supplier) from WH_SUPPLMENTAL.HOUSEHOLD_DEPENDENT_ID_MAP;
"""


# ========================================================== DATASET AGREGATION =============================================

export JULIA_NUM_THREADS=4
Threads.nthreads()


using DataStructures, DataFrames, StatsFuns, GLM , JuMP,NLopt, HDF5, JLD, Distributions, MixedModels, RCall, StatsBase, xCommon, Feather

dfd = readtable("/mnt/resource/analytics/trees/exposure.csv",header=false);
#names!(dfdx, [:household_id, :iri_week, :date1, :creative_id, :placement_id, :gross_imps, :val_imps ,:placement, :creative, :publisher] )
names!(dfd, [:hh_id, :week, :date, :creative_id, :placement_id, :gross_imps, :imps ,:placement, :creative, :publisher] )   

dfd=sort(dfd,cols=[:date])


dfd[:chunks]=-1
hhlst = unique(dfd[:hh_id])
#hhlen=length(hhlst)
hhlen = Int(floor(length(hhlst)/10000))
for hhcnt in 0:hhlen
    lb=(hhcnt*50000)+1
    if hhcnt < hhlen
        ub=(hhcnt+1)*50000
        println("chuncking : (",lb,":",ub,")")
        #dfd[findin(dfd[:panid],panids) ,:iso] = true
        dfd[ findin(dfd[:hh_id],hhlst[lb:ub])  ,:chunks] = hhcnt+1
    else
        println("chuncking : (",lb,":end)")
        #dfd[ findin(dfd[:hh_id],hhlst[lb:end])  ,:chunks] = hhcnt+1
        dfd[dfd[:chunks].<0,:chunks]=hhcnt+1
    end
end



#dfd[:dateX] = map(x->    Date(string(x),Dates.DateFormat("yyyymmdd"))   ,dfd[:date])
#dfd[:dateX] = convert(Array{Date}, dfd[:dateX])

#lag_cnt=Dict(:one=>1,:four=>4,:eight=>8,:sixteen=>16)

for brk in [:placement, :creative, :publisher]
    for lvl in unique(dfd[brk])
        for lag in [1,4] #[:one,:four,:eight]
            col=Symbol(string(brk)*"_"*replace(string(lvl)," ","_")*"_"*string(lag))
            println(col)
            dfd[col]=0
        end
    end
end

for chunk in 1:2   #maximum(dfd[:chunks])
    for row in eachrow(dfd[dfd[:chunks].==chunk,:])
        println(row[:hh_id]," ~~~ ",row[:date])
    end
end



x=by(dfd,[:date,:hh_id], df-> sum(     df[ ()&()  :imps]       ))
x=by(dfd,[:date,:hh_id], df-> sum(  df[ ()&()  :imps]       ))


y=by(dfd, [:date,:hh_id,:publisher], nrow)

    
    
    
by(dfd, [:date,:hh_id]) do df
    DataFrame(m = mean(df[:PetalLength]), s² = var(df[:PetalLength]))
end


gb = groupby(dfd, [:date])


by(iris, :Species) do df
    DataFrame(m = mean(df[:PetalLength]), s² = var(df[:PetalLength]))
end


#=========================== END =============================

x=by(dfd,:hh_id, df-> sum(df[:imps] ))
or : x=by(dfd,:hh_id, df-> sum(df[:gross_imps] ))
x[x[:x1].>0,:]
x[x[:hh_id].==1000120003,:]


function oliers(dfd::DataFrame, k::Symbol, c::Symbol)
    m = median(dfd[c])
    #dfd[c_med1] = abs(dfd[c]-m)
    #MAD=median(dfd[c_med1])
    MAD = median(abs(dfd[c]-m))
    dfd[c_med2] =  ((dfd[c]-m)*0.6745) / MAD 

    dfd_zsc = dfd[abs(dfd[c_med2]) .< 3.5 ,:]
    df_in_pout =  join(df_in, dfd_zsc, on = [ k, k ], kind = :inner)
end
oliers(dfd,:household_id, :val_imps)


# by(dfd, :, df -> sum(df[:PetalLength]))
function Pre_out(df_in::DataFram)
    df_cat_pre = df_in[df_in[:Buyer_Pre_P0] .==1 , [:Prd_0_Net_Pr_PRE,:experian_id]]
    median_df = median(df_cat_pre[:Prd_0_Net_Pr_PRE])
    df_cat_pre[:Prd_0_Net_Pr_PRE_med1] = abs(df_cat_pre[:Prd_0_Net_Pr_PRE]-median_df)
    MAD=median(df_cat_pre[:Prd_0_Net_Pr_PRE_med1])
    df_cat_pre[:Prd_0_Net_Pr_PRE_med2] = (0.6745*(df_cat_pre[:Prd_0_Net_Pr_PRE]-median_df))/MAD
    df_cat_pre_zsc = df_cat_pre[abs(df_cat_pre[:Prd_0_Net_Pr_PRE_med2]) .< 3.5 ,:]
    df_in_pout =  join(df_in, df_cat_pre_zsc, on = [ :experian_id, :experian_id ], kind = :inner)
end
Pre_out(df_in)


# --- parallel RF

function build_forest(labels, features, nsubfeatures, ntrees, ncpu=1)
                  if ncpu > nprocs()
                       addprocs(ncpu - nprocs())
                  end
                  Nlabels = length(labels)
                  Nsamples = int(0.7 * Nlabels)
                  forest = @parallel (vcat) for i in [1:ntrees]
                      inds = rand(1:Nlabels, Nsamples)
                      build_tree(labels[inds], features[inds,:], nsubfeatures)
                  end
                  return [forest]
              end

+++++++++Distribute Macro ++++++++++++++++++++++++++++
function sync_add(r)
    spawns = get(task_local_storage(), :SPAWNS, ())
    if spawns !== ()
        push!(spawns[1], r)
        if isa(r, Task)
            tls_r = get_task_tls(r)
            tls_r[:SUPPRESS_EXCEPTION_PRINTING] = true
        end
    end
    r
end

spawnat(p, thunk) = sync_add(remotecall(thunk, p))

macro spawnat(p, expr)
    expr = localize_vars(esc(:(()->($expr))), false)
    :(spawnat($(esc(p)), $expr))
end

macro run(p, expr)
    expr = localize_vars(esc(:(()->($expr))), false)
    :(spawnat($(esc(p)), $expr))
end
+++++++++++++++++++++++++++++++++++++


================== XGBoost ============================================= https://www.kaggle.com/wacaxx/rossmann-store-sales/julia-xgboost-starter-code
using BinDeps, DataFrames, XGBoost
#using Dates

y=convert(Array{Float32},dfd[:dol_per_trip_pre_p1] )
x=dfd[setdiff(names(dfd),[:dol_per_trip_pre_p1])]
#Define target
#y = convert(Array{Float32}, train[:Sales])

#Transform data to XGBoost matrices
#trainArray = convert(Array{Float32},  x[:, vcat(numericalColumns, categoricalColumns)])
#testArray = convert(Array{Float32}, test[:, vcat(numericalColumns, categoricalColumns)])
#dtrain = DMatrix(trainArray, label = costTrainingLog)
#dtest = DMatrix(testArray)

num_round = 250
param = ["eta" => 0.2, "max_depth" => 20, "objective" => "reg:linear", "silent" => 1]

XGBoostModel = xgboost(dtrain, num_round, param = param)


#Predictions using test data
preds = predict(XGBoostModel, dtest)
#Round to zero closed stores
preds[closedStoreIdx] = 0

#Write Results
sampleSubmission = DataFrame(Id = tesdIdx, Sales = preds)

==================  TREES WORKING ==================== examples https://github.com/bensadeghi/DecisionTree.jl
using DecisionTree
y=dfd[:dol_per_trip_pre_p1]
x=dfd[setdiff(names(dfd),[:dol_per_trip_pre_p1])]

model = build_tree(Array(y), Array(x), 5)
or 
m2 = build_forest(Array(y), Array(x), 2, 10, 5, 0.7)

apply_tree(model, Array(x))


xtest=x[1:1000,:]
xtrain=x[1000:end,:]
ytest=y[1:1000]
ytrain=y[1000:end]
model = build_tree(Array(ytrain), Array(xtrain), 5)
res = apply_tree(model, Array(xtest))
map((x,y)->x-y, res,ytest)

r2 = nfoldCV_tree(Array(ytrain), Array(xtrain), 3, 5)

m2 = build_forest(Array(ytrain), Array(xtrain), 2, 10, 5, 0.7)
res = apply_tree(m2, Array(xtest))

====================== END ===========================

df_in[:Dol_per_Trip_PRE_P1]

setdiff(names(df_in),:Dol_per_Trip_PRE_P1)


y=df_in[:Dol_per_Trip_PRE_P1]
x=df_in[setdiff(names(df_in),[:Dol_per_Trip_PRE_P1])]
#model = build_forest(y,x, 20, 50, 1.0)
model = build_tree(y, Array(x),20, 50, 1.0);

using DecisionTree

model = build_forest(df_in[:Dol_per_Trip_PRE_P1], df_in[[setdiff(names(df_in),:Dol_per_Trip_PRE_P1)]], 20, 50, 1.0)

x=Array(x[setdiff(names(x),[:core_based_statistical_areas,:person_1_birth_year_and_month])])

model = build_tree(Array(y), Array(x),20, 50, 1.0);






========================================================
#https://github.com/dmlc/XGBoost.jl/blob/master/demo/basic_walkthrough.jl

using XGBoost

function readlibsvm(fname::ASCIIString, shape)
    dmx = zeros(Float32, shape)
    label = Float32[]
    fi = open(fname, "r")
    cnt = 1
    for line in eachline(fi)
        line = split(line, " ")
        push!(label, float(line[1]))
        line = line[2:end]
        for itm in line
            itm = split(itm, ":")
            dmx[cnt, int(itm[1]) + 1] = float(int(itm[2]))
        end
        cnt += 1
    end
    close(fi)
    return (dmx, label)
end

train_X, train_Y = readlibsvm("/home/rmadmin/.julia/v0.4/XGBoost/data/agaricus.txt.train", (6513, 126))
test_X, test_Y = readlibsvm("/home/rmadmin/.julia/v0.4/XGBoost/data/agaricus.txt.test", (1611, 126))

=============================
using DecisionTree

#Train random forest with
#20 for number of features chosen at each random split,
#50 for number of trees,
#and 1.0 for ratio of subsampling.
model = build_forest(yTrain, xTrain, 20, 50, 1.0)


using DataStructures, DataFrames, StatsFuns, GLM , JuMP,NLopt, HDF5, JLD, Distributions, MixedModels, RCall, StatsBase, xCommon

function loadDF()
    cd("/media/u01/analytics/scoring/CDW5_792/")
    df_data = readtable("csv_final_cdw5_792.csv",header=false);
    df_h = readtable("Headers_cdw5.csv",header=false);
    names!(df_data, convert(Array{Symbol}, df_h[:x1]) )
end
df_in=loadDF()


