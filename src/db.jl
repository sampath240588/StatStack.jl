#using DataStructures, DataFrames, StatsFuns, GLM , JuMP,NLopt, StatStack, NLsolve, DBAPI, JDBC
#ENV["LD_LIBRARY_PATH"] = "/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.71-2.b15.el7_2.x86_64/jre/lib/amd64/:/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.71-2.b15.el7_2.x86_64/jre/lib/amd64/server/"
#JavaCall.addClassPath("/home/rmadmin/jdbc/ora/ojdbc7.jar")
#JavaCall.init(["-Duser.timezone=UTC","-Djava.class.path=/home/rmadmin/jdbc/ora/ojdbc7.jar"]);  #JDBC.init()

"""
ENV["LD_LIBRARY_PATH"] = "/usr/lib/jvm/java-1.7.0-openjdk-1.7.0.101.x86_64/jre/lib/amd64/:/usr/lib/jvm/java-1.7.0-openjdk-1.7.0.101.x86_64/jre/lib/amd64/server/"
JavaCall.addClassPath("/mapr/mapr04p/analytics0001/analytic_users/jpkg/v0.4/ojdbc7.jar")
JavaCall.init(["-Duser.timezone=UTC","-Djava.class.path=/mapr/mapr04p/analytics0001/analytic_users/jpkg/v0.4/ojdbc7.jar"]); 

/opt/jdk1.7.0_79/jre/lib/amd64/server/libjvm.so

scp -r rmadmin@10.63.36.7:g/StatStack ./
0
"""

function saveDBora(sdf::DataFrame)
    #conn = DriverManager.getConnection("jdbc:oracle:thin:PRSPC/prspc@//ex02-scan.ch3.prod.i.com:1521/SV01DWHD")
    conn = DriverManager.getConnection("jdbc:oracle:thin:WH_MEDIA_SYND_P1/Passw0rd@//ex04-scan1.ch3.prod.i.com:1521/SV01DWHP")
    stmt = createStatement(conn)
    #rs = executeQuery(stmt, "select count(*) cnt from PRSPC.WH_PARAMS")
    
    rout=Int64[]
    rs = executeQuery(stmt, "select max(SCORE_MODEL_RUN_ID)+1 as cnt from WH_MEDIA_SYND_P1.LIFT_SCORE")
    for r in rs  push!(rout,getInt(r, 1))  end
    rid=rout[1]
    
    
    sql="insert into WH_MEDIA_SYND_P1.LIFT_SCORE_MODEL_RUN (SCORE_MODEL_RUN_ID,SCORE_MODEL_NAME,SCORE_MODEL_VER, RUN_DATE,RUN_STATUS, "
    sql = sql*"VIEW_DIM_TYPE, OPT_TYPE,OPT_MESR_TYPE, RELS_CD, BATCH_ID, REGISTRATION_REQUEST_ID, REGISTRATION_ID, "
    sql = sql*"TM_AGG_PERD,START_WEEK,END_WEEK) VALUES ( "
    sql = sql*string(rid)*", "
    sql = sql*"NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,188765,188765,99,9999,9999) "
    executeQuery(stmt, sql)
    
    
    pref = "insert into WH_MEDIA_SYND_P1.LIFT_SCORE "
    pref=pref*"    ( SCORE_MODEL_RUN_ID,MODEL_DESC,MODEL,DEPNDNT_VAR,CNT_HH_EXPSD,UNADJ_AVG_EXPSD_HH_PRE,UNADJ_AVG_CNTRL_HH_PRE,UNADJ_AVG_EXPSD_HH_PST,"
    pref=pref*"      UNADJ_AVG_CNTRL_HH_PST,UNADJ_DOD_EFFCT,UNADJ_DIFF_EFFCT,ADJ_MEAN_EXPSD_GRP,ADJ_MEAN_CNTRL_GRP,ADJ_DOD_EFFCT,TWOTAIL_PVAL,ONETAIL_PVAL, "
    pref=pref*"      ABS_DIFF,DOL_DIFF,ONETAIL_80_PCT_INTRVL_UB,ONETAIL_80_PCT_INTRVL_LB,ONETAIL_90_PCT_INTRVL_UB,ONETAIL_90_PCT_INTRVL_LB, " 
    pref=pref*"  TWOTAIL_80_PCT_INTRVL_UB,TWOTAIL_80_PCT_INTRVL_LB,TWOTAIL_90_PCT_INTRVL_UB,TWOTAIL_90_PCT_INTRVL_LB,CNT_IMPRESNS,TWOTAIL_PVAL_TO_CAMPAIGN, "
    pref=pref*"      ONETAIL_PVAL_TO_CAMPAIGN,CNT_MODEL_HH,BATCH_ID "
    pref=pref*"    ) "
    pref=pref*"values " 
    
    
    
    function isnax(v::Any) isna(v) ? true: isnan(v) ? true : false end
    
    for i in 1:length(sdf[1])
        println("Number : ",i)
        r=AbstractString[]
        push!(r,"'"*sdf[i,:MODEL_DESC]*"', " )
        isnax(sdf[1,:Model]) ?  push!(r,"NULL, ") : push!(r,"'"*sdf[1,:Model]*"', ")
        #isnax(sdf[i,:dependent_variable]) ? push!(r,"NULL, ") : push!(r,"'"*sdf[i,:dependent_variable]*"', " )
        push!(r,"'"*sdf[i,:dependent_variable]*"', " )
        isnax(sdf[i,:CNT_EXPSD_HH]) ? push!(r,"NULL, ") : push!(r, string(sdf[i,:CNT_EXPSD_HH])*", "   )
        isnax(sdf[i,:UNADJ_AVG_EXPSD_HH_PRE]) ? push!(r,"NULL, ") : push!(r,string(sdf[i,:UNADJ_AVG_EXPSD_HH_PRE])*", ")
        isnax(sdf[i,:UNADJ_AVG_CNTRL_HH_PRE]) ? push!(r,"NULL, "): push!(r,string(sdf[i,:UNADJ_AVG_CNTRL_HH_PRE])*", ")
        isnax(sdf[i,:UNADJ_AVG_EXPSD_HH_PST]) ? push!(r,"NULL, "): push!(r,string(sdf[i,:UNADJ_AVG_EXPSD_HH_PST])*", ")
        isnax(sdf[i,:UNADJ_AVG_CNTRL_HH_PST]) ? push!(r,"NULL, "): push!(r,string(sdf[i,:UNADJ_AVG_CNTRL_HH_PST])*", ")
        isnax(sdf[i,:UNADJ_DOD_EFFCT]) ? push!(r,"NULL, "): push!(r,string(sdf[i,:UNADJ_DOD_EFFCT])*", ")
        isnax(sdf[i,:UNADJ_DIFF_EFFCT]) ? push!(r,"NULL, "): push!(r,string(sdf[i,:UNADJ_DIFF_EFFCT])*", ")
        isnax(sdf[i,:ADJ_MEAN_EXPSD_GRP]) ? push!(r,"NULL, "): push!(r,string(sdf[i,:ADJ_MEAN_EXPSD_GRP])*", ")
        isnax(sdf[i,:ADJ_MEAN_CNTRL_GRP]) ? push!(r,"NULL, "): push!(r,string(sdf[i,:ADJ_MEAN_CNTRL_GRP])*", ")
        isnax(sdf[i,:ADJ_DOD_EFFCT]) ? push!(r,"NULL, "): push!(r,string(sdf[i,:ADJ_DOD_EFFCT])*", ")
        isnax(sdf[i,:TWOTAIL_PVAL]) ? push!(r,"NULL, "): push!(r,string(sdf[i,:TWOTAIL_PVAL])*", ")
        isnax(sdf[i,:ONETAIL_PVAL]) ? push!(r,"NULL, "): push!(r,string(sdf[i,:ONETAIL_PVAL])*", ")
        isnax(sdf[i,:ABS_DIFF]) ? push!(r,"NULL, "): push!(r,string(sdf[i,:ABS_DIFF])*", ")
        isnax(sdf[i,:DOL_DIFF]) ? push!(r,"NULL, "): push!(r,string(sdf[i,:DOL_DIFF])*", ")
        isnax(sdf[i,:ONETAIL_80_PCT_INTRVL_UB]) ? push!(r,"NULL, "): push!(r,string(sdf[i,:ONETAIL_80_PCT_INTRVL_UB])*", ")
        isnax(sdf[i,:ONETAIL_80_PCT_INTRVL_LB]) ? push!(r,"NULL, "): push!(r,string(sdf[i,:ONETAIL_80_PCT_INTRVL_LB])*", ")
        isnax(sdf[i,:ONETAIL_90_PCT_INTRVL_UB]) ? push!(r,"NULL, "): push!(r,string(sdf[i,:ONETAIL_90_PCT_INTRVL_UB])*", ")
        isnax(sdf[i,:ONETAIL_90_PCT_INTRVL_LB]) ? push!(r,"NULL, "): push!(r,string(sdf[i,:ONETAIL_90_PCT_INTRVL_LB])*", ") 
        isnax(sdf[i,:TWOTAIL_80_PCT_INTRVL_UB]) ? push!(r,"NULL, "): push!(r,string(sdf[i,:TWOTAIL_80_PCT_INTRVL_UB])*", ")
        isnax(sdf[i,:TWOTAIL_80_PCT_INTRVL_LB]) ? push!(r,"NULL, "): push!(r,string(sdf[i,:TWOTAIL_80_PCT_INTRVL_LB])*", ")
        isnax(sdf[i,:TWOTAIL_90_PCT_INTRVL_UB]) ? push!(r,"NULL, "): push!(r,string(sdf[i,:TWOTAIL_90_PCT_INTRVL_UB])*", ")
        isnax(sdf[i,:TWOTAIL_90_PCT_INTRVL_LB]) ? push!(r,"NULL, "): push!(r,string(sdf[i,:TWOTAIL_90_PCT_INTRVL_LB])*", ")
        isnax(sdf[i,:CNT_IMPRESSIONS]) ? push!(r,"NULL, "): push!(r,string(sdf[i,:CNT_IMPRESSIONS])*", ")
        isnax(sdf[i,:TWOTAIL_PVAL_to_Campaign]) ? push!(r,"NULL, "): push!(r,string(sdf[i,:TWOTAIL_PVAL_to_Campaign])*", ")
        isnax(sdf[i,:ONETAIL_PVAL_to_Campaign]) ? push!(r,"NULL, "): push!(r,string(sdf[i,:ONETAIL_PVAL_to_Campaign])*", ")
        isnax(sdf[i,:CNT_Model_HH]) ? push!(r,"NULL, "): push!(r,string(sdf[i,:CNT_Model_HH])*", ")

        sql=pref*" ("*string(rid)*", "
        for x in 1:length(r)  sql=sql*r[x] end
        sql=sql*"0.0)"
        #println(rout)
        executeQuery(stmt, sql)
    end
    
      """  
        sql=pref*" ("*string(rid)*", "
        sql=sql*"'"*sdf[i,:MODEL_DESC]*"'"*", "
        sql=sql*"'"*sdf[i,:Model]*"'"*", "
        sql=sql*"'"*sdf[i,:DEPNDNT_VAR]*"'"*", "
        sql=sql+String(sdf[i,:CNT_HH_EXPSD])
        sql=sql+String(sdf[i,:UNADJ_AVG_EXPSD_HH_PRE])
        sql=sql+String(sdf[i,:UNADJ_AVG_CNTRL_HH_PRE])
        sql=sql+String(sdf[i,:UNADJ_AVG_EXPSD_HH_PST])
        sql=sql+String(sdf[i,:UNADJ_AVG_CNTRL_HH_PST])
        sql=sql+String(sdf[i,:UNADJ_DOD_EFFCT])
        sql=sql+String(sdf[i,:UNADJ_DIFF_EFFCT])
        sql=sql+String(sdf[i,:ADJ_MEAN_EXPSD_GRP])
        sql=sql+String(sdf[i,:ADJ_MEAN_CNTRL_GRP])
        sql=sql+String(sdf[i,:ADJ_DOD_EFFCT])
        sql=sql+String(sdf[i,:TWOTAIL_PVAL])
        sql=sql+String(sdf[i,:ONETAIL_PVAL])
        sql=sql+String(sdf[i,:ABS_DIFF])
        sql=sql+String(sdf[i,:DOL_DIFF])
        sql=sql+String(sdf[i,:ONETAIL_80_PCT_INTRVL_UB])
        sql=sql+String(sdf[i,:ONETAIL_80_PCT_INTRVL_LB])
        sql=sql+String(sdf[i,:ONETAIL_90_PCT_INTRVL_UB])
        sql=sql+String(sdf[i,:ONETAIL_90_PCT_INTRVL_LB]) 
        sql=sql+String(sdf[i,:TWOTAIL_80_PCT_INTRVL_UB])
        sql=sql+String(sdf[i,:TWOTAIL_80_PCT_INTRVL_LB])
        sql=sql+String(sdf[i,:TWOTAIL_90_PCT_INTRVL_UB])
        sql=sql+String(sdf[i,:TWOTAIL_90_PCT_INTRVL_LB])
        sql=sql+String(sdf[i,:CNT_IMPRESNS])
        sql=sql+String(sdf[i,:TWOTAIL_PVAL_TO_CAMPAIGN])
        sql=sql+String(sdf[i,:ONETAIL_PVAL_TO_CAMPAIGN])
        sql=sql+String(sdf[i,:CNT_MODEL_HH])
        sql=sql+"0.0"   #String(sdf[1,:BATCH_ID])  #BATCH_ID
    """

    #println(sql)
    #ji = JDBCRowIterator(rs); getInt(ji.rs, 1)
    #ji = JDBCRowIterator(rs); typeof(ji.get_methods[1](ji.rs, 1))
    #ji=JDBCRowIterator(rs); getInt(ji.get_methods[1](ji.rs, 1), 1)


    #rid=getInt(JDBCRowIterator(rs)[1], 1)
    #println(getInt(r, 1))
    #for r in rs  println(getInt(r, 1),  getString(r,"cnt"))  end
    
    
    #collect(or r in rs  getInt(r, 1)  end)[1]
    
    #execute!(csr, "insert into pi_table (pi_value) values (3.14);")
    #execute!(csr, "select * from my_table;")
    
    #csr = cursor(conn)
    #execute!(csr, "select count(*) cnt from PRSPC.WH_PARAMS")
    #rs = rows(csr)
    #for row in rs
    #    println(row[:cnt]) 
    #end
    close(conn)
end
#saveDBora(sdf)


