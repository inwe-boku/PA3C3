; Calculate the disaggregated precipitation based on the days when precipitation occurs.
; The total daily precipitation is 1, and will be scaled by the coefficients obtained from the 
; ratios of the precipitation to 1.
;
; 19.10.2012 Robert Goler

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

begin

 f1 = addfile("CNRM-tas-pr.nc","r")
 f2 = addfile("ICTP-tas-pr.nc","r")
 f3 = addfile("MPIM-tas-pr.nc","r")
 time = f1->time
 ntim = dimsizes(time)
 units = time@units

 fo1=addfile("CNRM-tas-pr-hourly.nc","c")
 fo2=addfile("ICTP-tas-pr-hourly.nc","c")
 fo3=addfile("MPIM-tas-pr-hourly.nc","c")
 filedimdef(fo1,"time",-1,True)
 filedimdef(fo2,"time",-1,True)
 filedimdef(fo3,"time",-1,True)

 pr_factor = new((/ntim,17,3/),float,9.96921e+36)
 pr_factor(:,:,0) = f1->prcp
 pr_factor(:,:,1) = f2->prcp
 pr_factor(:,:,2) = f3->prcp

 pr_ref_file = readAsciiTable("Output_new6/precip_dissag.dat",8,"float",0)

 model_array = (/"aladin","regcm3","remo"/)
 num_models = 3
 num_days = dimsizes(time)           ; no of days of daily data.

 year = pr_ref_file(:,0)
 num_time = dimsizes(year)

 month = pr_ref_file(:,1)
 day = pr_ref_file(:,2)
 hour = pr_ref_file(:,3)
 minute = pr_ref_file(:,4)
 second = minute
 second = 0.0

 time1 = cd_inv_calendar(floattointeger(year),floattointeger(month),floattointeger(day),floattointeger(hour),floattointeger(minute),floattointeger(second),units, 0)

 pr = pr_ref_file(:,5:7)
 pr_disagg = pr
 pr_disagg!0 = "time"
 pr_disagg&time=time1


 fmt = "1x,f5.0,1x,f3.0,1x,f3.0,1x,f3.0,1x,f3.0,3(1x,f10.3)"
 opt = True

 num_stations = 17
 data = asciiread("stationlist.txt",(/num_stations/),"string")
 riv = str_get_field(data,1," ")    ; river
 stn = str_get_field(data,2," ")    ; station name
 sta = str_get_field(data,3," ")    ; station abbreviation
 lat = stringtofloat(str_get_field(data,4," "))     ; latitude values
 lon = stringtofloat(str_get_field(data,5," "))     ; longitude values
 sht = stringtofloat(str_get_field(data,6," "))     ; longitude values

 print(riv+"           "+stn)

 do station = 0, num_stations - 1
   file_name = "Output_new6/"+sta(station)+".txt"
   do tm = 0, num_days - 1
     do model = 0, num_models - 1
      pr_disagg(tm * 24 : (tm + 1) * 24 - 1,model) = pr_factor(tm,station,model) * pr(tm * 24 : (tm + 1) * 24-1,model)
     end do
   end do

   do srch = 0, num_days*24 - 1
     do model = 0, num_models - 1
       if (pr_disagg(srch,model) .gt. 40.0) then
         print ("Station: " + stn(station) + " Model: " + model + " Disagg_index: " + srch + " pr: " + pr_disagg(srch,model))
       end if
     end do
   end do  
;-------------------------------------------------------------------------------------------
; Output data to ASCII file
 print (systemfunc ("date") + " Writing data to file")

   output_array = new((/num_days*24,8/),"float")

   output_array(:,0) = year
   output_array(:,1) = month
   output_array(:,2) = day
   output_array(:,3) = hour
   output_array(:,4) = minute
   output_array(:,5) = pr_disagg(:,0)
   output_array(:,6) = pr_disagg(:,1)
   output_array(:,7) = pr_disagg(:,2)

   ovar="pr_"+sta(station)

   fo1->$ovar$=pr_disagg(:,0)
   fo2->$ovar$=pr_disagg(:,1)
   fo3->$ovar$=pr_disagg(:,2)

   opt@title = ""
   opt@fout = file_name
   write_matrix(output_array(:,:),fmt,opt)
 
 end do         ; End station loop

end

