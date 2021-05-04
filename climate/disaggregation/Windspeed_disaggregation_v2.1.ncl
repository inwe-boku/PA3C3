; Generate a 15 minute dataset from mean daily wind data.
; The method used here is based on Equation (25) in Debele et al (2007) 
; "Accuracy evaluation of weather data generation and disaggregation methods at finer timescales".
; wss_15mins = wss_mean * [(-ln (rnd[0,1))]^0.3

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

begin

 ;nst_start = 0
 ;nst_end = 0
 
 time_interval = 15                  ; time interval in minutes
 ti_inhour = 60/time_interval
 tsteps = 24 * ti_inhour                     ; Number of time steps per day

 model_array = (/"aladin","regcm3","remo"/)
 nmods = 3

 f1a=addfile("CNRM-wspd-bcorr.nc","r")
 f2a=addfile("ICTP-wspd-bcorr.nc","r")
 f3a=addfile("MPIM-wspd-bcorr.nc","r")

 model_array = (/"aladin","regcm3","remo"/)
 nmods = dimsizes(model_array)
 
 time = f1a->time
 vtmp = f1a->wspd
 dims = dimsizes(vtmp)

 ndays = dims(0)
 nstat    = dims(1)

 print("ndays and nstat are "+ndays+" "+nstat)
 wss_all    = new((/ndays,nstat,nmods/),float,vtmp@_FillValue)

 wss_all(:,:,0)    = f1a->wspd
 wss_all(:,:,1)    = f2a->wspd
 wss_all(:,:,2)    = f3a->wspd
 utime             = ut_calendar(time,-5)
 years             = utime(:,0)
 months            = utime(:,1)
 days              = utime(:,2)

 file_name_list    = asciiread("station_names_list.txt",-1,"string")

 Pi = 4.0 * atan(1.0)

 fmt = "1x,f5.0,1x,f3.0,1x,f3.0,1x,f3.0,1x,f3.0,f10.3"
 opt = True

;-------------------------------------------------------------------------------------------
; Random number initialisation

  rnd_low = 0.0
  rnd_high = 1.0
  rnd_convert = (rnd_high - rnd_low) / 32766.0
  srand(123456789)
;-------------------------------------------------------------------------------------------
     year = new((/ndays,nmods/),"integer")
     month = new((/ndays,nmods/),"integer")
     day = new((/ndays,nmods/),"integer")
     do model = 0, nmods - 1
       year(:,model) = years
       month(:,model) = months
       day(:,model) = days
     end do

     wss_xmins = new((/ndays*tsteps,nstat,nmods/),"float")
     wss_mean_xmins = new((/ndays*tsteps,nmods/),"float")
;-------------------------------------------------------------------------------------------
; Time definitions
     time_year = new((/tsteps*ndays,nmods/),integer)
     time_month = new((/tsteps*ndays,nmods/),integer)
     time_day = new((/tsteps*ndays,nmods/),integer)
     time_hour = new((/tsteps*ndays,nmods/),integer)
     time_min = new((/tsteps*ndays,nmods/),integer)

     count = 0
     do dy = 0, ndays - 1
       dy_start = dy * tsteps
       dy_end = (dy + 1) * tsteps - 1
       count = dy * tsteps
       do i = dy_start, dy_end
         time_day(i,:) = day(dy,:)
	 time_month(i,:) = month(dy,:)
	 time_year(i,:) = year(dy,:)
;         wss_mean_xmins(i,:) = wss(dy,:)      ; Convert wss daily data to x minute intervals
       end do
     end do
    
     do model = 0, nmods - 1
       do i = 0, tsteps - 1
         time_hour(i::tsteps, model) = floattoint(i/ti_inhour)
       end do
       do i = 0, ti_inhour - 1
         time_min(i::ti_inhour,model) = time_interval * i
       end do
     end do
 
     rnd_num = new((/ndays*tsteps,nmods/),"float")

;-------------------------------------------------------------------------------------------
; Coefficients
  
     coefficients_month = asciiread("windspeed_v2.1_coeffs.dat",(/tsteps, 12/),"float")
     coefficients_day = new((/ndays,tsteps/),"float")
     coefficients = new((/ndays*tsteps,3/),"float")
   
     do dy = 0, ndays - 1
       coefficients_day(dy,:) = coefficients_month(:,month(dy,0)-1)
     end do

     do dy = 0, ndays - 1
       dy_start = dy * tsteps
       dy_end = (dy + 1) * tsteps - 1
;       do i = dy_start, dy_end
         coefficients(dy_start:dy_end,0) = coefficients_day(dy,:)      ; Convert coefficients daily data to x minute intervals
;       end do
     end do
     coefficients(:,1) = coefficients(:,0)
     coefficients(:,2) = coefficients(:,0)
     
;--------------------- START OF STATION LOOP ---------------


 do nst = nst_start, nst_end   ;          0, nstat - 1
   print (systemfunc ("date") + "  Station: " + file_name_list(nst))
   file_name = "Output/" + model_array + "_" + file_name_list(nst)
   wss = wss_all(:,nst,:)
;-------------------------------------------------------------------------------------------
; Random number generation
   do model = 0, nmods - 1
     do t = 0, ndays * tsteps - 1
       random_num = rnd_convert * rand()
       do while (random_num .eq. 0)
         random_num = rnd_convert * rand()
       end do
       rnd_num(t,model) = random_num
     end do
   end do
 
   do dy = 0, ndays - 1
     dy_start = dy * tsteps
     dy_end = (dy + 1) * tsteps - 1
     do i = dy_start, dy_end
       wss_mean_xmins(i,:) = wss(dy,:)      ; Convert wss daily data to x minute intervals
     end do
   end do
 
;=============================================================================================
; Calculate wind speed at x minute intervals

 print (systemfunc ("date") + " Calculating wind speed in " + time_interval + " minute intervals")

   wss_xmins(:,nst,:) = wss_mean_xmins(:,:) * (-1.0 * log(rnd_num(:,:)))^0.3 * coefficients(:,:)
;   print (rnd_num(0:10,1) + " " + (-1.0 * log(rnd_num(0:10,1)))^0.3 + " " + wss_mean_xmins(0:10,1))
;-------------------------------------------------------------------------------------------
; Output data to ASCII file
   print (systemfunc ("date") + " Writing data to file")

   output_array = new((/tsteps*ndays,nmods,6/),"float")
   
   output_array(:,:,0) = tofloat(time_year(:,:))
   output_array(:,:,1) = time_month(:,:)
   output_array(:,:,2) = time_day(:,:)
   output_array(:,:,3) = time_hour(:,:)
   output_array(:,:,4) = time_min(:,:)
   output_array(:,:,5) = wss_xmins(:,nst,:)

;   opt@title = ""
   opt@fout = file_name(0)
   write_matrix(output_array(:,0,:),fmt,opt)
   opt@fout = file_name(1)
   write_matrix(output_array(:,1,:),fmt,opt)
   opt@fout = file_name(2)
   write_matrix(output_array(:,2,:),fmt,opt)
 
 end do         ; End station loop
 
 wss_xmins!0 = "time"
 wss_xmins!1 = "stations"
 wss_xmins!2 = "models"

 units  = "minutes since 1950-01-01 00:00:00"
 seconds    = time_min(:,0)
 seconds    = 0

 time1   = cd_inv_calendar(time_year(:,0),time_month(:,0),time_day(:,0),time_hour(:,0),time_min(:,0),seconds,units, 0)
 time1!0 = "time"
 time1&time = time1

 wss_xmins&time = time1
 wss_xmins&stations = ispan(1,nstat,1)
 wss_xmins&models  = ispan(1,nmods,1)

 outfname = "CNRM-ICTP-MPIM-wss_15mins_"+nst_start+".nc"
 system(" 'rm' -f " + outfname)
 fout=addfile(outfname,"c")
 filedimdef(fout,"time",-1,True)
 fout->wss_xmins=wss_xmins

end
