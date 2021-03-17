; Generate a 15 minute dataset from maximum and minimum temperature data.
; The method used here is based on that from the "European solar radation atlas"
; where the temperature profile is described using three piecewise cosine functions.
; Added a correction for when the minimum temperature of the following day is much colder
; than that on the actual day, as the temperature would go below the assigned minimum 
; temperature

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

begin

 time_interval = 60                  ; time interval in minutes
 ti_inhour = 60/time_interval

 f1a=addfile("CNRM-tas-pr-Bruck-Hundsmuehlbach.nc","r")
 f1b=addfile("CNRM-tasmin-tasmax-Bruck-Hundsmuehlbach.nc","r")
 
 time = f1a->time
 ndays = dimsizes(time)
 
 utime  = ut_calendar(time,-5)
 year   = utime(:,0)
 month  = utime(:,1)
 day    = utime(:,2)
 
 dayofyear = day_of_year(year,month,day)
 
 tas = new((/ndays,2,2/),float)
 tas!0 = "time"
 tas!1 = "y"
 tas!2 = "x"
 tas&time=time
 tasmin = tas
 tasmax = tas
 
 nys   = 2
 nxs   = 2

 print("ndays and ny and nx  are "+ndays+"   "+nys+" and   "+nxs)

 tas0    = f1a->tas
 tasmin0 = f1b->tasmin
 tasmax0 = f1b->tasmax
 
 tas    = conform(tas,tas0,(/0,1/))
 tasmin = conform(tasmin,tasmin0,(/0,1/))
 tasmax = conform(tasmax,tasmax0,(/0,1/))

 Pi = 4.0 * atan(1.0)

;-------------------------------------------------------------------------------------------
; Open sunrise data file - Salzburg is used for the reference (corrected for summer time change)
; Note that the times in this file are LOCAL WINTER TIME. Subtract 1 hour from all days to get
; the time in UTC.
     sunrise_raw = asciiread("sunrise_graz.dat",(/365,4/),"float")
     sunrise = new((/366/),integer)

     sunrise(:364) = round((sunrise_raw(:,2) - 1 + sunrise_raw(:,3)/60)*ti_inhour,3)
     sunrise(365) = sunrise(364)                                       ; take care of leap year

; Copy sunrise for all days
     time_dawn = new((/ndays/),integer)
      do dy = 0, ndays - 1 
         time_dawn(dy) = sunrise(dayofyear(dy)-1)
       end do
     
; Setting time of peak temperature
     time_peak  = new((/ndays/),integer)
     time_steps = 24 * ti_inhour                             ; Number of time steps per day
     time_peak  = (15 - 1 ) * ti_inhour                       ; Index UTC time of peak temperature - modified later on
     t_peak1 = new(1,integer)
     t_peak2 = t_peak1
     t_dawn1 = t_peak1
     t_dawn2 = t_peak1
     t_dawn3 = t_peak1
     
     tlast = time_steps - 1

     tas_xmins1 = new((/time_steps,nys,nxs/),float)
     tas_xmins2 = new((/time_steps,nys,nxs/),float)
     tas_xmins3 = new((/time_steps,nys,nxs/),float)
     tas_xmins = new((/ndays*time_steps,nys,nxs/),float)        ; will contain disaggregated temperature
     tasmax_xmins  = (/tas_xmins/)                              ; will contain original tasmax conformed to disaggregated time interval
     tasmin_xmins  = (/tas_xmins/)                              ; will contain original tasmax conformed to disaggregated time interval
     tasmean_xmins = (/tas_xmins/)                              ; will contain original tasmax conformed to disaggregated time interval

;-------------------------------------------------------------------------------------------
; Time definitions
     time_year = new(time_steps*ndays,integer)
     time_month = new(time_steps*ndays,integer)
     time_day = new(time_steps*ndays,integer)
     time_hour = new(time_steps*ndays,integer)

     count = 0
     do dy = 0, ndays - 1
       dy_start = dy * time_steps
       dy_end = (dy + 1) * time_steps - 1
       count = dy * time_steps
       do i = dy_start, dy_end
         time_day(i) = day(dy)
	 time_month(i) = month(dy)
	 time_year(i) = year(dy)
       end do
     end do

     do i = 0, time_steps - 1
       time_hour(i::time_steps) = floattoint(i/ti_inhour)
     end do

;   end if


;-------------------------------------------------------------------------------------------
; Set daily max and min to x minute intervals for output

do y=0,nys-1
 do x=0,nxs-1


   t_count = 0
   do dy = 0, ndays - 1
     do i = 0, time_steps - 1
       tasmax_xmins(t_count,y,x) = tasmax(dy,y,x)
       tasmin_xmins(t_count,y,x) = tasmin(dy,y,x)
       tasmean_xmins(t_count,y,x) = tas(dy,y,x)
       t_count = t_count + 1
     end do
   end do
;=============================================================================================
; Define time of peak temperature - only important if the minimum of the following day is
; greater than the maximum of the current day.
; For this case the time of maximum is at the last time step in the day (e.g. 23:45), 
; and the time of the minimum of the following day is at 00:00. 
; Note that to avoid a jump in the temperature from max to min across these times, 
; the average of the maximum and minimum temperatures are used.

   print (systemfunc ("date") + " Defining peak temperature time")

 
   
        do dy = 0, (ndays - 1) - 1
          if (tasmax(dy,y,x) .lt. tasmin(dy+1,y,x)) then
            time_peak(dy) = time_steps - 1
            time_dawn(dy+1) = 0
            maxmin_avg = (tasmax(dy,y,x) + tasmin(dy+1,y,x))/2
            tasmax(dy,y,x) = maxmin_avg
            tasmin(dy+1,y,x) = maxmin_avg           
          end if
        end do

;-------------------------------------------------------------------------------------------
;-------------------------------------------------------------------------------------------
; Calculate temperature at 15 minute intervals

   print (systemfunc ("date") + " Calculating temperature in " + time_interval + " minute intervals")

   modify_morning = "n"

   time_total = 0
   do dy = 0, ndays - 1
;     print ("print1: " + modify_morning(0) + " " + modify_morning(1) + " " + modify_morning(2))      

     do j = 0, time_steps - 1                     ; index to count time steps in a day

; First period of day
       if (month(dy) .eq. 1 .and. day(dy) .eq. 1) then
         if (dy .eq. 0) then
           tas_m1 = (tasmax(dy,y,x) + tasmin(dy,y,x)) / 2
           tas_a1 = (tasmax(dy,y,x) - tasmin(dy,y,x)) / 2  
           t_peak1 = time_peak(dy)       
         else
           tas_m1 = (tasmax(dy-1,y,x) + tasmin(dy,y,x)) / 2
           tas_a1 = (tasmax(dy-1,y,x) - tasmin(dy,y,x)) / 2  
           t_peak1 = time_peak(dy-1)        
         end if
       else
         tas_m1 = (tasmax(dy-1,y,x) + tasmin(dy,y,x)) / 2
         tas_a1 = (tasmax(dy-1,y,x) - tasmin(dy,y,x)) / 2  
         t_peak1 = time_peak(dy-1)     
       end if
       t_dawn1 = time_dawn(dy)     

; Second period of day
       tas_m2 = (tasmax(dy,y,x) + tasmin(dy,y,x)) / 2
       tas_a2 = (tasmax(dy,y,x) - tasmin(dy,y,x)) / 2
       t_peak2 = time_peak(dy)
       t_dawn2 = time_dawn(dy)

; Third period of day
       if (month(dy) .eq. 12 .and. day(dy) .eq. 31) then
         if (dy .eq. ndays - 1) then
           tas_m3 = (tasmax(dy,y,x) + tasmin(dy,y,x)) / 2
           tas_a3 = (tasmax(dy,y,x) - tasmin(dy,y,x)) / 2   
           t_dawn3 = time_dawn(dy)   
         else
           tas_m3 = (tasmax(dy,y,x) + tasmin(dy+1,y,x)) / 2
           tas_a3 = (tasmax(dy,y,x) - tasmin(dy+1,y,x)) / 2
           t_dawn3 = time_dawn(dy+1)
         end if  
       else
         tas_m3 = (tasmax(dy,y,x) + tasmin(dy+1,y,x)) / 2
         tas_a3 = (tasmax(dy,y,x) - tasmin(dy+1,y,x)) / 2
         t_dawn3 = time_dawn(dy+1)
       end if       

       tas_xmins1(j,y,x) = tas_m1 - tas_a1 * cos(Pi * (t_dawn1 - j)/(time_steps + t_dawn1 - t_peak1)) 


             if (j .eq. t_dawn2) then
;             print ("print2: " + dy +"   "+ modify_morning)      
             if (modify_morning .eq. "y") then
                if (t_dawn1 .ne. 0) then
                   ; From t_midnight to t_dawn
                   tasmax_test = (tasmin(dy-1,y,x)-tasmin(dy,y,x)/2.*(1+cos(Pi * (t_dawn1-0)/(time_steps + t_dawn1 - t_peak1))))/(0.5-0.5*cos(Pi * (t_dawn1 - 0)/(time_steps + t_dawn1 - t_peak1)))
                else
                   tasmax_test = tasmax(dy-1,y,x)
                end if
                tas_m3_2 = (tasmax_test + tasmin(dy,y,x))/2
                tas_a3_2 = (tasmax_test - tasmin(dy,y,x))/2
                do jj = 0, t_dawn2 - 1
                   tas_xmins1(jj,y,x) = tas_m3_2 - tas_a3_2 * cos(Pi * (t_dawn1 - jj)/(time_steps + t_dawn1 - t_peak1)) 
                end do
             end if
             modify_morning = "n"
             end if

       tas_xmins2(j,y,x) = tas_m2 + tas_a2 * cos(Pi * (t_peak2 - j)/(t_peak2 - t_dawn2)) 
       tas_xmins3(j,y,x) = tas_m3 - tas_a3 * cos(Pi * (time_steps + t_dawn3 - j)/(time_steps + t_dawn3 - t_peak2))
   
; Construct the temperature array from the three curves at the end of each day
       if (j .eq. tlast) then

; Check that minimum temperature is not exceeded on the transition to the next day's minimum

          minval = tas_xmins3(j,y,x)-tasmin(dy,y,x)
	  if (.not.ismissing(minval) .and. minval .lt. 0.0) then
            ; From t_peak to t_midnight
            tasmin_test = (tasmin(dy,y,x) - tasmax(dy,y,x)/2.*(1-cos(Pi * (time_steps + t_dawn3 - time_steps)/(time_steps + t_dawn3 - t_peak2))))/(0.5+0.5*cos(Pi * (time_steps + t_dawn3 - time_steps)/(time_steps + t_dawn3 - t_peak2)))
            tas_m3_1 = (tasmax(dy,y,x) + tasmin_test)/2.0
            tas_a3_1 = (tasmax(dy,y,x) - tasmin_test)/2.0
	    do jj = t_peak2, time_steps - 1
              tas_xmins3(jj,y,x) = tas_m3_1 - tas_a3_1*cos(Pi * (time_steps + t_dawn3 - jj)/(time_steps + t_dawn3 - t_peak2))
            end do
            modify_morning = "y" 
;            print ("print3: " + dy + " " + modify_morning)      
          end if
    
	    

           ;print("  y and x are "+y+"    "+x)
           ;print (time_total + " " + tlast + " " + t_dawn2 + " " + tas_xmins1(0,y,x) + " " + tas_xmins1(1,y,x))
            tas_xmins(time_total-tlast:time_total-tlast+t_dawn2,y,x) = tas_xmins1(0:t_dawn2,y,x)
            tas_xmins(time_total-tlast+t_dawn2:time_total-tlast+t_peak2,y,x) = tas_xmins2(t_dawn2:t_peak2,y,x)
            tas_xmins(time_total-tlast+t_peak2:time_total,y,x) = tas_xmins3(t_peak2,y,x)
            ;print(t_dawn3+"  "+t_peak2+"  "+t_dawn2+"    are values")
;         print ("print4: " + modify_morning(0) + " " + modify_morning(1) + " " + modify_morning(2))      
       end if
       time_total = time_total + 1
     end do
   end do
   ;delete ([/tas,tasmin,tasmax/])
   ;delete ([/time_dawn,time_peak,t_peak1,t_peak2,t_dawn1,t_dawn2,t_dawn3/])
   delete ([/tas_a1,tas_a2,tas_a3,tas_m1,tas_m2,tas_m3/])
   ;delete ([/tas_xmins1,tas_xmins2,tas_xmins3,tas_xmins/])
   
 end do
end do


;-------------------------------------------------------------------------------------------
   

  
 tas_xmins!0 = "time"
 tas_xmins!1 = "y"
 tas_xmins!2 = "x"

 base_julian = greg2jul(1900,1,1,0)
 time1       = greg2jul(time_year,time_month,time_day,time_hour)-base_julian
 time1@units = "days since 1900-1-1 00:00:0.0"
 tas_xmins&time = time1
 tas_xmins&y = ispan(1,nys,1)
 tas_xmins&x  = ispan(1,nxs,1)
 copy_VarMeta(tas_xmins,tasmin_xmins)
 copy_VarMeta(tas_xmins,tasmean_xmins)
 copy_VarMeta(tas_xmins,tasmax_xmins)

 outfname ="02d-CNRM-tasmin-tasmean_tasmax_tas_1hourly_Bruck_new_2D_both_loops_fullloop.nc"
 system(" 'rm' -f " + outfname)
 fout=addfile(outfname,"c")
 filedimdef(fout,"time",-1,True)
 fout->tas_xmins=tas_xmins
 fout->tasmin_xmins=tasmin_xmins
 fout->tasmean_xmins=tasmean_xmins
 fout->tasmax_xmins=tasmax_xmins

end

