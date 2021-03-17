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

 f1a=addfile("CNRM-tas-pr.nc","r")
 f1b=addfile("CNRM-tasmin-tasmax.nc","r")
 f2a=addfile("ICTP-tas-pr.nc","r")
 f2b=addfile("ICTP-tasmin-tasmax.nc","r")
 f3a=addfile("MPIM-tas-pr.nc","r")
 f3b=addfile("MPIM-tasmin-tasmax.nc","r")

 model_array = (/"aladin","regcm3","remo"/)
 num_models = dimsizes(model_array)
 
 time = f1a->time
 vtmp = f1a->tas
 dims = dimsizes(vtmp)

 num_days = dims(0)
 num_stations = dims(1)

 print("num_days and num_stations are "+num_days+" "+num_stations)

 tas_all    = new((/num_days,num_stations,num_models/),float,vtmp@_FillValue)
 tasmin_all = new((/num_days,num_stations,num_models/),float,vtmp@_FillValue)
 tasmax_all = new((/num_days,num_stations,num_models/),float,vtmp@_FillValue)

 tas_all(:,:,0)    = f1a->tas
 tasmin_all(:,:,0) = f1b->tasmin
 tasmax_all(:,:,0) = f1b->tasmax
 tas_all(:,:,1)    = f2a->tas
 tasmin_all(:,:,1) = f2b->tasmin
 tasmax_all(:,:,1) = f2b->tasmax
 tas_all(:,:,2)    = f3a->tas
 tasmin_all(:,:,2) = f3b->tasmin
 tasmax_all(:,:,2) = f3b->tasmax
 utime         = ut_calendar(time,-5)
        years  = utime(:,0)
        months = utime(:,1)
          days = utime(:,2)


 file_name_list    = asciiread("station_names_list.txt",-1,"string")

 Pi = 4.0 * atan(1.0)

 fmt = "1x,f5.0,1x,f3.0,1x,f3.0,1x,f3.0,1x,f3.0,f10.3,f10.3,f10.3,f10.3"
 opt = True

 do station = 0, num_stations - 1
   print (systemfunc ("date") + "  Station: " + file_name_list(station))
   file_name = "Output_new/" + model_array + "_" + file_name_list(station)

   tas = tas_all(:,station,:)
   tasmin = tasmin_all(:,station,:)
   tasmax = tasmax_all(:,station,:)

;   if (station .eq. 0) then   

     year = new((/num_days,num_models/),"integer")
     month = new((/num_days,num_models/),"integer")
     day = new((/num_days,num_models/),"integer")
     do model = 0, num_models - 1
       year(:,model) = years
       month(:,model) = months
       day(:,model) = days
     end do
        
;-------------------------------------------------------------------------------------------
; Open sunrise data file - Salzburg is used for the reference (corrected for summer time change)
; Note that the times in this file are LOCAL WINTER TIME. Subtract 1 hour from all days to get
; the time in UTC.
     sunrise_raw = asciiread("sunrise_graz.dat",(/365,4/),"float")
     sunrise = new((/366/),integer)

     sunrise(:364) = round((sunrise_raw(:,2) - 1 + sunrise_raw(:,3)/60)*ti_inhour,3)
     sunrise(365) = sunrise(364)                                       ; take care of leap year

; Copy sunrise for all days
     time_dawn = new((/num_days,num_models/),integer)

     do model = 0, num_models - 1
       do dy = 0, num_days - 1 
         dayofyear = day_of_year(year(dy,0),month(dy,0),day(dy,0))
         time_dawn(dy,model) = sunrise(dayofyear-1)
       end do
     end do
     
; Setting time of peak temperature
     time_peak = new((/num_days,num_models/),integer)
     time_steps = 24 * ti_inhour                     ; Number of time steps per day
     time_peak = (15 - 1 ) * ti_inhour                       ; Index UTC time of peak temperature - modified later on
     t_peak1 = new((/num_models/),integer)
     t_peak2 = t_peak1
     t_peak3 = t_peak1
     t_dawn1 = t_peak1
     t_dawn2 = t_peak1
     t_dawn3 = t_peak1
     
     a = time_steps - 1

     tas_xmins1 = new((/time_steps,num_models/),float)
     tas_xmins2 = new((/time_steps,num_models/),float)
     tas_xmins3 = new((/time_steps,num_models/),float)
     tas_xmins = new((/num_days*time_steps,num_models/),float)
     tasmax_xmins = tas_xmins
     tasmin_xmins = tas_xmins
     tasmean_xmins = tas_xmins

;-------------------------------------------------------------------------------------------
; Time definitions
     time_year = new((/time_steps*num_days,num_models/),integer)
     time_month = new((/time_steps*num_days,num_models/),integer)
     time_day = new((/time_steps*num_days,num_models/),integer)
     time_hour = new((/time_steps*num_days,num_models/),integer)
     time_min = new((/time_steps*num_days,num_models/),integer)

     count = 0
     do dy = 0, num_days - 1
       dy_start = dy * time_steps
       dy_end = (dy + 1) * time_steps - 1
       count = dy * time_steps
       do i = dy_start, dy_end
         time_day(i,:) = day(dy,:)
	 time_month(i,:) = month(dy,:)
	 time_year(i,:) = year(dy,:)
       end do
     end do

     do model = 0, num_models - 1
       do i = 0, time_steps - 1
         time_hour(i::time_steps, model) = floattoint(i/ti_inhour)
       end do
       do i = 0, ti_inhour - 1
         time_min(i::ti_inhour,model) = time_interval * i
       end do
     end do
;   end if

;-------------------------------------------------------------------------------------------
; Set daily max and min to x minute intervals for output
     
   t_count = 0
   do dy = 0, num_days - 1
     do i = 0, time_steps - 1
       tasmax_xmins(t_count, :) = tasmax(dy,:)
       tasmin_xmins(t_count, :) = tasmin(dy,:)
       tasmean_xmins(t_count,:) = tas(dy,:)
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

   do model = 0, num_models - 1
     do dy = 0, (num_days - 1) - 1 
       if (tasmax(dy,model) .lt. tasmin(dy+1,model)) then
         time_peak(dy,model) = time_steps - 1
         time_dawn(dy+1,model) = 0
         maxmin_avg = (tasmax(dy,model) + tasmin(dy+1,model))/2
         tasmax(dy,model) = maxmin_avg
         tasmin(dy+1,model) = maxmin_avg
       end if
     end do
   end do

;-------------------------------------------------------------------------------------------
;-------------------------------------------------------------------------------------------
; Calculate temperature at 15 minute intervals

 print (systemfunc ("date") + " Calculating temperature in " + time_interval + " minute intervals")

   modify_morning = (/"n","n","n"/)

   time_total = 0
   do dy = 0, num_days - 1
;     print ("print1: " + modify_morning(0) + " " + modify_morning(1) + " " + modify_morning(2))      

     do j = 0, time_steps - 1                     ; index to count time steps in a day

; First period of day
       if (month(dy,0) .eq. 1 .and. day(dy,0) .eq. 1) then
         if (dy .eq. 0) then
           tas_m1 = (tasmax(dy,:) + tasmin(dy,:)) / 2
           tas_a1 = (tasmax(dy,:) - tasmin(dy,:)) / 2  
           t_peak1 = time_peak(dy,:)       
         else
           tas_m1 = (tasmax(dy-1,:) + tasmin(dy,:)) / 2
           tas_a1 = (tasmax(dy-1,:) - tasmin(dy,:)) / 2  
           t_peak1 = time_peak(dy-1,:)        
         end if
       else
         tas_m1 = (tasmax(dy-1,:) + tasmin(dy,:)) / 2
         tas_a1 = (tasmax(dy-1,:) - tasmin(dy,:)) / 2  
         t_peak1 = time_peak(dy-1,:)     
       end if
       t_dawn1 = time_dawn(dy,:)     

; Second period of day
       tas_m2 = (tasmax(dy,:) + tasmin(dy,:)) / 2
       tas_a2 = (tasmax(dy,:) - tasmin(dy,:)) / 2
       t_peak2 = time_peak(dy,:)    
       t_dawn2 = time_dawn(dy,:)    


; Third period of day
       if (month(dy,0) .eq. 12 .and. day(dy,0) .eq. 31) then
         if (dy .eq. num_days - 1) then
           tas_m3 = (tasmax(dy,:) + tasmin(dy,:)) / 2
           tas_a3 = (tasmax(dy,:) - tasmin(dy,:)) / 2   
           t_dawn3 = time_dawn(dy,:)     
         else
           tas_m3 = (tasmax(dy,:) + tasmin(dy+1,:)) / 2
           tas_a3 = (tasmax(dy,:) - tasmin(dy+1,:)) / 2
           t_dawn3 = time_dawn(dy+1,:)   
         end if  
       else
         tas_m3 = (tasmax(dy,:) + tasmin(dy+1,:)) / 2
         tas_a3 = (tasmax(dy,:) - tasmin(dy+1,:)) / 2
         t_dawn3 = time_dawn(dy+1,:)   
       end if       
       t_peak3 = time_peak(dy,:)    
  

       tas_xmins1(j,:) = tas_m1 - tas_a1 * cos(Pi * (t_dawn1 - j)/(time_steps + t_dawn1 - t_peak1)) 

          do model = 0, num_models - 1
             if (j .eq. t_dawn2(model)) then
;             print ("print2: " + dy + " " + model + " " + modify_morning(model))      
             if (modify_morning(model) .eq. "y") then
                if (t_dawn1(model) .ne. 0) then
                   ; From t_midnight to t_dawn
                   tasmax_test = (tasmin(dy-1,model)-tasmin(dy,model)/2.*(1+cos(Pi * (t_dawn1(model)-0)/(time_steps + t_dawn1(model) - t_peak1(model)))))/(0.5-0.5*cos(Pi * (t_dawn1(model) - 0)/(time_steps + t_dawn1(model) - t_peak1(model))))
                else
                   tasmax_test = tasmax(dy-1,model)
                end if
                tas_m3_2 = (tasmax_test + tasmin(dy,model))/2
                tas_a3_2 = (tasmax_test - tasmin(dy,model))/2
                do jj = 0, t_dawn2(model) - 1
                   tas_xmins1(jj,model) = tas_m3_2 - tas_a3_2 * cos(Pi * (t_dawn1(model) - jj)/(time_steps + t_dawn1(model) - t_peak1(model))) 
                end do
;                print (model + " " + tasmin(dy-1,model) + " " + tas_xmins1(0,model) + " " + tas_xmins1(1,model)+ " " + tas_xmins1(2,model) + " " + tas_xmins1(3,model)) 
             end if
             modify_morning(model) = "n"
             end if
          end do
       tas_xmins2(j,:) = tas_m2 + tas_a2 * cos(Pi * (t_peak2 - j)/(t_peak2 - t_dawn2)) 
       tas_xmins3(j,:) = tas_m3 - tas_a3 * cos(Pi * (time_steps + t_dawn3 - j)/(time_steps + t_dawn3 - t_peak2))
   
; Construct the temperature array from the three curves at the end of each day
       if (j .eq. a) then

; Check that minimum temperature is not exceeded on the transition to the next day's minimum
         do model = 0, num_models - 1
	    if (tas_xmins3(j,model) .lt. tasmin(dy,model)) then
	       ; From t_peak to t_midnight
               tasmin_test = (tasmin(dy,model) - tasmax(dy,model)/2.*(1-cos(Pi * (time_steps + t_dawn3(model) - time_steps)/(time_steps + t_dawn3(model) - t_peak2(model)))))/(0.5+0.5*cos(Pi * (time_steps + t_dawn3(model) - time_steps)/(time_steps + t_dawn3(model) - t_peak2(model))))
               tas_m3_1 = (tasmax(dy,model) + tasmin_test)/2
               tas_a3_1 = (tasmax(dy,model) - tasmin_test)/2
	       do jj = t_peak2(model), time_steps - 1
                  tas_xmins3(jj,model) = tas_m3_1 - tas_a3_1*cos(Pi * (time_steps + t_dawn3(model) - jj)/(time_steps + t_dawn3(model) - t_peak2(model)))
               end do
               modify_morning(model) = "y" 
;               print ("print3: " + dy + " " + model + " " + modify_morning(model))      
            end if
	    
	 end do
;---         
         do model = 0, num_models - 1
;            print (model + " " + time_total + " " + a + " " + t_dawn2(model) + " " + tas_xmins1(0,model) + " " + tas_xmins1(1,model))
            tas_xmins(time_total-a:time_total-a+ t_dawn2(model),model) = tas_xmins1(0:t_dawn2(model),model)
            tas_xmins(time_total-a+ t_dawn2(model):time_total-a+t_peak2(model),model) = tas_xmins2(t_dawn2(model):t_peak2(model),model)
            tas_xmins(time_total-a+t_peak2(model):time_total,model) = tas_xmins3(t_peak2(model):,model)
	 end do
;         print ("print4: " + modify_morning(0) + " " + modify_morning(1) + " " + modify_morning(2))      
       end if
       time_total = time_total + 1
     end do
   end do

;-------------------------------------------------------------------------------------------
; Output data to ASCII file
   ;print (systemfunc ("date") + " Writing data to file")

   output_array = new((/time_steps*num_days,num_models,9/),"float")
   
   output_array(:,:,0) = tofloat(time_year(:,:))
   output_array(:,:,1) = time_month(:,:)
   output_array(:,:,2) = time_day(:,:)
   output_array(:,:,3) = time_hour(:,:)
   output_array(:,:,4) = time_min(:,:)
   output_array(:,:,5) = tasmin_xmins(:,:)
   output_array(:,:,6) = tasmean_xmins(:,:)
   output_array(:,:,7) = tasmax_xmins(:,:)
   output_array(:,:,8) = tas_xmins(:,:)

;   opt@title = ""
   opt@fout = file_name(0)
   write_matrix(output_array(:,0,:),fmt,opt)
   opt@fout = file_name(1)
   write_matrix(output_array(:,1,:),fmt,opt)
   opt@fout = file_name(2)
   write_matrix(output_array(:,2,:),fmt,opt)

   delete ([/tas,tasmin,tasmax/])
   delete ([/time_dawn,time_peak,t_peak1,t_peak2,t_peak3,t_dawn1,t_dawn2,t_dawn3/])
   delete ([/tas_a1,tas_a2,tas_a3,tas_m1,tas_m2,tas_m3/])
   delete ([/tas_xmins1,tas_xmins2,tas_xmins3,tas_xmins/])

 end do         ; End station loop

end

