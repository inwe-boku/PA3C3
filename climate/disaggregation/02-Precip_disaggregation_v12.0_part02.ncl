; Calculate the disaggregated precipitation based on the days when precipitation occurs.
; The total daily precipitation is 1, and will be scaled by the coefficients obtained from the 
; ratios of the precipitation to 1.
; Alternative to precipitation_v11.1.ncl to reduce the number of time steps with precipitation below 0.1 mm
; 31.10.2012 Robert Goler

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

begin

 precip_day_file = readAsciiTable("precip_day.dat",6,"float",0)

 model_array = (/"aladin","regcm3","remo"/)
 num_models = 3

 year = tointeger(precip_day_file(:,0))
 num_days = dimsizes(year)

 month = tointeger(precip_day_file(:,1))
 day = tointeger(precip_day_file(:,2))

 pr_ref = precip_day_file(:,3:5)

 fmt = "1x,f5.0,1x,f3.0,1x,f3.0,1x,f3.0,1x,f3.0,3(1x,f11.8)"
 opt = True

 precip_upplimit_60mins = 40.0       ; Precipitation in 15 minutes should not exceed twice this amount - 40mm/hr
 precip_lowlimit_60mins = 0.28      ; Precipitation in 15 minutes should be at least this amount - 0.28mm/hr
                                   ; based on maximum precipitation.
; Define precipitation based on precipitation day with daily total precipitation = 1

 time_interval = 60
 ti_inhour = 60/time_interval

 time_steps = 24 * ti_inhour                     ; Number of time steps per day
 a = time_steps - 1

 pr_new = new((/num_days,time_steps,num_models/),float)
 pr_new = 0.0 

;-------------------------------------------------------------------------------------------
; Time definitions
   time_year = new((/time_steps*num_days/),integer)
   time_month = new((/time_steps*num_days/),integer)
   time_day = new((/time_steps*num_days/),integer)
   time_hour = new((/time_steps*num_days/),integer)
   time_min = new((/time_steps*num_days/),integer)

   count = 0
   do dy = 0, num_days - 1
     dy_start = dy * time_steps
     dy_end = (dy + 1) * time_steps - 1
     count = dy * time_steps
     do i = dy_start, dy_end
       time_day(i) = tointeger(day(dy))
       time_month(i) = tointeger(month(dy))
       time_year(i) = tointeger(year(dy))
     end do
   end do

   do i = 0, time_steps - 1
     time_hour(i::time_steps) = (i/ti_inhour)
   end do
   do i = 0, ti_inhour - 1
     time_min(i::ti_inhour) = time_interval * i
   end do
 
;-------------------------------------------------------------------------------------------
; Precipitation statistics

 wet_period_start_file =  readAsciiTable("Statistics_11101/time_precip_start_1hr.dat",17,"float",1)
 wet_period_duration_file = readAsciiTable("Statistics_11101/wetperiods_duration_hourly.dat",17,"float",1)
 wet_period_number_file = readAsciiTable("Statistics_11101/wetperiods_number_hourly.dat",17,"float",1)
 wet_period_precip_file = readAsciiTable("Statistics_11101/wetperiods_precip_hourly.dat",17,"float",1)
 precip_intensity_file = readAsciiTable("Statistics_11101/precip_intensity_hourly.dat",17,"float",1)
  
 wet_period_start = new((/12,24/),float)   ; hour of precip start
 wet_period_duration = new((/12,23/),float)  ; number of wet periods with duration in 60 min steps
 wet_period_number = new((/12,28/),float)  ; number of wet periods 
 wet_period_precip = new((/12,1000/),float)   ; amount of precip per wet period
 precip_intensity = new((/12,298/),float)   ; amount of precip per wet period

 do mn = 0, 11
   wet_period_start(mn,0:23) = wet_period_start_file(0:23,mn+1) + fspan(0,0.0001,24)      ; to make arrays monotonic
   wet_period_duration(mn,0:22) = wet_period_duration_file(0:22,mn+1) + fspan(0,0.0001,23)
   wet_period_number(mn,0:27) = wet_period_number_file(0:27,mn+1) + fspan(0,0.0001,28) 
   wet_period_precip(mn,0:999) = wet_period_precip_file(0:999,mn+1) + fspan(0,0.001,1000)
   precip_intensity(mn,0:297) = precip_intensity_file(0:297,mn+1) + fspan(0,0.0001,298)
 end do

;-------------------------------------------------------------------------------------------
; Sort through daily precip

; Produce random numbers
 srand(123)
  
 do model = 0, num_models - 1
 
   do dy = 0, num_days - 1

     if (pr_ref(dy,model) .ne. 0) then

;--
; Define number of wet periods in the day - based on statistical distribution from station

       rand_numwetperiods = (1.2 - 0) / 32766.0 * rand()                      ; 1.4 used to increase number of periods per day
       numwetperiods = closest_val(rand_numwetperiods,wet_period_number(month(dy)-1,:)) + 1
;       if (pr_ref(dy,model) .ne. 0 .and. numwetperiods .eq. 0) then
;          print ("Problem with numwetperiods: model " + model + "  day " + dy + "  pr_ref " + pr_ref(dy,model) + "  printensity " + printensity)
;       end if
       if (numwetperiods .eq. 0) then
         numwetperiods = 1
       end if
;-- 

       fallen_precip1 = 0.0
       do wetperiod = 0, numwetperiods
         ;--
         ; Define amount of precip in period - done randomly
         if (wetperiod .ne. numwetperiods) then
            rand_period_precip = ((pr_ref(dy,model) - fallen_precip1) - 0) / 32766.0 * rand()     ;Using the maximum precipitation 
            period_precip = rand_period_precip    ; closest_val(rand_period_precip,wet_period_precip(month(dy)-1,:)) + 1
            fallen_precip1 = fallen_precip1 + period_precip
         else
            period_precip = pr_ref(dy,model) - fallen_precip1
         end if

;         if (period_precip .eq. 0) then 
;           period_precip = 1
;         end if
         ;--
         ; Define starting time of period - based on statistical distribution from station

         rand_period_start_hr = (1 - 0) / 32766.0 * rand()               ; start of period - hour
         period_start_hr = closest_val(rand_period_start_hr,wet_period_start(month(dy)-1,:))
         rand_period_start_mn = (3 - 0) / 32766.0 * rand()               ; start of period - minute
         period_start_timeindex = period_start_hr*4 + round(rand_period_start_mn,3)

         ;--
         ; Define duration of period - based on statistical distribution from station

         rand_period_duration = (1. - 0) / 32766.0 * rand()                ; 1.2 used to increase length of wet periods 
         period_duration = closest_val(rand_period_duration,wet_period_duration(month(dy)-1,:)) + 1

         do while (period_precip / period_duration .ge. precip_upplimit_60mins)
           period_duration = period_duration + 1
         end do

;         do while (period_precip / period_duration .le. precip_lowlimit_60mins)
;           period_duration = period_duration - 1
;         end do

;         print (numwetperiods + " " + year(dy) + " " + month(dy) + " " + day(dy) + " " + rand_period_precip + " " + period_precip + " " + period_start_timeindex + " " + period_duration)
         ;--
         ; Define ending time of period - based on above information

         period_end_timeindex = period_start_timeindex + period_duration
         if (period_end_timeindex .gt. time_steps) then
           period_start_timeindex = period_start_timeindex - (period_end_timeindex - time_steps)
           if (period_start_timeindex .lt. 0) then
              period_start_timeindex = 0
           end if
           period_end_timeindex = time_steps
;           new_period_duration = period_end_timeindex - period_start_timeindex
;           print ("model: " + model + " day: " + dy + " period_precip= " + period_precip + " new_period_duration=" + new_period_duration + " period_precip/new_period_duration = " + period_precip / new_period_duration)
         end if
         ;--

         ; Introduce random fluctuations on the rain in each interval

         fallen_precip2 = 0.0
         rand_precip_array = new((/period_end_timeindex - period_start_timeindex/),float)

;         precip_period_array = new((/period_end_timeindex - period_start_timeindex/),float)
         do t = period_start_timeindex, period_end_timeindex - 1
            t_index = t - period_start_timeindex

            if (t .ne. period_end_timeindex - 1) then 
               rand_precip = period_precip / period_duration * ((1.5 - 0.5)/32766.0 * rand() + 0.5)
               do while (rand_precip .gt. precip_upplimit_60mins) 
                 rand_precip = period_precip / period_duration * ((1.5 - 0.5)/32766.0 * rand() + 0.5)
               end do
;               do while (rand_precip .lt. precip_lowlimit_60mins) 
;;                 if (period_duration .gt. 1) then
;;                    period_duration = period_duration - 1    
;;                    period_end_timeindex = period_end_timeindex - 1
;;                 end if
;                 rand_precip = rand_precip + precip_lowlimit_60mins * ((1.5 - 0.5)/32766.0 * rand() + 0.5)
;;                 rand_precip = period_precip / period_duration * ((1.5 - 0.5)/32766.0 * rand() + 0.5)
;               end do
            else
               rand_precip = period_precip - fallen_precip2
               if (rand_precip .gt. precip_upplimit_60mins) then
                  period_start_timeindex = period_start_timeindex - 1
                  if (period_start_timeindex .lt. 0) then
                     period_start_timeindex = 0
                     period_end_timeindex = period_end_timeindex + 1
                  end if
                  delete(rand_precip_array)
                  rand_precip_array = new((/period_end_timeindex - period_start_timeindex/),float)
;                  print ("period_start_timeindex: " + period_start_timeindex + " period_end_timeindex: " + period_end_timeindex)
                  period_duration = period_duration + 1
                  t = period_start_timeindex - 1
                  fallen_precip2 = 0.0
                  rand_precip = 0.0
               end if
            end if

            if (fallen_precip2 + rand_precip .gt. period_precip) then
              rand_precip = period_precip - fallen_precip2
            end if
;            precip_period_array(t_index) = rand_precip
;            print ("model= " + model + " day= " + dy + " t= " + t + " period_precip/period_duration= " + period_precip/period_duration + " rand_precip= " + rand_precip)

            rand_precip_array(t_index) = rand_precip

            fallen_precip2 = fallen_precip2 + rand_precip
         end do
         pr_new(dy,period_start_timeindex:period_end_timeindex-1,model) = pr_new(dy,period_start_timeindex:period_end_timeindex-1,model) + rand_precip_array
         delete(rand_precip_array)
         
       end do
       
       ;--
       ; Introduce random fluctuations on the rain in each interval
               
;       do t = 0, time_steps - 1
;         pr_new(dy,t,model) = (pr_new(dy,t,model) * (2.0-0.5)/32766.0 * rand()) ; / 10.0
;         if (pr_new(dy,t,model) .gt. 1.5*precip_upplimit_60mins) then        
;            print ("After_random: 15minute precip = " + pr_new(dy,t,model) + " " + "  Model: " + model)
;;           pr_new(dy,t,model) = 1.0
;         end if
;        end do
;       print (pr_new(dy,:,model) + " ")

       ;--
       ; Match the total daily precip to that from the model

       day_precip_sum = dim_sum(pr_new(dy,:,model))
;       print ("Check: day_precip_sum/max_precip = " + day_precip_sum/pr_ref(dy,model))
;       if (day_precip_sum .eq. 0) then
;         day_precip_sum = 1.0
;       end if
       precip_corrfactor = 1.0 / day_precip_sum                        ; pr_ref(dy,model) / day_precip_sum
       pr_new(dy,:,model) = pr_new(dy,:,model) * precip_corrfactor 

     end if
     
   end do
   
 end do  


; Reorder Array
 pr_xmins = new((/time_steps*num_days,num_models/),float)
 do dy = 0, num_days - 1
   pr_xmins(dy*time_steps:(dy+1)*time_steps - 1,:) = pr_new(dy,:,:)
 end do

;-------------------------------------------------------------------------------------------
; Output data to ASCII file
 print (systemfunc ("date") + " Writing data to file")


   output_array = new((/time_steps*num_days,8/),"float")
   
   output_array(:,0) = tofloat(time_year)
   output_array(:,1) = time_month
   output_array(:,2) = time_day
   output_array(:,3) = time_hour
   output_array(:,4) = time_min
   output_array(:,5) = pr_xmins(:,0)
   output_array(:,6) = pr_xmins(:,1)
   output_array(:,7) = pr_xmins(:,2)

   file_name = "Output_new6/precip_dissag.dat"
   opt@fout = file_name
   write_matrix(output_array,fmt,opt)
  
; end do         ; End station loop


end

