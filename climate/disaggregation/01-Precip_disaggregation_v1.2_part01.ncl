; Analyse the precipitation distribution with time.

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

begin

 intensity_mon = new((/15,25,100/),"integer")     ; nonsense... this variable is not calculated and written as zero
 intensity_mon = 0                          ; nonsense

 fin   =  addfile("All_Stations_t2m_precip_inca_hourly.nc","r")
 time  = fin->time
 utime = ut_calendar(time,-5)

 year   = utime(:,0)
 month  = utime(:,1)
 day    = utime(:,2)
 hour   = utime(:,3)
 minute = utime(:,4)

 print("Minutes are "+min(minute)+"    "+max(minute))

 ntim = dimsizes(time)

;--------------
; Begin analysis on monthly basis

 precip_limit = 300     ; upper bound for 10-min precipitation intensity
 intensity = new((/16,24,precip_limit + 1/),"integer")  ; 12 months + four seasons ,  24 hours (I replaced 24*6 with 24)
 intensity = 0

 precip_intensity = new((/16,301/),"float") 
 precip_intensity_sum = new((/16,301/),"float") 
 precip_intensity = 0
 precip_intensity_sum = 0

 time_precip_start = new((/16,25/),"float")  ; only interested in what hour
 time_precip_start_sum = new((/16,25/),"float")  ; only interested in what hour
 time_precip_end = new((/16,25/),"float")    ; precip starts or ends
 time_precip_start = 0
 time_precip_start_sum = 0
 time_precip_end = 0
 
 wet_periods_number = new((/16,31/),"float")
 wet_periods_number_sum = new((/16,31/),"float")
 wet_periods_number = 0
 wet_periods_number_sum = 0
; dry_periods_number = new((/16/),"float")
; dry_periods_number = 0

 dry_periods_limit = 4100       ; never used
 wet_periods_limit = 144        ; never used
 wet_periods_duration = new((/16,145/),"float")
 wet_periods_duration_sum = new((/16,145/),"float")
 wet_periods_duration_rescale = new((/16,25/),"float")      ; I replaced 97 with 25
 wet_periods_duration = 0
 wet_periods_duration_sum = 0
 wet_periods_duration_rescale = 0
; dry_periods_duration = new((/16,dry_periods_limit+1/),"integer")
; dry_periods_duration = 0
 
 wet_periods_precip = new((/16,4001/),"float")
 wet_periods_precip_sum = new((/16,4001/),"float")
 wet_periods_precip = 0
 wet_periods_precip_sum = 0

 pr = floattointeger((fin->pr(:,4))*10)           ; reading lafnitz Dobersdorf
 pr@_FillValue = -999

 printMinMax(pr,True)
 
 if (.not.ismissing(pr(0))) then
    intensity(month(0)-1,hour(0),pr(0)) = intensity(month(0)-1,hour(0),pr(0)) + 1
 end if
  
 if (.not.ismissing(pr(ntim-1))) then
    intensity(month(ntim-1)-1,hour(ntim-1),pr(ntim-1)) = intensity(month(ntim-1)-1,hour(ntim-1),pr(ntim-1)) + 1
 end if
 
 wet_day = 0
 dry_day = 0
 wet_duration = 0
 wet_period = 0
 precip_period = 0
 do t = 1, ntim - 1 - 1

    if (.not.ismissing(pr(t)) .and. pr(t) .le. precip_limit) then

        mnth = month(t) - 1
        time_index = hour(t) + minute(t)/10

;--
; count length of wet period in terms of time units (10mins)
        if (pr(t) .eq. 0 .and. wet_duration .ne. 0) then
          wet_periods_duration(mnth,wet_duration) = wet_periods_duration(mnth,wet_duration) + 1
          wet_periods_precip(mnth,precip_period) = wet_periods_precip(mnth,precip_period) + 1
          wet_duration = 0
          if (.not.ismissing(pr(t+1)) .and. pr(t+1) .eq. 0) then
             wet_period = wet_period + 1
             precip_period = 0
          end if
        end if
        if (day(t) .ne. day(t-1)) then
           wet_duration = 0
           precip_period = 0
;           print (year(t-1) + " " + month(t-1) + " " + day(t-1) + " " + wet_period + " ")
           wet_periods_number(month(t-1)-1,wet_period) = wet_periods_number(month(t-1)-1,wet_period) + 1
           wet_period = 0
        end if
        if (pr(t) .ne. 0) then
;          print (pr(t) + " ")
          precip_intensity(mnth,pr(t)) = precip_intensity(mnth,pr(t)) + 1
          precip_period = precip_period + pr(t)
;           print (year(t) + " " + month(t) + " " + day(t) + " " + precip_period + " ")
          wet_duration = wet_duration + 1
        end if
;--

        intensity(mnth,time_index,pr(t)) = intensity(mnth,time_index,pr(t)) + 1
	
	if (.not.ismissing(pr(t-1)) .and. .not.ismissing(pr(t+1))) then
;--
; count time of precip start and end
           if (pr(t-1) .eq. 0 .and. pr(t) .ne. 0) then
	      time_precip_start(mnth,hour(t)) = time_precip_start(mnth,hour(t)) + 1
	   end if

           if (pr(t+1) .eq. 0 .and. pr(t) .ne. 0) then
	      time_precip_end(mnth,hour(t)) = time_precip_end(mnth,hour(t)) + 1
           end if
       end if
	
    end if
 end do
 
 intensity(12,:,:) = dim_sum_n(intensity(0:1,:,:),0) + intensity(11,:,:)
 intensity(13,:,:) = dim_sum_n(intensity(2:4,:,:),0)
 intensity(14,:,:) = dim_sum_n(intensity(5:7,:,:),0)
 intensity(15,:,:) = dim_sum_n(intensity(8:10,:,:),0)

 precip_intensity(12,:) = dim_sum_n(precip_intensity(0:1,:),0) + precip_intensity(11,:)
 precip_intensity(13,:) = dim_sum_n(precip_intensity(2:4,:),0)
 precip_intensity(14,:) = dim_sum_n(precip_intensity(5:7,:),0)
 precip_intensity(15,:) = dim_sum_n(precip_intensity(8:10,:),0)
 precip_intensity(:,300) = dim_sum_n(precip_intensity(:,0:299),1)

; Accumulative sum
 do i = 0, 299
   precip_intensity_sum(:,i) = dim_sum_n(precip_intensity(:,0:i),1)
 end do
 precip_intensity(:,0:299) = precip_intensity_sum(:,0:299)

 time_precip_start(12,:) = dim_sum_n(time_precip_start(0:1,:),0) + time_precip_start(11,:)
 time_precip_start(13,:) = dim_sum_n(time_precip_start(2:4,:),0)
 time_precip_start(14,:) = dim_sum_n(time_precip_start(5:7,:),0)
 time_precip_start(15,:) = dim_sum_n(time_precip_start(8:10,:),0)
 time_precip_start(:,24) = dim_sum_n(time_precip_start(:,0:23),1)
 
; Accumulative sum
 do i = 0, 23
   time_precip_start_sum(:,i) = dim_sum_n(time_precip_start(:,0:i),1)
 end do
 time_precip_start(:,0:23) = time_precip_start_sum(:,0:23)

 time_precip_end(12,:) = dim_sum_n(time_precip_end(0:1,:),0) + time_precip_end(11,:)
 time_precip_end(13,:) = dim_sum_n(time_precip_end(2:4,:),0)
 time_precip_end(14,:) = dim_sum_n(time_precip_end(5:7,:),0)
 time_precip_end(15,:) = dim_sum_n(time_precip_end(8:10,:),0)
 time_precip_end(:,24) = dim_sum_n(time_precip_end(:,0:23),1)

 wet_periods_number(12,:) = dim_sum_n(wet_periods_number(0:1,:),0) + wet_periods_number(11,:)
 wet_periods_number(13,:) = dim_sum_n(wet_periods_number(2:4,:),0)
 wet_periods_number(14,:) = dim_sum_n(wet_periods_number(5:7,:),0)
 wet_periods_number(15,:) = dim_sum_n(wet_periods_number(8:10,:),0)
 wet_periods_number(:,30) = dim_sum_n(wet_periods_number(:,0:29),1)

; Accumulative sum
 do i = 0, 29
   wet_periods_number_sum(:,i) = dim_sum_n(wet_periods_number(:,0:i),1)
 end do
 wet_periods_number(:,0:29) = wet_periods_number_sum(:,0:29)

 wet_periods_precip(12,:) = dim_sum_n(wet_periods_precip(0:1,:),0) + wet_periods_precip(11,:)
 wet_periods_precip(13,:) = dim_sum_n(wet_periods_precip(2:4,:),0)
 wet_periods_precip(14,:) = dim_sum_n(wet_periods_precip(5:7,:),0)
 wet_periods_precip(15,:) = dim_sum_n(wet_periods_precip(8:10,:),0)
 wet_periods_precip(:,4000) = dim_sum_n(wet_periods_precip(:,0:3999),1)

; Accumulative sum
 do i = 0, 3999
   wet_periods_precip_sum(:,i) = dim_sum_n(wet_periods_precip(:,0:i),1)
 end do
 wet_periods_precip(:,0:3999) = wet_periods_precip_sum(:,0:3999)

 wet_periods_duration(12,:) = dim_sum_n(wet_periods_duration(0:1,:),0) + wet_periods_duration(11,:)
 wet_periods_duration(13,:) = dim_sum_n(wet_periods_duration(2:4,:),0)
 wet_periods_duration(14,:) = dim_sum_n(wet_periods_duration(5:7,:),0)
 wet_periods_duration(15,:) = dim_sum_n(wet_periods_duration(8:10,:),0)
 wet_periods_duration(:,144) = dim_sum_n(wet_periods_duration(:,0:143),1)

; Accumulative sum
 do i = 0, 143
   wet_periods_duration_sum(:,i) = dim_sum_n(wet_periods_duration(:,0:i),1)
 end do
 wet_periods_duration(:,0:143) = wet_periods_duration_sum(:,0:143)


 swap = "n"
 j = 1
 do i = 1, 24
   if (swap .eq. "n") then
     wet_periods_duration_rescale(:,i) = wet_periods_duration(:,j) + wet_periods_duration(:,j+1)/2
     j = j + 2
     swap = "y"
   else
     wet_periods_duration_rescale(:,i) = wet_periods_duration(:,j-1)/2 + wet_periods_duration(:,j)
     swap = "n"
     j = j + 1
   end if
 end do
 
 do i = 0, 15
   time_precip_start(i,0:23) = time_precip_start(i,0:23) / time_precip_start(i,24)
   time_precip_end(i,0:23) = time_precip_end(i,0:23) / time_precip_end(i,24)
   wet_periods_duration_rescale(i,0:23) = wet_periods_duration_rescale(i,0:23) / wet_periods_duration_rescale(i,24)
   wet_periods_number(i,0:29) = wet_periods_number(i,0:29) / wet_periods_number(i,30)
   wet_periods_precip(i,0:3999) = wet_periods_precip(i,0:3999) / wet_periods_precip(i,4000)
   precip_intensity(i,0:299) = precip_intensity(i,0:299) / precip_intensity(i,300)
 end do

;-------------------------------------------------------------------------------------------
; Output data to ASCII file - time precip start
  print (systemfunc ("date") + " Writing data to file")

 output_array_time_precip_start = new((/25/),"string")

 periods = ispan(0,23,1)

 output_array_time_precip_start(0) = " Hr   Jan    Feb    Mar    Apr    May    Jun    Jul    Aug    Sep    Oct    Nov    Dec      Win    Spr    Sum    Aut"
 do p = 0, 23 
   output_array_time_precip_start(p+1) = sprinti("%3i",periods(p)) + " " + sprintf("%6.4f",time_precip_start(0,p)) + " " + sprintf("%6.4f",time_precip_start(1,p)) + " " + sprintf("%6.4f",time_precip_start(2,p)) + " " + sprintf("%6.4f",time_precip_start(3,p)) + " " + sprintf("%6.4f",time_precip_start(4,p)) + " " + sprintf("%6.4f",time_precip_start(5,p)) + " " + sprintf("%6.4f",time_precip_start(6,p)) + " " + sprintf("%6.4f",time_precip_start(7,p)) + " " + sprintf("%6.4f",time_precip_start(8,p)) + " " + sprintf("%6.4f",time_precip_start(9,p)) + " " + sprintf("%6.4f",time_precip_start(10,p)) + " " + sprintf("%6.4f",time_precip_start(11,p)) + "   " + sprintf("%6.4f",time_precip_start(12,p)) + " " + sprintf("%6.4f",time_precip_start(13,p)) + " " + sprintf("%6.4f",time_precip_start(14,p)) + " " + sprintf("%6.4f",time_precip_start(15,p))
 end do
 
 asciiwrite("Statistics/time_precip_start_1hr_"+sta(st)+".dat"", output_array_time_precip_start)

;-------------------------------------------------------------------------------------------
; Output data to ASCII file - time precip end
  print (systemfunc ("date") + " Writing data to file")

 output_array_time_precip_end = new((/25/),"string")

 periods = ispan(0,23,1)

 output_array_time_precip_end(0) = " Hr   Jan    Feb    Mar    Apr    May    Jun    Jul    Aug    Sep    Oct    Nov    Dec      Win    Spr    Sum    Aut"
 do p = 0, 23 
   output_array_time_precip_end(p+1) = sprinti("%3i",periods(p)) + " " + sprintf("%6.4f",time_precip_end(0,p)) + " " + sprintf("%6.4f",time_precip_end(1,p)) + " " + sprintf("%6.4f",time_precip_end(2,p)) + " " + sprintf("%6.4f",time_precip_end(3,p)) + " " + sprintf("%6.4f",time_precip_end(4,p)) + " " + sprintf("%6.4f",time_precip_end(5,p)) + " " + sprintf("%6.4f",time_precip_end(6,p)) + " " + sprintf("%6.4f",time_precip_end(7,p)) + " " + sprintf("%6.4f",time_precip_end(8,p)) + " " + sprintf("%6.4f",time_precip_end(9,p)) + " " + sprintf("%6.4f",time_precip_end(10,p)) + " " + sprintf("%6.4f",time_precip_end(11,p)) + "   " + sprintf("%6.4f",time_precip_end(12,p)) + " " + sprintf("%6.4f",time_precip_end(13,p)) + " " + sprintf("%6.4f",time_precip_end(14,p)) + " " + sprintf("%6.4f",time_precip_end(15,p))
 end do
 
 asciiwrite("Statistics/time_precip_end_1hr_"+sta(st)+".dat"", output_array_time_precip_end)

 delete (periods)

;-------------------------------------------------------------------------------------------
; Output data to ASCII file - wet period duration
  print (systemfunc ("date") + " Writing data to file")

 output_array_wet_periods_duration = new(24,"string")

 periods = ispan(0,24+1,1)

 output_array_wet_periods_duration(0) = "Num   Jan    Feb    Mar    Apr    May    Jun    Jul    Aug    Sep    Oct    Nov    Dec      Win    Spr    Sum    Aut"
 do p = 1, 24 - 1
   output_array_wet_periods_duration(p) = sprinti("%3i",periods(p)) + " " + sprintf("%6.4f",wet_periods_duration_rescale(0,p)) + " " + sprintf("%6.4f",wet_periods_duration_rescale(1,p)) + " " + sprintf("%6.4f",wet_periods_duration_rescale(2,p)) + " " + sprintf("%6.4f",wet_periods_duration_rescale(3,p)) + " " + sprintf("%6.4f",wet_periods_duration_rescale(4,p)) + " " + sprintf("%6.4f",wet_periods_duration_rescale(5,p)) + " " + sprintf("%6.4f",wet_periods_duration_rescale(6,p)) + " " + sprintf("%6.4f",wet_periods_duration_rescale(7,p)) + " " + sprintf("%6.4f",wet_periods_duration_rescale(8,p)) + " " + sprintf("%6.4f",wet_periods_duration_rescale(9,p)) + " " + sprintf("%6.4f",wet_periods_duration_rescale(10,p)) + " " + sprintf("%6.4f",wet_periods_duration_rescale(11,p)) + "   " + sprintf("%6.4f",wet_periods_duration_rescale(12,p)) + " " + sprintf("%6.4f",wet_periods_duration_rescale(13,p)) + " " + sprintf("%6.4f",wet_periods_duration_rescale(14,p)) + " " + sprintf("%6.4f",wet_periods_duration_rescale(15,p))
 end do
 
 asciiwrite("Statistics/wetperiods_duration_hourly_"+sta(st)+".dat"", output_array_wet_periods_duration)
 delete (periods)

; print (fuck)
;-------------------------------------------------------------------------------------------
; Output data to ASCII file - wet periods number
  print (systemfunc ("date") + " Writing data to file")

 output_array_wet_periods_number = new((/30/),"string")

 periods = ispan(0,29,1)

 output_array_wet_periods_number(0) = "Num   Jan    Feb    Mar    Apr    May    Jun    Jul    Aug    Sep    Oct    Nov    Dec      Win    Spr    Sum    Aut"
 do p = 1, 29
   output_array_wet_periods_number(p) = sprinti("%3i",periods(p)) + " " + sprintf("%6.4f",wet_periods_number(0,p)) + " " + sprintf("%6.4f",wet_periods_number(1,p)) + " " + sprintf("%6.4f",wet_periods_number(2,p)) + " " + sprintf("%6.4f",wet_periods_number(3,p)) + " " + sprintf("%6.4f",wet_periods_number(4,p)) + " " + sprintf("%6.4f",wet_periods_number(5,p)) + " " + sprintf("%6.4f",wet_periods_number(6,p)) + " " + sprintf("%6.4f",wet_periods_number(7,p)) + " " + sprintf("%6.4f",wet_periods_number(8,p)) + " " + sprintf("%6.4f",wet_periods_number(9,p)) + " " + sprintf("%6.4f",wet_periods_number(10,p)) + " " + sprintf("%6.4f",wet_periods_number(11,p)) + "   " + sprintf("%6.4f",wet_periods_number(12,p)) + " " + sprintf("%6.4f",wet_periods_number(13,p)) + " " + sprintf("%6.4f",wet_periods_number(14,p)) + " " + sprintf("%6.4f",wet_periods_number(15,p))
 end do
 
 asciiwrite("Statistics/wetperiods_number_hourly_"+sta(st)+".dat"", output_array_wet_periods_number)
 delete (periods)

;-------------------------------------------------------------------------------------------
; Output data to ASCII file - wet periods number
  print (systemfunc ("date") + " Writing data to file")

 output_array_wet_periods_precip = new((/4000/),"string")

 periods = ispan(0,3999,1)

 output_array_wet_periods_precip(0) = "Num   Jan    Feb    Mar    Apr    May    Jun    Jul    Aug    Sep    Oct    Nov    Dec      Win    Spr    Sum    Aut"
 do p = 1, 3999
   output_array_wet_periods_precip(p) = sprinti("%3i",periods(p)) + " " + sprintf("%6.4f",wet_periods_precip(0,p)) + " " + sprintf("%6.4f",wet_periods_precip(1,p)) + " " + sprintf("%6.4f",wet_periods_precip(2,p)) + " " + sprintf("%6.4f",wet_periods_precip(3,p)) + " " + sprintf("%6.4f",wet_periods_precip(4,p)) + " " + sprintf("%6.4f",wet_periods_precip(5,p)) + " " + sprintf("%6.4f",wet_periods_precip(6,p)) + " " + sprintf("%6.4f",wet_periods_precip(7,p)) + " " + sprintf("%6.4f",wet_periods_precip(8,p)) + " " + sprintf("%6.4f",wet_periods_precip(9,p)) + " " + sprintf("%6.4f",wet_periods_precip(10,p)) + " " + sprintf("%6.4f",wet_periods_precip(11,p)) + "   " + sprintf("%6.4f",wet_periods_precip(12,p)) + " " + sprintf("%6.4f",wet_periods_precip(13,p)) + " " + sprintf("%6.4f",wet_periods_precip(14,p)) + " " + sprintf("%6.4f",wet_periods_precip(15,p))
 end do
 
 asciiwrite("Statistics/wetperiods_precip_hourly_"+sta(st)+".dat"", output_array_wet_periods_precip)
 delete (periods)

;-------------------------------------------------------------------------------------------
; Output data to ASCII file - wet periods number
  print (systemfunc ("date") + " Writing data to file")

 output_array_precip_intensity = new((/300/),"string")

 periods = ispan(0,299,1)

 output_array_precip_intensity(0) = "Num   Jan    Feb    Mar    Apr    May    Jun    Jul    Aug    Sep    Oct    Nov    Dec      Win    Spr    Sum    Aut"
 do p = 1, 299
   output_array_precip_intensity(p) = sprinti("%3i",periods(p)) + " " + sprintf("%6.4f",precip_intensity(0,p)) + " " + sprintf("%6.4f",precip_intensity(1,p)) + " " + sprintf("%6.4f",precip_intensity(2,p)) + " " + sprintf("%6.4f",precip_intensity(3,p)) + " " + sprintf("%6.4f",precip_intensity(4,p)) + " " + sprintf("%6.4f",precip_intensity(5,p)) + " " + sprintf("%6.4f",precip_intensity(6,p)) + " " + sprintf("%6.4f",precip_intensity(7,p)) + " " + sprintf("%6.4f",precip_intensity(8,p)) + " " + sprintf("%6.4f",precip_intensity(9,p)) + " " + sprintf("%6.4f",precip_intensity(10,p)) + " " + sprintf("%6.4f",precip_intensity(11,p)) + "   " + sprintf("%6.4f",precip_intensity(12,p)) + " " + sprintf("%6.4f",precip_intensity(13,p)) + " " + sprintf("%6.4f",precip_intensity(14,p)) + " " + sprintf("%6.4f",precip_intensity(15,p))
 end do
 
 asciiwrite("Statistics/precip_intensity_hourly_"+sta(st)+".dat"", output_array_precip_intensity)

 ;-------------------------------------------------------------------------------------------
; Output data to ASCII file
  print (systemfunc ("date") + " Writing data to file")

 output_array_mon = new((/24*100/),"string")

 time_count = 0
 mn  = 0
 do prcp = 1, 100 - 1
   do hr = 0, 24 - 1
        output_array_mon(time_count) = sprintf("%4.1f",prcp/10.) + " " + sprinti("%3i",hr) + sprinti("%3i",mn*10) + " " + sprintf("%7.4f",hr + mn/6.) + " " + sprinti("%3i",intensity_mon(0,hr,prcp)) + " " + sprinti("%3i",intensity_mon(1,hr,prcp)) + " " + sprinti("%3i",intensity_mon(2, hr,prcp))  + " " + sprinti("%3i",intensity_mon(3, hr,prcp)) + " " + sprinti("%3i",intensity_mon(4, hr,prcp))  + " " + sprinti("%3i",intensity_mon(5, hr,prcp))  + " " + sprinti("%3i",intensity_mon(6, hr,prcp))  + " " + sprinti("%3i",intensity_mon(7, hr,prcp))  + " " + sprinti("%3i",intensity_mon(8, hr,prcp))  + " " + sprinti("%3i",intensity_mon(9, hr,prcp)) + " " + sprinti("%3i",intensity_mon(10, hr,prcp)) + " " + sprinti("%3i",intensity_mon(11, hr,prcp)) + "  " + sprinti("%4i",intensity_mon(12, hr,prcp)) + " " + sprinti("%4i",intensity_mon(13, hr,prcp)) + " " + sprinti("%4i",intensity_mon(14, hr,prcp))
        time_count = time_count + 1
   end do  
 end do

 asciiwrite("precip_mon_"+sta(st)+".dat"", output_array_mon)

end
