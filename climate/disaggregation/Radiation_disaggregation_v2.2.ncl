; Generate a 15 minute dataset from maximum and minimum temperature data.
; The method used here is based on that from the "European solar radation atlas"
; where the temperature profile is described using three piecewise cosine functions.

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

begin
 
 time_interval = 15
 ti_inhour = 60/time_interval
 time_steps = 24 * ti_inhour                       ; Number of time steps per day
 a = time_steps - 1

 f1=addfile("CNRM-hurs-rsds-bcorr.nc","r")
 f2=addfile("ICTP-hurs-rsds-bcorr.nc","r")
 f3=addfile("MPIM-hurs-rsds-bcorr.nc","r")

 model_array = (/"aladin","regcm3","remo"/)
 nmods = dimsizes(model_array)

 time = f1->time
 vtmp = f1->rsds
 dims = dimsizes(vtmp)

 ndays = dims(0)
 nstat = dims(1)

 print("ndays and nstat are "+ndays+" "+nstat)

 rad_all    = new((/ndays,nstat,nmods/),float,vtmp@_FillValue)

 rad_all(:,:,0) = f1->rsds
 rad_all(:,:,1) = f2->rsds
 rad_all(:,:,2) = f3->rsds
 utime          = ut_calendar(time,-5)
 years          = utime(:,0)
 months         = utime(:,1)
 days           = utime(:,2)

 file_name_list    = asciiread("station_names_list.txt",-1,"string")

 Pi = 4.0 * atan(1.0)
 
 fmt = "1x,f5.0,1x,f3.0,1x,f3.0,1x,f3.0,1x,f3.0,f10.3"
 opt = True
 

;-------------------------------------------------------------------------------------------
; Open sunrise data file - Graz is used for the reference (corrected for summer time change)
 sunrise_raw = asciiread("sunrise_sunset_graz.dat",(/366,6/),"float")
 sunrise = new((/366/),integer)
 sunset = new((/366/),integer)
 sunrisem = new((/366/),float)
 sunsetm = new((/366/),float)

 sunrise(:365) = round((sunrise_raw(:,2) - 1 + sunrise_raw(:,3)/60)*4,3)
 sunset(:365) = round((sunrise_raw(:,4) - 1 + sunrise_raw(:,5)/60)*4,3)
 sunrisem(:365) = (sunrise_raw(:,2) - 1 + sunrise_raw(:,3)/60)
 sunsetm(:365) = (sunrise_raw(:,4) - 1 + sunrise_raw(:,5)/60)
 sms = sunset - sunrise
 sps = sunset + sunrise
 smsm = sunsetm - sunrisem
 spsm = sunsetm + sunrisem
;-------------------------------------------------------------------------------------------

  year = new((/ndays,nmods/),"integer")
  month = new((/ndays,nmods/),"integer")
  day = new((/ndays,nmods/),"integer")
  do model = 0, nmods - 1
    year(:,model) = years
    month(:,model) = months
    day(:,model) = days
  end do

  rsds_max = new((/ndays, nmods/),float)
  rsds_xmins = new((/ndays*time_steps,nstat,nmods/),float)
  rsds_xmins = 0.0

  output_array = new((/time_steps*ndays,nmods,6/),"float")

;-------------------------------------------------------------------------------------------
; Time definitions
  time_year = new((/time_steps*ndays,nmods/),integer)
  time_month = new((/time_steps*ndays,nmods/),integer)
  time_day = new((/time_steps*ndays,nmods/),integer)
  time_hour = new((/time_steps*ndays,nmods/),integer)
  time_min = new((/time_steps*ndays,nmods/),integer)

  count = 0
  do dy = 0, ndays - 1
    dy_start = dy * time_steps
    dy_end = (dy + 1) * time_steps - 1
    count = dy * time_steps
    do i = dy_start, dy_end
      time_day(i,:) = day(dy,:)
      time_month(i,:) = month(dy,:)
      time_year(i,:) = year(dy,:)
    end do
  end do

  do model = 0, nmods - 1
    do i = 0, time_steps - 1
      time_hour(i::time_steps, model) = floattoint(i/ti_inhour)
    end do
    do i = 0, ti_inhour - 1
      time_min(i::ti_inhour,model) = time_interval * i
    end do
  end do

   print(sunrise)
   print(sunset)
   sleep(500)
  

do nst = 0, nstat - 1
   rsds_mean = rad_all(:,nst,:)
   file_name = "Output/"+model_array+"_"+file_name_list(nst)
   print (systemfunc ("date") + "  Station: " + file_name_list(nst))

;-------------------------------------------------------------------------------------------
; Assume a quadratic profile for radiation of the form:
; rsds_parabola = - (t - sunrise) * (t - sunset) / (sms/2)^2 * rsds_max
; rsds_cosine = (cos(2*Pi*((t - sps/2) / (sms))) + 1)* rsds_max / 2
;
; rsds_parabola_integral = -(2/sms)**2 * rsds_max * (1./3 * t**2 - sps/2 * t + sunrise * sunset)
; rsds_cosine_integral = sms/(2*Pi) * (sin(2*Pi*((t - sps/2) / sms)) + t) * rsds_max / 2

; Sum of two integrals
; rsds = 0.5*(rsds_parabola_integral + rsds_cosine_integral)
; rsds = 0.5*(rsds_max * (-(2/sms)**2*(1./3 * t**2 - sps/2 * t + sunrise * sunset) + sms/(2*Pi) * (sin(2*Pi*((t - sps/2) / sms)) + t) / 2.))

; Sum of the two integrals must be equal to 24 * rsds_mean
; rsds_max = 24 * rsds_mean / (-(2/sms)**2*(1./3 * t**2 - sps/2 * t + sunrise * sunset) + sms/(2*Pi) * (sin(2*Pi*((t - sps/2) / sms)) + t) / 2.)

;-------------------------------------------------------------------------------------------
;-------------------------------------------------------------------------------------------
; Calculate radiation at 15 minute intervals

 print (systemfunc ("date") + " Calculating radiation in " + time_interval + " minute intervals")

 do dy = 0, ndays - 1
   if (month(dy,0) .eq. 1 .and. day(dy,0) .eq. 1) then
     day_year = 0
   end if
   rsds_max(dy,:) = 2 * 24. * rsds_mean(dy,:) / (-1.*(2/smsm(day_year))^2 * (sunsetm(day_year)^2 * (-sunsetm(day_year)/6. + sunrisem(day_year)/2.) - 1. * sunrisem(day_year)^2 * (-sunrisem(day_year)/6. + sunsetm(day_year)/2.)) + smsm(day_year)/2.)

   do t = sunrise(day_year), sunset(day_year)
     t_count = t + dy * time_steps
     rsds_xmins(t_count,nst,:) = rsds_max(dy,:)/2. * (-1. * (2./sms(day_year))^2*(t^2 - sps(day_year) * t + sunrise(day_year) * sunset(day_year)) + (cos(2*Pi*((t - sps(day_year)/2.) / sms(day_year))) + 1.) / 2.)
   end do
   day_year = day_year + 1
 end do

;-------------------------------------------------------------------------------------------
; Output data to ASCII file
   print (systemfunc ("date") + " Writing data to file")
   
   output_array(:,:,0) = (/tofloat(time_year(:,:))/)
   output_array(:,:,1) = (/time_month(:,:)/)
   output_array(:,:,2) = (/time_day(:,:)/)
   output_array(:,:,3) = (/time_hour(:,:)/)
   output_array(:,:,4) = (/time_min(:,:)/)
   output_array(:,:,5) = (/rsds_xmins(:,nst,:)/)

   opt@fout = file_name(0)
   write_matrix(output_array(:,0,:),fmt,opt)
   opt@fout = file_name(1)
   write_matrix(output_array(:,1,:),fmt,opt)
   opt@fout = file_name(2)
   write_matrix(output_array(:,2,:),fmt,opt)
 
end do       ; End station loop

 rsds_xmins!0 = "time"
 rsds_xmins!1 = "stations"
 rsds_xmins!2 = "models"
 
 units  = "minutes since 1950-01-01 00:00:00"
 seconds    = time_min(:,0)
 seconds    = 0

 time1   = cd_inv_calendar(time_year(:,0),time_month(:,0),time_day(:,0),time_hour(:,0),time_min(:,0),seconds,units, 0)
 time1!0 = "time"
 time1&time = time1
 
 rsds_xmins&time = time1
 rsds_xmins&stations = ispan(1,nstat,1)
 rsds_xmins&models  = ispan(1,nmods,1)

 outfname = "CNRM-ICTP-MPIM-rsds-bcorr_15mins.nc"
 system(" 'rm' -f " + outfname)
 fout=addfile(outfname,"c")
 filedimdef(fout,"time",-1,True)
 fout->rsds_xmins=rsds_xmins

end


