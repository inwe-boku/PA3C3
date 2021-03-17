; Generate a 15 minute dataset from maximum and minimum temperature data.
; The method used here is based on that from the "European solar radation atlas"
; where the temperature profile is described using three piecewise cosine functions.

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

begin

 model_array = (/"aladin","regcm3","remo"/)
 nmods = dimsizes(model_array)

 ;nst_start = 0
 ;nst_end = 0

 time_interval = 60                  ; time interval in minutes
 ti_inhour = 60/time_interval
 tsteps = 24 * ti_inhour                       ; Number of time steps per day
 a = tsteps - 1

 f1a=addfile("CNRM-tas-pr.nc","r")
 f2a=addfile("ICTP-tas-pr.nc","r")
 f3a=addfile("MPIM-tas-pr.nc","r")

 f1b=addfile("CNRM-hurs-rsds-bcorr.nc","r")
 f2b=addfile("ICTP-hurs-rsds-bcorr.nc","r")
 f3b=addfile("MPIM-hurs-rsds-bcorr.nc","r")

 fhourly=addfile("CNRM-ICTP-MPIM-tasmin-tasmean_tasmax_tas_hourly_allinOne.nc","r")

 ftd = addfile("CNRM-ICTP-MPIM-dewpoint-temperature_daily.nc","r")

 time = f1a->time
 vtmp = f1a->tas
 dims = dimsizes(vtmp)
 
 ndays  = dims(0)
 nstats = dims(1)

 utime             = ut_calendar(time,-5)
 years             = utime(:,0)
 months            = utime(:,1)
 days              = utime(:,2)

 print("ndays and nstats are "+ndays+" "+nstats)

 tas_all    = new((/ndays,nstats,nmods/),float,vtmp@_FillValue)
 tas_all(:,:,0)    = f1a->tas
 tas_all(:,:,1)    = f2a->tas
 tas_all(:,:,2)    = f3a->tas

 hurs_all    = new((/ndays,nstats,nmods/),float,vtmp@_FillValue)
 hurs_all(:,:,0)    = f1b->hurs
 hurs_all(:,:,1)    = f2b->hurs
 hurs_all(:,:,2)    = f3b->hurs

 tas_xmins_all = fhourly->tas_xmins
 
 td_all    = ftd->td

 file_name_list    = asciiread("station_names_list.txt",-1,"string")

 printVarSummary(tas_all)
 printVarSummary(hurs_all)
 printVarSummary(td_all)
 printVarSummary(tas_xmins_all)

 year = new((/ndays,nmods/),"integer")
 month = new((/ndays,nmods/),"integer")
 day = new((/ndays,nmods/),"integer")
 do model = 0, nmods - 1
   year(:,model) = years
   month(:,model) = months
   day(:,model) = days
 end do

 tas_xmins = new((/tsteps*ndays,3/),"float")
 td_day = new((/ndays*tsteps,3/),"float")
 vap_press = new((/ndays*tsteps,3/),"float")
 vap_press_sat = new((/ndays*tsteps,3/),"float")
 output_array = new((/tsteps*ndays,nmods,8/),"float")

 Pi = 4.0 * atan(1.0)

 fmt = "1x,f5.0,1x,f3.0,1x,f3.0,1x,f3.0,1x,f3.0,f10.3,f10.3,f10.3"
 opt = True


 ;--------------------------------------------------------------------------------------------
 ; Daily variation of dew point - depends on monthly radiation

 t_count = 0
 do dy = 0, ndays - 1
     do t = 0, tsteps - 1
         td_day(t_count,:) = 0.5 * sin(2*Pi*((t-9.*ti_inhour)/tsteps))
	 t_count = t_count + 1
     end do	 
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
 time_dawn = new((/ndays/),integer)

 do dy = 0, ndays - 1 
     dayofyear = day_of_year(year(dy,0),month(dy,0),day(dy,0))
     time_dawn(dy) = sunrise(dayofyear-1)
 end do

 td_xmins = new((/ndays*tsteps,nmods/),float)
 td_xmins_final = new((/ndays*tsteps,nstats,nmods/),float)
 rh_xmins = new((/ndays*tsteps,nstats,nmods/),float)

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

 output_array(:,:,0) = tofloat(time_year(:,:))
 output_array(:,:,1) = time_month(:,:)
 output_array(:,:,2) = time_day(:,:)
 output_array(:,:,3) = time_hour(:,:)
 output_array(:,:,4) = time_min(:,:)

 do nst = nst_start, nst_end
     print (systemfunc ("date") + "  Station: " + "tas" + file_name_list(nst))

     tas  = tas_all(:,nst,:)
     hurs = hurs_all(:,nst,:)
     td   = td_all(:,nst,:)
     tas_xmins = tas_xmins_all(:,nst,:)

     file_name = "Output1/rh_" + model_array + "_" + file_name_list(nst)

     ;==========================================================================================                                                                 
     ; Calculate relative humidity according to Debele et al. (2007)
     ;
     ; However, here the mean daily dew point is assumed to occur at 9am local time in accordance 
     ; with the mean dew point temperatures as calculated from the station data.
 
     print (systemfunc ("date") + " Calculating dew-point temperature in " + time_interval + " minute intervals")

     t_count = 0
     do dy = 0, ndays - 1
         do t = 0, tsteps - 1                     ; index to count time steps in a day
	     if (dy .eq. 0) then
	         if (t .lt. 9 * ti_inhour) then
		     ref_9am = 0.0
		     td_xmins(t_count,:) = td(dy,:) + td_day(t,:)
		 else
		     ref_9am = t - 9.*ti_inhour
		     td_xmins(t_count,:) = td(dy,:) + ref_9am / tsteps * (td(dy+1,:) - td(dy,:)) + td_day(t,:)
		 end if
	     end if
       
             if (dy .gt. 0 .and. dy .lt. ndays - 1) then
	         if (t .lt. 9 * ti_inhour) then
                     ref_9am = 15.*ti_inhour + t  
                     td_xmins(t_count,:) = td(dy-1,:) + ref_9am / tsteps * (td(dy,:) - td(dy-1,:)) + td_day(t,:)
		 else
		     ref_9am = t - 9.*ti_inhour
                     td_xmins(t_count,:) = td(dy,:) + ref_9am / tsteps * (td(dy+1,:) - td(dy,:)) + td_day(t,:)
	         end if
             end if

	     if (dy .eq. ndays - 1) then
	         if (t .lt. 9 * ti_inhour) then
		     ref_9am = 15.*ti_inhour + t  
		     td_xmins(t_count,:) = td(dy-1,:) + ref_9am / tsteps * (td(dy,:) - td(dy-1,:)) + td_day(t,:)
		 else
                     td_xmins(t_count,:) = td(dy,:) + td_day(t,:)
		 end if
	     end if

             t_count = t_count + 1
         end do
     end do
     td_xmins_final(:,nst,:) = where(td_xmins .le. tas_xmins, td_xmins, tas_xmins)   ; Check that td <= tas
          
     vap_press = 0.6108 * exp(17.27 * td_xmins_final(:,nst,:)/ (td_xmins_final(:,nst,:) + 237.3))
     vap_press_sat = 0.6108 * exp(17.27 * tas_xmins / (tas_xmins + 237.3))
     rh_xmins(:,nst,:) = 100.0 * vap_press / vap_press_sat

     ;-------------------------------------------------------------------------------------------
     ; Output data to ASCII file
     print (systemfunc ("date") + " Writing data to file")
     
     output_array(:,:,5) = (/tas_xmins(:,:)/)
     output_array(:,:,6) = (/td_xmins_final(:,nst,:)/)
     output_array(:,:,7) = (/rh_xmins(:,nst,:)/)

     opt@fout = file_name(0)
     write_matrix(output_array(:,0,:),fmt,opt)
     opt@fout = file_name(1)
     write_matrix(output_array(:,1,:),fmt,opt)
     opt@fout = file_name(2)
     write_matrix(output_array(:,2,:),fmt,opt)
     
     delete ([/tas,hurs,td/])

 end do         ; End station loop

 copy_VarMeta(tas_xmins_all,td_xmins_final)
 copy_VarMeta(tas_xmins_all,rh_xmins)

 outfname = "CNRM-ICTP-MPIM-tasmin_rh_1hourly_"+nst_start+".nc"
 system(" 'rm' -f " + outfname)
 fout=addfile(outfname,"c")
 filedimdef(fout,"time",-1,True)
 fout->tas_xmins=tas_xmins_all
 fout->td_xmins_final=td_xmins_final
 fout->rh_xmins=rh_xmins
 
end

