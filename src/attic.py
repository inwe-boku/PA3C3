def calc_pvoutput(point, ts_rtw, tracking, capacity_kWp, tz='UTC'):
    """
    calculates ac output in Wh of a PV installation either heading
    to the ecuator and an inclination equal to the latitude or assuming
    a single-axis tracking system for a particular location. It requires
    time series of solar radiation, temperature (in °C at 2 m)and wind
    speed (in m/s at 1 m altitude) as well as the coordinates of the location
    and either 0 or 1 to define the type of tracker as input
    """
    altitude = 0
    if point['altitutde']:
        altitude = point['altitutde']
    location = pvlib.location.Location(
        latitude=point['geometry'].y[0],
        longitude=point['geometry'].x[0],
        altitude=point['altitude'],
        tz='UTC')
    temperature = ts_rtw['temp']
    wind_speed = ts_rtw['wind']
    ghi_input = ts_rtw['rad']
    timeindex = ghi_input.asfreq(freq='1H').index
    dayofyear = timeindex.dayofyear

    solarpos = pvlib.solarposition.pyephem(timeindex, location.latitude,
                                           location.longitude,
                                           temperature=np.mean(temperature))
    zenith = solarpos['zenith']

    if tracking == 0:
        slope = lats
        if lats >= 0:
            aspect = 0
        elif lats < 0:
            aspect = 180
    elif tracking == 1:

        tracker_data = pvlib.tracking.singleaxis(solarpos['apparent_zenith'],
                                                 solarpos['azimuth'],
                                                 axis_tilt=0,
                                                 axis_azimuth=0,
                                                 max_angle=90,
                                                 backtrack=True,
                                                 gcr=2.0 / 7.0)
        slope = tracker_data['surface_tilt']
        aspect = tracker_data['surface_azimuth']
    # solartime = solarpos['solar_time']
    # clearsky_irrad = location.get_clearsky(timeindex)
    # clearsky_irrad['2018-01-01'].plot()
    dni_pre = pvlib.irradiance.disc(ghi_input, Zenith, dayofyear)['dni']
    dhi_pre = ghi_input - dni_pre * cosd(Zenith)
    weather = pd.DataFrame({'ghi': ghi_input,
                            'dni': dni_pre,
                            'dhi': dhi_pre,
                            'temp_air': temperature,
                            'wind_speed': wind_speed},
                           index=timeindex)
    # weather['2017-06-01':'2017-06-08'].plot(figsize=(18,6))
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
    # the follow selection requires some sort of automatization
    sandia_module = sandia_modules['Silevo_Triex_U300_Black__2014_']
    # Tries with the stc where understimating the yearly sum. Decided to use
    # the PTC
    PTC = 280.5
    cec_inverter = cec_inverters['ABB__MICRO_0_3_I_OUTD_US_240_240V__CEC_2014_']
    # check that the Paco is at least equal to the STC
    number_of_panels_1kWp = 1000 / PTC
    area_1kWp = number_of_panels_1kWp * sandia_module['Area']
    system = PVSystem(surface_tilt=slope, surface_azimuth=aspect,
                      module_parameters=sandia_module,
                      inverter_parameters=cec_inverter)
    mc = ModelChain(system, location)
    mc.run_model(times=weather.index, weather=weather)
    pv_output = (mc.ac * number_of_panels_1kWp *
                 installed_capacity_kWp).fillna(0)
    return pv_output


def calcPV():
    cecmodules = pvlib.pvsystem.retrieve_sam('CECMod')
    cecinverters = pvlib.pvsystem.retrieve_sam('CECInverter')

    # 'LG Electronics Inc. LG415N2T-L5'
    module = cecmodules['Aleo Solar S59y280']
    # 'LG Electronics Inc : D007KEEN261 [240V]'
    inverter = cecinverters['AEG Power Solutions: Protect MPV.150.01 [480V]']
    temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[
        'pvsyst']['open_rack_glass_glass']

   # for location in coordinates:
   #     latitude, longitude, name, altitude, timezone = location
   #     weather = pvlib.iotools.get_pvgis_tmy(latitude, longitude,
   #                                           map_variables=True)[0]

    #    weather.index.name = "utc_time"
    #    tmys.append(weather)

    system = {'module': module, 'inverter': inverter, 'surface_azimuth': 180}
    energies = {}
    for location, weather in zip(coordinates, tmys):
        latitude, longitude, name, altitude, timezone = location

    system['surface_tilt'] = latitude

    solpos = pvlib.solarposition.get_solarposition(

        time=weather.index,

        latitude=latitude,

        longitude=longitude,

        altitude=altitude,

        temperature=weather["temp_air"],

        pressure=pvlib.atmosphere.alt2pres(altitude),

    )

    dni_extra = pvlib.irradiance.get_extra_radiation(weather.index)

    airmass = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'])

    pressure = pvlib.atmosphere.alt2pres(altitude)

    am_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)

    aoi = pvlib.irradiance.aoi(
        system['surface_tilt'],
        system['surface_azimuth'],
        solpos["apparent_zenith"],
        solpos["azimuth"],
    )

    total_irradiance = pvlib.irradiance.get_total_irradiance(
        system['surface_tilt'],
        system['surface_azimuth'],
        solpos['apparent_zenith'],
        solpos['azimuth'],
        weather['dni'],
        weather['ghi'],
        weather['dhi'],
        dni_extra=dni_extra,
        model='haydavies',
    )

    cell_temperature = pvlib.temperature.sapm_cell(
        total_irradiance['poa_global'],
        weather["temp_air"],
        weather["wind_speed"],
        **temperature_model_parameters,
    )

    effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(
        total_irradiance['poa_direct'],
        total_irradiance['poa_diffuse'],
        am_abs,
        aoi,
        module,
    )

    dc = pvlib.pvsystem.sapm(effective_irradiance, cell_temperature, module)
    ac = pvlib.inverter.sandia(dc['v_mp'], dc['p_mp'], inverter)
    annual_energy = ac.sum()
    energies[name] = annual_energy
    energies = pd.Series(energies)
