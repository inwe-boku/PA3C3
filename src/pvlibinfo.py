from pvlib import pvsystem

invdb = pvsystem.retrieve_sam('CECInverter')

invdb = invdb.transpose()

## filter all 240V systems

invdb = invdb[invdb["Vac"] == '240']

## filter by Watts Peak Power from Panels
Wmin = 5000
Wmax = 6000

invdb = invdb[invdb["Pdco"] >= Wmin]
invdb = invdb[invdb["Pdco"] <= Wmax]

print(invdb)