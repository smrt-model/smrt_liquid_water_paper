

def get_location_info(location):

    name = '---'
    if location == 'amery':
        coords = dict(lat=-70.35416643, lon=70.94754255)
        ice_thickness = 626
        name = 'Amery'
    elif location == 'larsenc_s':
        coords = dict(lat=-68.21535785, lon=-63.18366130)
        ice_thickness = 300
        name = 'Larsen C'
    elif location == 'larsenc_n':
        coords = dict(lat=-66.46583890, lon=-61.91861995)
        ice_thickness = 300
        name = 'Larsen C'
    elif location == 'george6':
        coords = dict(lat=-73, lon=-70.17)
        ice_thickness = 350
    elif location == 'wilkins':
        coords = dict(lat=-70.71, lon=-71.94)
        ice_thickness = 150
        name = 'Wilkins'
    elif location == "roi_baudouin":
        coords = dict(lat=-69.82102916, lon=31.40237151)
        ice_thickness = 120
        name = 'Baudouin'
    elif location == "shackleton":
        coords = dict(lat=-66.12341724, lon=98.39536370)
        ice_thickness = 200
        name = 'Shackleton'
    elif location == "abbot_o":
        coords = dict(lat=-72.43113569, lon=-101.37158775)
        ice_thickness = 200
    elif location == "aws11":
        coords = dict(lat=-71.170000, lon=-6.800000)
        name = 'Halv.'
        ice_thickness = 886
    elif location == "aws14":
        coords = dict(lat=-67.020000, lon=-61.500000)
        name = 'Larsen C'
        ice_thickness = 279
    elif location == "aws15":
        coords = dict(lat=-67.570000, lon=-62.150000)
        ice_thickness = 292
        name = 'Larsen C'
    elif location == "aws16":
        coords = dict(lat=-71.950000, lon=23.330000)
        raise Exception("not a shelf")
    elif location == "aws17":
        coords = dict(lat=-65.930000, lon=-61.850000)
        name = 'Larsen B'
        ice_thickness = 206
    elif location == "aws18":
        coords = dict(lat=-66.400000, lon=-63.730000)
        ice_thickness = 551
    elif location == "aws19":
        coords = dict(lat=-70.950000, lon=26.270000)
        name = 'Baudouin'
        ice_thickness = 376
    elif location == "aws4":
        coords = dict(lat=-72.750000, lon=-15.480000)
        ice_thickness = 237
    elif location == "aws5":
        coords = dict(lat=-73.100000, lon=-13.170000)
        ice_thickness = 681
        name = 'Maud.'
    elif location == "aws6":
        coords = dict(lat=-74.470000, lon=-11.520000)
        raise Exception("not a shelf")
    else:
        raise Exception(f"Unknown location '{location}'")

    return coords, ice_thickness, name


def site_name(sites):
    
    if isinstance(sites, str):
        return get_location_info(sites)[2]
    else:
        return list(map(lambda site: get_location_info(site)[2], sites))
        
