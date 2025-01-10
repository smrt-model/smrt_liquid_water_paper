
import json
import copy
from collections.abc import Sequence

import numpy as np
from smrt import make_model, make_snowpack, sensor_list, make_ice_column
from smrt.core.globalconstants import DENSITY_OF_ICE, DENSITY_OF_WATER
from smrt.core import lib

from smrt.permittivity.water import water_permittivity
from smrt.permittivity.ice import ice_permittivity_maetzler06

import matplotlib.pyplot as plt

import xarray as xr
import pandas as pd

depth = [0, 3, 8, 20]
depth_v2 = [0, 3, 8, 20]

frequency_colors = {'01': '#77C000', '06': '#ee188d', '10': '#03aa87', '19': '#f46817', '37': "#7b7bc6"}


def prepare_snowpack(annual_temperature, season_temperature, ice_thickness, density=[350, 500, 600, 850], ssa=[20, 6, 5, 3],
                     sigma_density=[0, 0, 0, 0], microstructure_model="teubner_strey", resolution="medium", add_water_substrate=True):

    if resolution == "high":
        z = np.arange(0, 30, 0.1)
    elif resolution == "medium":
        z = np.hstack((np.arange(0, 2, 0.1), np.arange(2, 10, 0.2), np.arange(10, 30, 0.5)))
    elif resolution == "low":
        z = np.hstack((np.arange(0, 2, 0.2), np.arange(2, 10, 0.4), np.arange(10, 30, 1)))
    else:
        raise Exception("Unknown resolution")

    # go to the closeoff (assumed at 60m) and then to the ice bottom (-> ocean for ice shelves)
    z = np.hstack((z, np.linspace(30, 60, 3), [ice_thickness]))

    density = np.append(density, [910, 912])
    ssa = np.append(ssa, [0.5, 0.5])
    sigma_density = np.append(sigma_density, [1, 1])
    ldepth = depth + [60, ice_thickness]

    z_mid = (z[1:] + z[:-1]) / 2
    thickness = np.diff(z)

    ssa = np.interp(z_mid, ldepth, ssa)

    density = np.interp(z_mid, ldepth, density)
    sigma_density = np.interp(z_mid, ldepth, sigma_density)

    #density = np.minimum(density0 + (density1 - density0) * (z / z_density), 916)
    #density = np.sqrt((density1**2 - density0**2) * z / z_density + density0**2)
    # add noise
    density += np.random.normal(0, sigma_density)

    # clip
    density = np.minimum(density, 912)  # the ice must be a bit bubbly !!

    alpha = z_mid / ice_thickness
    temperature = (season_temperature - annual_temperature) * np.exp(-z_mid / 2.0)
    temperature += annual_temperature * (1 - alpha) + (273 - 1.8) * alpha

    # take the longest corr_length
    corr_length = 4 * (np.maximum(1 - density / 917, density / 917)) / (ssa * 917)

    if microstructure_model == "teubner_strey":
        q_factor = 0.92
        microstructure = dict(corr_length=corr_length, repeat_distance=2 * np.pi * q_factor * corr_length)
    elif microstructure_model == "sticky_hard_spheres":
        microstructure = dict(stickiness=0.14, radius=3 / (ssa * 917))
    elif microstructure_model == "extended_teubner_strey":
        microstructure = dict(corr_length=corr_length, q_factor=4, case=1)
    else:
        raise Exception("microstructure_model!!")

    # water substrate
    sp = make_snowpack(thickness=thickness, density=density, temperature=temperature,
                       microstructure_model=microstructure_model, **microstructure,
                       ssa_value=ssa)  # just to keep it in memory, not used with TS microstructure

    # add the water substrate (does NOT account for the pressure on temperature)
    if add_water_substrate:
        sp += make_ice_column("firstyear", [], 273.15 - 1.8, "", add_water_substrate=True)

    #print("eps=", sp.substrate.permittivity_model(1.4e9, 273))
    # modele inversion !! VERY important !
    if microstructure_model == "sticky_hard_spheres":
        sp.layers = [lay.inverted_medium() if lay.density > 917/2 else lay for lay in sp.layers]

    return sp


def prepare_snowpack_v2(annual_temperature, season_temperature, ice_thickness,
                        density=[350, 500, 600, 850], mwgs=[0.1e-3, 0.2e-3, 0.3e-3, 0.4e-3],
                        ice_layer_density=[1, 1, 1, 1], microstructure_model="exponential", resolution="medium", add_water_substrate=True):

    if resolution == "high":
        z = np.arange(0, 30, 0.1)
    elif resolution == "medium":
        z = np.hstack((np.arange(0, 1, 0.1), np.arange(1, 10, 0.3), np.arange(10, 30, 1)))
    elif resolution == "low":
        z = np.hstack((np.arange(0, 2, 0.2), np.arange(2, 10, 0.4), np.arange(10, 30, 1)))
    else:
        raise Exception("Unknown resolution")

    # go to the closeoff (assumed at 60m) and then to the ice bottom (-> ocean for ice shelves)
    z = np.hstack((z, np.arange(30, 61, 10), [ice_thickness]))
    z_mid = (z[1:] + z[:-1]) / 2
    ldepth = depth_v2 + [60, ice_thickness]

    # set deep firn properties
    density = np.append(density, [912, 912])
    mwgs = np.append(mwgs, [0.003e-3, 0.003e-3])  # very small bubbles. To avoid 0.

    # interpolate density and grain size
    mwgs = np.interp(z_mid, ldepth, mwgs)
    density = np.interp(z_mid, ldepth, density)

    # interpolate ice_layer density
    ice_layer_density = np.interp(z_mid, ldepth, np.append(ice_layer_density, [0, 0]))

    # compute where ice layers must be
    ice_layers = np.ceil(np.cumsum(ice_layer_density * np.diff(z)))
    ice_layers_z = z_mid[:-1][np.diff(ice_layers) > 0]

    # insert the ice layers
    ice_layer_thickness = 0.01  # must be less t
    for ice_layer_z in ice_layers_z:
        i = np.searchsorted(z, ice_layer_z, side='left')
        if i > 0:
            assert z[i - 1] < ice_layer_z < z[i]
        #print(z[i:i+2], ice_layer_z)
        #print("don't forget this is temp")
        # continue
        z = np.insert(z, i, [ice_layer_z, ice_layer_z + ice_layer_thickness])
        assert np.all(np.diff(z) > 0)
        density = np.insert(density, i, [912, density[i]])
        mwgs = np.insert(mwgs, i, [0.03e-3, mwgs[i]])

    z_mid = (z[1:] + z[:-1]) / 2
    thickness = np.diff(z)

    # clip
    density = np.minimum(density, 912)  # the ice must be a bit bubbly !!

    alpha = z_mid / ice_thickness
    temperature = (season_temperature - annual_temperature) * np.exp(-z_mid / 2.0)
    temperature += annual_temperature * (1 - alpha) + (273 - 1.8) * alpha

    if microstructure_model == "exponential":
        microstructure = dict(corr_length=mwgs)
    else:
        raise Exception("microstructure_model!!")

    # water substrate
    sp = make_snowpack(thickness=thickness, density=density, temperature=temperature,
                       microstructure_model=microstructure_model, **microstructure,
                       )


    # add the water substrate (does NOT account for the pressure on temperature)
    if add_water_substrate:
        sp += make_ice_column("firstyear", [], 273.15 - 1.8, "", add_water_substrate=True)
        if (sp.layers[-1].thickness == 0) and (sp.nlayer > 1):
            # this is kinda bug that needs to be solved in SMRT but is there.
            # Remove this layers
            sp.layers.pop()
            sp.interfaces.pop()

    return sp


def prepare_snowpack_from_params(params, annual_temperature, season_temperature, ice_thickness, **kwargs):
    ssa = params[0:4]
    density = params[4:8]
    sigma_density = params[8:12]

    sp = prepare_snowpack(annual_temperature=annual_temperature, season_temperature=season_temperature,
                          ice_thickness=ice_thickness,
                          density=density, ssa=ssa,
                          sigma_density=sigma_density,
                          microstructure_model="sticky_hard_spheres", **kwargs)
    return sp


def prepare_snowpack_from_params_v2(params, annual_temperature, season_temperature, ice_thickness, **kwargs):
    mwgs = params[0:4]
    density = params[4:8]
    ice_layer_density = params[8:12]

    sp = prepare_snowpack_v2(annual_temperature=annual_temperature, season_temperature=season_temperature,
                             ice_thickness=ice_thickness,
                             density=density, mwgs=mwgs,
                             ice_layer_density=ice_layer_density,
                             microstructure_model="exponential", **kwargs)
    return sp


def prepare_best_snowpack(filename, annual_temperature, season_temperature, ice_thickness, **kwargs):
    params = read_best_parameters(filename)
    return prepare_snowpack_from_params(params, annual_temperature, season_temperature, ice_thickness, **kwargs)


# def prepare_good_snowpacks(filename, annual_temperature, season_temperature, ice_thickness, size, unique=True, **kwargs):
#    params_list = read_good_parameters(filename)[:size]
#
#    for i, params in pd.DataFrame(params_list).groupby(list(range(12)):


#    return [prepare_snowpack_from_params(params, annual_temperature, season_temperature, ice_thickness, **kwargs) for i, params in params_list]


def read_best_parameters(filename, **kwargs):
    return read_good_parameters(filename, n=1, **kwargs)[0]


def read_good_parameters(filename, n=None, burn=0.5, return_log_ps=False):
    ds = xr.open_dataset(filename)

    if 'iteration' in ds.coords:
        ds = ds.rename({'iteration': 'draw'})

    ndraws = ds.dims['draw']

    nburn = int(ndraws * burn)  # burn the first samples
    burnt_ds = ds.sel(draw=slice(nburn, None)).stack(n=('chain', 'draw'))

    flat_sampled_params = burnt_ds['samples']
    flat_log_ps = burnt_ds['log_p']

    #ilgood = pd.DataFrame(flat_sampled_params.values).groupby(list(range(12))).size().argsort()

    ilgood = np.argsort(flat_log_ps.values)

    if n is None:
        n = len(ilgood)
    ilgood = ilgood[-1:-(n+1):-1]

    samples = [flat_sampled_params.values[:, i] for i in ilgood]
    log_ps = [flat_log_ps.values[i] for i in ilgood]

    if return_log_ps:
        return samples, log_ps
    else:
        return samples


def insert_layer(z, sp):

    iz = np.searchsorted(sp.layer_depths, z)
    h = sp.layer_depths[iz] - z

    assert h >= 0

    if h > 0 and h < sp.layers[iz].thickness:
        sp.layers.insert(iz, copy.deepcopy(sp.layers[iz]))
        sp.interfaces.insert(iz, copy.deepcopy(sp.interfaces[iz]))

        sp.layers[iz].thickness -= h
        sp.layers[iz + 1].thickness = h

        assert sp.layers[iz].thickness > 0
        assert sp.layers[iz + 1].thickness > 0

        return iz + 1
    elif h == 0:
        return iz + 1
    else:
        return iz


def prepare_wet_snowpacks(snowpack, zmin, zmax, total_liquid_water):

    lsp = snowpack.deepcopy()

    izmin = insert_layer(zmin, lsp)
    izmax = insert_layer(zmax, lsp)

    water_density = total_liquid_water / (zmax - zmin)  # kg/m3

    for i in range(izmin, izmax):

        dry_density = lsp.layers[i].density
        water_volume = water_density / DENSITY_OF_WATER
        ice_volume = dry_density / DENSITY_OF_ICE
        if water_volume + ice_volume > 1:
            # the layer is saturated, we remove some ice...
            ice_volume = 1 - water_volume
            dry_density = ice_volume * DENSITY_OF_ICE

        #print(f"{dry_density} {water_density} {water_volume} {ice_volume}")

        lsp.layers[i].update(density=dry_density + water_density,
                             volumetric_liquid_water=water_volume)

    assert np.allclose(total_liquid_water, compute_total_liquid_water(lsp))

    if total_liquid_water > 0:
        for i in range(izmin, izmax):
            lsp.layers[i].temperature = 273.15

    return lsp


def prepare_saturated_snowpacks(snowpack, zmin, zmax, water_proportion=None, dry_density=None):
    lsp = snowpack.deepcopy()

    izmin = insert_layer(zmin, lsp)
    izmax = insert_layer(zmax, lsp)

    for i in range(izmin, izmax):

        if dry_density is not None:
            ice_volume = np.clip(dry_density / DENSITY_OF_ICE, 0, 1)
        else:
            ice_volume = np.clip(lsp.layers[i].density / DENSITY_OF_ICE, 0, 1)

        lsp.layers[i].density = DENSITY_OF_WATER * (1 - ice_volume) + DENSITY_OF_ICE * ice_volume
        lsp.layers[i].volumetric_liquid_water = 1 - ice_volume

        lsp.layers[i].permittivity_model = (water_permittivity, ice_permittivity_maetzler06)

    for i in range(0, izmax):
        lsp.layers[i].temperature = 273.15

    return lsp


def total_liquid_water(sp):
    lw = np.array(sp.profile('liquid_water'))
    water_density = np.array(sp.profile('density')) * lw / ((1 - lw) * DENSITY_OF_ICE/DENSITY_OF_WATER + lw)

    return np.sum(water_density * np.array(sp.layer_thicknesses))


compute_total_liquid_water = total_liquid_water


def show_snowpack(sp, show_corr_length=False):
    n = 4 if show_corr_length else 3
    f, axs = plt.subplots(1, n)
    z = sp.mid_layer_depths[:-1]
    layers = sp.layers[:-1]

    axs[0].plot(sp.layer_densities[:-1], -z, '.-')
    axs[0].set_xlabel("Density (kg m$^{-3}$)")
    if hasattr(sp.layers[0], "ssa_value"):
        axs[1].plot(sp.profile("ssa_value")[:-1], -z)
        axs[1].set_xlabel("SSA (m$^{2}$ kg$^{-1}$)")
    else:
        axs[1].plot(sp.profile("corr_length", where="microstructure")[:-1] * 1e3, -z)
        axs[1].set_xlabel("Microwave grain size (mm)")

    if show_corr_length:
        try:
            axs[3].plot(sp.profile("corr_length", where="microstructure")[:-1], -z)
        except AttributeError:
            pass
    axs[2].plot(sp.profile("temperature")[:-1], -z)
    axs[2].set_xlabel("Temperature (K)")


def run_model(sensor, sp, channels=None, error_handling='nan'):

    if sensor == "amsr2":
        sensor = sensor_list.amsr2(channels)
    elif sensor == "smos":
        sensor = sensor_list.smos(channels)
    elif sensor == "ascat":
        sensor = sensor_list.ascat(theta=40)
    else:
        raise Exception("Invalid sensor")

    m = make_model("symsce_torquato21", "dort",
                   # emmodel_options=dict(scaled=False),
                   rtsolver_options=dict(prune_deep_snowpack=5, error_handling=error_handling))
    res = m.run(sensor, sp)

    return res.to_series()


class GoodSnowpacks(object):

    def __init__(self, location, season, version=31):

        if version == 2:
            self.paramsfilename = f"DreamData/smrt2_dream_12params_{location}_{season}.nc"
        elif version == 3:
            self.paramsfilename = f"DreamData/smrt3_dream_13params_{location}_{season}.nc"
        elif version == 31:
            self.paramsfilename = f"DreamData/smrt3-1_dream_13params_{location}_{season}.nc"
        else:
            raise Exception("invalid version")

        jsonfilename = f"DreamData/metadata_{location}_{season}.json"

        self.best_params = read_best_parameters(self.paramsfilename)

        with open(jsonfilename) as f:
            self.metadata = json.load(f)

    def prepare_best_snowpack(self, **kwargs):

        options = dict(annual_temperature=self.metadata['annual_temperature'],
                       season_temperature=self.metadata['season_temperature'],
                       ice_thickness=self.metadata['ice_thickness'])
        options.update(kwargs)

        return prepare_snowpack_from_params_v2(self.best_params[0:12], **options)

    def prepare_good_snowpacks(self, size, as_dataframe, **kwargs):
        """return in the 'size' first top samples the unique parameter sets with their frequency of appearance

"""
        options = dict(annual_temperature=self.metadata['annual_temperature'],
                       season_temperature=self.metadata['season_temperature'],
                       ice_thickness=self.metadata['ice_thickness'])
        options.update(kwargs)

        params_list = read_good_parameters(self.paramsfilename)[:size]
        sps = []
        frequency = []
        for params, group in pd.DataFrame(params_list).groupby(list(range(12))):
            sps.append(prepare_snowpack_from_params_v2(params[0:12], **options))
            frequency.append(len(group))

        if as_dataframe:
            return pd.DataFrame({'snowpack': sps, 'weight': frequency})
        else:
            return sps, frequency

    def run_best_snowpack(self, sensor, channels=None, sp=None, **kwargs):

        if sp is None:
            sp = self.prepare_best_snowpack()

        return run_model(sensor=sensor, sp=sp, channels=channels, **kwargs)


def lband_adjusted_snowpack(snowpacks):

    if isinstance(snowpacks, pd.DataFrame):
        snowpacks2 = snowpacks.copy()
        snowpacks2['snowpack'] = lband_adjusted_snowpack(snowpacks['snowpack'].tolist())
        return snowpacks2
    if not isinstance(snowpacks, Sequence):
        snowpacks = [snowpacks]

    snowpacks2 = copy.deepcopy(snowpacks)
    for sp in snowpacks2:
        for l in sp.layers:
            l.microstructure.corr_length *= 2.8   # empirically determined
    return snowpacks2


def smos_amsr2(channel=None, frequency=None, polarization=None, theta=55):
    """ Configuration for AMSR-2 +SMOS sensor.

    """

    frequency_dict = {
        '01': 1.41e9,
        '06': 6.925e9,
        '07': 7.3e9,
        '10': 10.65e9,
        '19': 18.7e9,
        '23': 23.8e9,
        '37': 36.5e9,
        '89': 89e9}

    return sensor_list.common_conical_pmw("SMOS+AMSR2", frequency_dict, channel=channel, frequency=frequency, theta=theta, name='smos+asmr2')
