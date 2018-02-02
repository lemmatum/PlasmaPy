# coding=utf-8
"""Tests for functions that calculate transport coefficients."""


import numpy as np
import pytest
from astropy import units as u
from plasmapy.atomic.atomic import ion_mass, integer_charge
from plasmapy.utils.exceptions import (PhysicsError, PhysicsWarning,
                                       RelativityWarning, RelativityError)
from plasmapy.physics.parameters import Hall_parameter
from ...constants import c, m_p, m_e, e, mu0

from ..transport import (Coulomb_logarithm,
                         b_perp,
                         impact_parameter,
                         collision_frequency,
                         mean_free_path,
                         Spitzer_resistivity,
                         mobility,
                         Knudsen_number,
                         coupling_parameter,
                         classical_transport,
                         _nondim_tc_e_braginskii,
                         _nondim_tc_i_braginskii,
                         _nondim_tec_braginskii,
                         _nondim_resist_braginskii,
                         _nondim_visc_i_braginskii,
                         _nondim_visc_e_braginskii,
                         _nondim_tc_e_spitzer,
                         _nondim_tec_spitzer,
                         _nondim_resist_spitzer,
                         _nondim_thermal_conductivity,
                         _nondim_resistivity,
                         _nondim_te_conductivity,
                         _nondim_tc_e_ji_held,
                         _nondim_tc_i_ji_held,
                         _nondim_tec_ji_held,
                         _nondim_resist_ji_held,
                         _nondim_visc_e_ji_held,
                         _nondim_visc_i_ji_held,
                         _nondim_viscosity,
                         _check_Z,
                         )


def count_decimal_places(digits):
    '''Return the number of decimal places of the input digit string'''
    integral, _, fractional = digits.partition(".")
    return len(fractional)


class Test_Coulomb_logarithm(object):
    def setup_method(self):
        """initializing parameters for tests """
    def test_Chen_Q_machine(self):
        """
        Tests whether Coulomb logarithm gives value consistent with 
        Chen's Introduction to Plasma Physics and Controlled Fusion
        section 5.6.2 Q-machine example.
        """
        T = 0.2 * u.eV
        T = T.to(u.K, equivalencies=u.temperature_energy())
        n = 1e15 * u.m ** -3
        # factor of np.log(2) corrects for different definitions of thermal
        # velocity. Chen uses v**2 = k * T / m  whereas we use
        # v ** 2 = 2 * k * T / m
        lnLambdaChen = 9.1 + np.log(2) 
        lnLambda = Coulomb_logarithm(T, n, ('e', 'p'))
        testTrue = np.isclose(lnLambda,
                              lnLambdaChen,
                              rtol=1e-1,
                              atol=0.0)
        errStr = ("Q-machine value of Coulomb logarithm should be "
                  f"{lnLambdaChen} and not {lnLambda}.")
        assert testTrue, errStr
    def test_Chen_lab(self):
        """
        Tests whether Coulomb logarithm gives value consistent with 
        Chen's Introduction to Plasma Physics and Controlled Fusion
        section 5.6.2 lab plasma example.
        """
        T = 2 * u.eV
        T = T.to(u.K, equivalencies=u.temperature_energy())
        n = 1e17 * u.m ** -3
        # factor of np.log(2) corrects for different definitions of thermal
        # velocity. Chen uses v**2 = k * T / m  whereas we use
        # v ** 2 = 2 * k * T / m
        lnLambdaChen = 10.2 + np.log(2) 
        lnLambda = Coulomb_logarithm(T, n, ('e', 'p'))
        testTrue = np.isclose(lnLambda,
                              lnLambdaChen,
                              rtol=1e-1,
                              atol=0.0)
        errStr = ("Lab plasma value of Coulomb logarithm should be "
                  f"{lnLambdaChen} and not {lnLambda}.")
        assert testTrue, errStr
    def test_Chen_torus(self):
        """
        Tests whether Coulomb logarithm gives value consistent with 
        Chen's Introduction to Plasma Physics and Controlled Fusion
        section 5.6.2 torus example.
        """
        T = 100 * u.eV
        T = T.to(u.K, equivalencies=u.temperature_energy())
        n = 1e19 * u.m ** -3
        # factor of np.log(2) corrects for different definitions of thermal
        # velocity. Chen uses v**2 = k * T / m  whereas we use
        # v ** 2 = 2 * k * T / m
        lnLambdaChen = 13.7 + np.log(2) 
        lnLambda = Coulomb_logarithm(T, n, ('e', 'p'))
        testTrue = np.isclose(lnLambda,
                              lnLambdaChen,
                              rtol=1e-1,
                              atol=0.0)
        errStr = ("Torus value of Coulomb logarithm should be "
                  f"{lnLambdaChen} and not {lnLambda}.")
        assert testTrue, errStr
    def test_Chen_fusion(self):
        """
        Tests whether Coulomb logarithm gives value consistent with 
        Chen's Introduction to Plasma Physics and Controlled Fusion
        section 5.6.2 fusion reactor example.
        """
        T = 1e4 * u.eV
        T = T.to(u.K, equivalencies=u.temperature_energy())
        n = 1e21 * u.m ** -3
        # factor of np.log(2) corrects for different definitions of thermal
        # velocity. Chen uses v**2 = k * T / m  whereas we use
        # v ** 2 = 2 * k * T / m
        lnLambdaChen = 16 + np.log(2) 
        lnLambda = Coulomb_logarithm(T, n, ('e', 'p'))
        testTrue = np.isclose(lnLambda,
                              lnLambdaChen,
                              rtol=1e-1,
                              atol=0.0)
        errStr = ("Fusion reactor value of Coulomb logarithm should be "
                  f"{lnLambdaChen} and not {lnLambda}.")
        assert testTrue, errStr
    def test_Chen_laser(self):
        """
        Tests whether Coulomb logarithm gives value consistent with 
        Chen's Introduction to Plasma Physics and Controlled Fusion
        section 5.6.2 laser plasma example.
        """
        T = 1e3 * u.eV
        T = T.to(u.K, equivalencies=u.temperature_energy())
        n = 1e27 * u.m ** -3
        # factor of np.log(2) corrects for different definitions of thermal
        # velocity. Chen uses v**2 = k * T / m  whereas we use
        # v ** 2 = 2 * k * T / m
        lnLambdaChen = 6.8 + np.log(2) 
        lnLambda = Coulomb_logarithm(T, n, ('e', 'p'))
        testTrue = np.isclose(lnLambda,
                              lnLambdaChen,
                              rtol=1e-1,
                              atol=0.0)
        errStr = ("Laser plasma value of Coulomb logarithm should be "
                  f"{lnLambdaChen} and not {lnLambda}.")
        assert testTrue, errStr
    def test_relativity_warn(self):
        """Tests whether relativity warning is raised at high velocity."""
        with pytest.warns(RelativityWarning):
            Coulomb_logarithm(1e5 * u.K, 1 * u.m**-3, ('e', 'p'), V = 0.9 * c)
    def test_relativity_error(self):
        """Tests whether relativity error is raised at light speed."""
        with pytest.raises(RelativityError):
            Coulomb_logarithm(1e5 * u.K, 1 * u.m**-3, ('e', 'p'), V = 1.1 * c)
    def test_unit_conversion_error(self):
        """
        Tests whether unit conversion error is raised when arguments
        are given with incorrect units.
        """
        with pytest.raises(u.UnitConversionError):
            Coulomb_logarithm(1e5 * u.g, 1 * u.m**-3,
                          ('e', 'p'), V = 29979245 * u.m / u.s)
    def test_single_particle_error(self):
        """
        Tests whether an error is raised if only a single particle is given.
        """
        with pytest.raises(ValueError):
            Coulomb_logarithm(1 * u.K, 5 * u.m**-3, ('e'))
    def test_invalid_particle_error(self):
        """
        Tests whether an error is raised when an invalid particle name
        is given.
        """
        with pytest.raises(ValueError):
            Coulomb_logarithm(1 * u.K, 5 * u.m**-3, ('e', 'g'))


class Test_b_perp(object):
    def setup_method(self):
        """initializing parameters for tests """
        self.T = 11604 * u.K
        self.particles = ('e', 'p')
        self.V = 1e4 * u.km / u.s
        self.True1 = 7.200146594293746e-10
    def test_known1(self):
        """
        Test for known value.
        """
        methodVal = b_perp(self.T,
                           self.particles,
                           V=np.nan*u.m/u.s)
        testTrue = np.isclose(self.True1,
                              methodVal.si.value,
                              rtol=1e-1,
                              atol=0.0)
        errStr = ("Distance of closest approach for 90 degree Coulomb "
                  f"collision, b_perp, should be {self.True1} and "
                  f"not {methodVal}.")
        assert testTrue, errStr
    def test_fail1(self):
        """
        Tests if test_known1() would fail if we slightly adjusted the
        value comparison by some quantity close to numerical error.
        """
        fail1 = self.True1 + 1e-15
        methodVal = b_perp(self.T,
                           self.particles,
                           V=np.nan*u.m/u.s)
        testTrue = not np.isclose(methodVal.si.value,
                                  fail1,
                                  rtol=1e-16,
                                  atol=0.0)
        errStr = (f"b_perp value test gives {methodVal} and "
                  f"should not be equal to {fail1}.")
        assert testTrue, errStr


class Test_impact_parameter(object):
    def setup_method(self):
        """initializing parameters for tests """
        self.T = 11604 * u.K
        self.n_e = 1e17 * u.cm ** -3
        self.particles = ('e', 'p')
        self.z_mean = 2.5
        self.V = 1e4 * u.km / u.s
        self.True1 = np.array([7.200146594293746e-10, 2.3507660003984624e-08])
    def test_known1(self):
        """
        Test for known value.
        """
        methodVal = impact_parameter(self.T,
                                     self.n_e,
                                     self.particles,
                                     z_mean=np.nan*u.dimensionless_unscaled,
                                     V=np.nan*u.m/u.s,
                                     method="classical")
        bmin, bmax = methodVal
        methodVal = bmin.si.value, bmax.si.value
        testTrue = np.allclose(self.True1,
                               methodVal,
                               rtol=1e-1,
                               atol=0.0)
        errStr = (f"Impact parameters should be {self.True1} and "
                  f"not {methodVal}.")
        assert testTrue, errStr
    def test_fail1(self):
        """
        Tests if test_known1() would fail if we slightly adjusted the
        value comparison by some quantity close to numerical error.
        """
        fail1 = self.True1 * (1 + 1e-15)
        methodVal = impact_parameter(self.T,
                                     self.n_e,
                                     self.particles,
                                     z_mean=np.nan*u.dimensionless_unscaled,
                                     V=np.nan*u.m/u.s,
                                     method="classical")
        bmin, bmax = methodVal
        methodVal = bmin.si.value, bmax.si.value
        testTrue = not np.allclose(methodVal,
                                   fail1,
                                   rtol=1e-16,
                                   atol=0.0)
        errStr = (f"Impact parameter value test gives {methodVal} and "
                  f"should not be equal to {fail1}.")
        assert testTrue, errStr


class Test_collision_frequency(object):
    def setup_method(self):
        """initializing parameters for tests """
        self.T = 11604 * u.K
        self.n = 1e17 * u.cm ** -3
        self.particles = ('e', 'p')
        self.z_mean = 2.5
        self.V = 1e4 * u.km / u.s
        self.True1 = 1.3468281539854646e12
    def test_known1(self):
        """
        Test for known value.
        """
        methodVal = collision_frequency(self.T,
                                        self.n,
                                        self.particles,
                                        z_mean=np.nan*u.dimensionless_unscaled,
                                        V=np.nan*u.m/u.s,
                                        method="classical")
        testTrue = np.isclose(self.True1,
                              methodVal.si.value,
                              rtol=1e-1,
                              atol=0.0)
        errStr = (f"Collision frequency should be {self.True1} and "
                  f"not {methodVal}.")
        assert testTrue, errStr
    def test_fail1(self):
        """
        Tests if test_known1() would fail if we slightly adjusted the
        value comparison by some quantity close to numerical error.
        """
        fail1 = self.True1 * (1 + 1e-15)
        methodVal = collision_frequency(self.T,
                                        self.n,
                                        self.particles,
                                        z_mean=np.nan*u.dimensionless_unscaled,
                                        V=np.nan*u.m/u.s,
                                        method="classical")
        testTrue = not np.isclose(methodVal.si.value,
                                  fail1,
                                  rtol=1e-16,
                                  atol=0.0)
        errStr = (f"Collision frequency value test gives {methodVal} and "
                  f"should not be equal to {fail1}.")
        assert testTrue, errStr


class Test_mean_free_path(object):
    def setup_method(self):
        """initializing parameters for tests """
        self.T = 11604 * u.K
        self.n_e = 1e17 * u.cm ** -3
        self.particles = ('e', 'p')
        self.z_mean = 2.5
        self.V = 1e4 * u.km / u.s
        self.True1 = 4.4047571877932046e-07
    def test_known1(self):
        """
        Test for known value.
        """
        methodVal = mean_free_path(self.T,
                                   self.n_e,
                                   self.particles,
                                   z_mean=np.nan*u.dimensionless_unscaled,
                                   V=np.nan*u.m/u.s,
                                   method="classical")
        testTrue = np.isclose(self.True1,
                              methodVal.si.value,
                              rtol=1e-1,
                              atol=0.0)
        errStr = (f"Mean free path should be {self.True1} and "
                  f"not {methodVal}.")
        assert testTrue, errStr
    def test_fail1(self):
        """
        Tests if test_known1() would fail if we slightly adjusted the
        value comparison by some quantity close to numerical error.
        """
        fail1 = self.True1 * (1 + 1e-15)
        methodVal = mean_free_path(self.T,
                                   self.n_e,
                                   self.particles,
                                   z_mean=np.nan*u.dimensionless_unscaled,
                                   V=np.nan*u.m/u.s,
                                   method="classical")
        testTrue = not np.isclose(methodVal.si.value,
                                  fail1,
                                  rtol=1e-16,
                                  atol=0.0)
        errStr = (f"Mean free path value test gives {methodVal} and "
                  f"should not be equal to {fail1}.")
        assert testTrue, errStr


class Test_Spitzer_resistivity(object):
    def setup_method(self):
        """initializing parameters for tests """
        self.T = 11604 * u.K
        self.n = 1e12 * u.cm ** -3
        self.particles = ('e', 'p')
        self.z_mean = 2.5
        self.V = 1e4 * u.km / u.s
        self.True1 = 1.2665402649805445e-3
    def test_known1(self):
        """
        Test for known value.
        """
        methodVal = Spitzer_resistivity(self.T,
                                        self.n,
                                        self.particles,
                                        z_mean=np.nan*u.dimensionless_unscaled,
                                        V=np.nan*u.m/u.s,
                                        method="classical")
        testTrue = np.isclose(self.True1,
                              methodVal.si.value,
                              rtol=1e-1,
                              atol=0.0)
        errStr = (f"Spitzer resistivity should be {self.True1} and "
                  f"not {methodVal}.")
        assert testTrue, errStr
    def test_fail1(self):
        """
        Tests if test_known1() would fail if we slightly adjusted the
        value comparison by some quantity close to numerical error.
        """
        fail1 = self.True1 * (1 + 1e-15)
        methodVal = Spitzer_resistivity(self.T,
                                        self.n,
                                        self.particles,
                                        z_mean=np.nan*u.dimensionless_unscaled,
                                        V=np.nan*u.m/u.s,
                                        method="classical")
        testTrue = not np.isclose(methodVal.si.value,
                                  fail1,
                                  rtol=1e-16,
                                  atol=0.0)
        errStr = (f"Spitzer resistivity value test gives {methodVal} and "
                  f"should not be equal to {fail1}.")
        assert testTrue, errStr


class Test_mobility(object):
    def setup_method(self):
        """initializing parameters for tests """
        self.T = 11604 * u.K
        self.n_e = 1e17 * u.cm ** -3
        self.particles = ('e', 'p')
        self.z_mean = 2.5
        self.V = 1e4 * u.km / u.s
        self.True1 = 0.13066090887074902
    def test_known1(self):
        """
        Test for known value.
        """
        methodVal = mobility(self.T,
                             self.n_e,
                             self.particles,
                             z_mean=np.nan*u.dimensionless_unscaled,
                             V=np.nan*u.m/u.s,
                             method="classical")
        testTrue = np.isclose(self.True1,
                              methodVal.si.value,
                              rtol=1e-1,
                              atol=0.0)
        errStr = (f"Mobility should be {self.True1} and "
                  f"not {methodVal}.")
        assert testTrue, errStr
    def test_fail1(self):
        """
        Tests if test_known1() would fail if we slightly adjusted the
        value comparison by some quantity close to numerical error.
        """
        fail1 = self.True1 * (1 + 1e-15)
        methodVal = mobility(self.T,
                             self.n_e,
                             self.particles,
                             z_mean=np.nan*u.dimensionless_unscaled,
                             V=np.nan*u.m/u.s,
                             method="classical")
        testTrue = not np.isclose(methodVal.si.value,
                                  fail1,
                                  rtol=1e-16,
                                  atol=0.0)
        errStr = (f"Mobility value test gives {methodVal} and "
                  f"should not be equal to {fail1}.")
        assert testTrue, errStr


class Test_Knudsen_number(object):
    def setup_method(self):
        """initializing parameters for tests """
        self.length = 1 * u.nm
        self.T = 11604 * u.K
        self.n_e = 1e17 * u.cm ** -3
        self.particles = ('e', 'p')
        self.z_mean = 2.5
        self.V = 1e4 * u.km / u.s
        self.True1 = 440.4757187793204
    def test_known1(self):
        """
        Test for known value.
        """
        methodVal = Knudsen_number(self.length,
                                   self.T,
                                   self.n_e,
                                   self.particles,
                                   z_mean=np.nan*u.dimensionless_unscaled,
                                   V=np.nan*u.m/u.s,
                                   method="classical")
        testTrue = np.isclose(self.True1,
                              methodVal,
                              rtol=1e-1,
                              atol=0.0)
        errStr = (f"Knudsen number should be {self.True1} and "
                  f"not {methodVal}.")
        assert testTrue, errStr
    def test_fail1(self):
        """
        Tests if test_known1() would fail if we slightly adjusted the
        value comparison by some quantity close to numerical error.
        """
        fail1 = self.True1 * (1 + 1e-15)
        methodVal = Knudsen_number(self.length,
                                   self.T,
                                   self.n_e,
                                   self.particles,
                                   z_mean=np.nan*u.dimensionless_unscaled,
                                   V=np.nan*u.m/u.s,
                                   method="classical")
        testTrue = not np.isclose(methodVal,
                                  fail1,
                                  rtol=0.0,
                                  atol=1e-16)
        errStr = (f"Knudsen number value test gives {methodVal} and "
                  f"should not be equal to {fail1}.")
        assert testTrue, errStr


class Test_coupling_parameter(object):
    def setup_method(self):
        """initializing parameters for tests """
        self.T = 11604 * u.K
        self.n_e = 1e21 * u.cm ** -3
        self.particles = ('e', 'p')
        self.z_mean = 2.5
        self.V = 1e4 * u.km / u.s
        self.True1 = 2.3213156755481195
    def test_known1(self):
        """
        Test for known value.
        """
        methodVal = coupling_parameter(self.T,
                                       self.n_e,
                                       self.particles,
                                       z_mean=np.nan*u.dimensionless_unscaled,
                                       V=np.nan*u.m/u.s,
                                       method="classical")
        testTrue = np.isclose(self.True1,
                              methodVal,
                              rtol=1e-1,
                              atol=0.0)
        errStr = (f"Coupling parameter should be {self.True1} and "
                  f"not {methodVal}.")
        assert testTrue, errStr
    def test_fail1(self):
        """
        Tests if test_known1() would fail if we slightly adjusted the
        value comparison by some quantity close to numerical error.
        """
        fail1 = self.True1 * (1 + 1e-15)
        methodVal = coupling_parameter(self.T,
                                       self.n_e,
                                       self.particles,
                                       z_mean=np.nan*u.dimensionless_unscaled,
                                       V=np.nan*u.m/u.s,
                                       method="classical")
        testTrue = not np.isclose(methodVal,
                                  fail1,
                                  rtol=1e-16,
                                  atol=0.0)
        errStr = (f"Coupling parameter value test gives {methodVal} and "
                  f"should not be equal to {fail1}.")
        assert testTrue, errStr


# test class for classical_transport class:
class Test_classical_transport:

    def setup_method(self):
        """set up some initial values for tests"""
        u.set_enabled_equivalencies(u.temperature_energy())
        self.T_e = 1000 * u.eV
        self.n_e = 2e13 / u.cm ** 3
        self.ion_particle = 'D +1'
        self.m_i = ion_mass(self.ion_particle)
        self.Z = integer_charge(self.ion_particle)
        self.T_i = self.T_e
        self.n_i = self.n_e / self.Z
        self.B = 0.01 * u.T
        self.coulomb_log_val_ei = 17
        self.coulomb_log_val_ii = 17
        self.hall_e = None
        self.hall_i = None
        self.V_ei = None
        self.V_ii = None
        self.mu = m_e / self.m_i
        self.theta = self.T_e / self.T_i
        self.model = 'Braginskii'
        self.field_orientation = 'all'
        self.ct = classical_transport(
            T_e=self.T_e,
            n_e=self.n_e,
            T_i=self.T_i,
            n_i=self.n_i,
            ion_particle=self.ion_particle,
            Z=self.Z,
            B=self.B,
            model=self.model,
            field_orientation=self.field_orientation,
            coulomb_log_ei=self.coulomb_log_val_ei,
            coulomb_log_ii=self.coulomb_log_val_ii,
            V_ei=self.V_ei,
            V_ii=self.V_ii,
            hall_e=self.hall_e,
            hall_i=self.hall_i,
            mu=self.mu,
            theta=self.theta,
        )
        self.all_variables = self.ct.all_variables()

    def test_spitzer_vs_formulary(self):
        """Spitzer resistivity should agree with approx. in NRL formulary"""
        ct2 = classical_transport(T_e=self.T_e,
                                  n_e=self.n_e,
                                  T_i=self.T_i,
                                  n_i=self.n_i,
                                  ion_particle=self.ion_particle,
                                  model='spitzer',
                                  field_orientation='perp')
        alpha_spitzer_perp_NRL = (1.03e-4 * ct2.Z *
                                  ct2.coulomb_log_ei *
                                  ((ct2.T_e).to(u.eV)).value ** (-3 / 2) *
                                  u.Ohm * u.m)
        assert(np.isclose(ct2.resistivity().value,
                          alpha_spitzer_perp_NRL.value, rtol=2e-2))

    def test_resistivity_units(self):
        """output should be a Quantity with units of Ohm m"""
        assert(self.ct.resistivity().unit == u.Ohm * u.m)

    def test_thermoelectric_conductivity_units(self):
        """output should be a Quantity with units of dimensionless"""
        assert(self.ct.thermoelectric_conductivity().unit == u.m / u.m)

    def test_ion_thermal_conductivity_units(self):
        """output should be Quantity with units of W / (m K)"""
        assert(self.ct.ion_thermal_conductivity().unit == u.W / u.m / u.K)

    def test_electron_thermal_conductivity_units(self):
        """output should be Quantity with units of W / (m K)"""
        assert(self.ct.electron_thermal_conductivity().unit == u.W / u.m / u.K)

    def test_ion_viscosity_units(self):
        """output should be Quantity with units of Pa s """
        assert(self.ct.ion_viscosity().unit == u.Pa * u.s)

    def test_electron_viscosity_units(self):
        """output should be Quantity with units of Pa s"""
        assert(self.ct.electron_viscosity().unit == u.Pa * u.s)

    def test_particle_mass(self):
        """should raise ValueError if particle mass not found"""
        with pytest.raises(ValueError):
            ct2 = classical_transport(T_e=self.T_e,
                                      n_e=self.n_e,
                                      T_i=self.T_i,
                                      n_i=self.n_i,
                                      ion_particle='empty moment',
                                      Z=1)

    def test_particle_charge_state(self):
        """should raise ValueError if particle charge state not found"""
        with pytest.raises(ValueError):
            ct2 = classical_transport(T_e=self.T_e,
                                      n_e=self.n_e,
                                      T_i=self.T_i,
                                      n_i=self.n_i,
                                      ion_particle='empty moment',
                                      m_i=m_p)

    def test_Z_checks(self):
        """should raise ValueError if Z is negative"""
        with pytest.raises(ValueError):
            ct2 = classical_transport(T_e=self.T_e,
                                      n_e=self.n_e,
                                      T_i=self.T_i,
                                      n_i=self.n_i,
                                      ion_particle=self.ion_particle,
                                      Z=-1)

    def test_coulomb_log_warnings(self):
        """should warn PhysicsWarning if coulomb log is near 1"""
        with pytest.warns(PhysicsWarning):
            ct2 = classical_transport(T_e=self.T_e,
                                      n_e=self.n_e,
                                      T_i=self.T_i,
                                      n_i=self.n_i,
                                      ion_particle=self.ion_particle,
                                      coulomb_log_ii=1.3)

        with pytest.warns(PhysicsWarning):
            ct2 = classical_transport(T_e=self.T_e,
                                      n_e=self.n_e,
                                      T_i=self.T_i,
                                      n_i=self.n_i,
                                      ion_particle=self.ion_particle,
                                      coulomb_log_ei=1.3)

    def test_coulomb_log_errors(self):
        """should raise PhysicsError if coulomb log is < 1"""
        with pytest.raises(PhysicsError), pytest.warns(PhysicsWarning):
            ct2 = classical_transport(T_e=self.T_e,
                                      n_e=self.n_e,
                                      T_i=self.T_i,
                                      n_i=self.n_i,
                                      ion_particle=self.ion_particle,
                                      coulomb_log_ii=0.3)

        with pytest.raises(PhysicsError), pytest.warns(PhysicsWarning):
            ct2 = classical_transport(T_e=self.T_e,
                                      n_e=self.n_e,
                                      T_i=self.T_i,
                                      n_i=self.n_i,
                                      ion_particle=self.ion_particle,
                                      coulomb_log_ei=0.3)

    def test_coulomb_log_calc(self):
        """if no coulomb logs are input, they should be calculated"""
        ct2 = classical_transport(T_e=self.T_e,
                                  n_e=self.n_e,
                                  T_i=self.T_i,
                                  n_i=self.n_i,
                                  ion_particle=self.ion_particle)
        cl_ii = Coulomb_logarithm(self.T_i,
                                  self.n_e,
                                  [self.ion_particle, self.ion_particle],
                                  self.V_ii)
        cl_ei = Coulomb_logarithm(self.T_e,
                                  self.n_e,
                                  ['e', self.ion_particle],
                                  self.V_ei)
        assert(cl_ii == ct2.coulomb_log_ii)
        assert(cl_ei == ct2.coulomb_log_ei)

    def test_hall_calc(self):
        """if no hall parameters are input, they should be calculated"""
        ct2 = classical_transport(T_e=self.T_e,
                                  n_e=self.n_e,
                                  T_i=self.T_i,
                                  n_i=self.n_i,
                                  ion_particle=self.ion_particle)
        hall_i = Hall_parameter(ct2.n_i,
                                ct2.T_i,
                                ct2.B,
                                ct2.ion_particle,
                                ct2.ion_particle,
                                ct2.coulomb_log_ii,
                                ct2.V_ii)
        hall_e = Hall_parameter(
            ct2.n_e,
            ct2.T_e,
            ct2.B,
            ct2.e_particle,
            ct2.ion_particle,
            ct2.coulomb_log_ei,
            ct2.V_ei)
        assert(hall_i == ct2.hall_i)
        assert(hall_e == ct2.hall_e)

    def test_invalid_model(self):
        with pytest.raises(ValueError):
            ct2 = classical_transport(T_e=self.T_e,
                                      n_e=self.n_e,
                                      T_i=self.T_i,
                                      n_i=self.n_i,
                                      ion_particle=self.ion_particle,
                                      model="standard")

    def test_invalid_field(self):
        with pytest.raises(ValueError):
            ct2 = classical_transport(T_e=self.T_e,
                                      n_e=self.n_e,
                                      T_i=self.T_i,
                                      n_i=self.n_i,
                                      ion_particle=self.ion_particle,
                                      field_orientation='to the left')

    def test_precalculated_parameters(self):
        ct2 = classical_transport(T_e=self.T_e,
                                  n_e=self.n_e,
                                  T_i=self.T_i,
                                  n_i=self.n_i,
                                  ion_particle=self.ion_particle,
                                  hall_i=0,
                                  hall_e=0)
        assert np.isclose(ct2.resistivity(),
                          2.8184954e-8 * u.Ohm * u.m,
                          atol=1e-6 * u.Ohm * u.m)

    @pytest.mark.parametrize("model, method, field_orientation, expected", [
        ("ji-held", "resistivity", "all", 3),
        ("ji-held", "thermoelectric_conductivity", "all", 3),
        ("ji-held", "electron_thermal_conductivity", "all", 3),
        ("ji-held", "ion_thermal_conductivity", "all", 3),
        ("spitzer", "resistivity", "all", 2),
    ])
    def test_number_of_returns(self, model, method, field_orientation,
                               expected):
        ct2 = classical_transport(T_e=self.T_e,
                                  n_e=self.n_e,
                                  T_i=self.T_i,
                                  n_i=self.n_i,
                                  ion_particle=self.ion_particle,
                                  model=model,
                                  field_orientation=field_orientation)
        method_to_call = getattr(ct2, method)
        assert(np.size(method_to_call()) == expected)

    @pytest.mark.parametrize("model, expected", [
        ("ji-held", 2.77028546e-8 * u.Ohm * u.m),
        ("spitzer", 2.78349687e-8 * u.Ohm * u.m),
        ("braginskii", 2.78349687e-8 * u.Ohm * u.m)
    ])
    def test_resistivity_by_model(self, model, expected):
        ct2 = classical_transport(T_e=self.T_e,
                                  n_e=self.n_e,
                                  T_i=self.T_i,
                                  n_i=self.n_i,
                                  ion_particle=self.ion_particle,
                                  model=model)
        assert np.isclose(ct2.resistivity(), expected, atol=1e-6 * u.Ohm * u.m)

    @pytest.mark.parametrize("model, expected", [
        ("ji-held", 0.702 * u.s / u.s),
        ("spitzer", 0.69944979 * u.s / u.s),
        ("braginskii", 0.711084 * u.s / u.s)
    ])
    def test_thermoelectric_conductivity_by_model(self, model, expected):
        ct2 = classical_transport(T_e=self.T_e,
                                  n_e=self.n_e,
                                  T_i=self.T_i,
                                  n_i=self.n_i,
                                  ion_particle=self.ion_particle,
                                  model=model)
        assert np.isclose(ct2.thermoelectric_conductivity(), expected,
                          atol=1e-6 * u.s / u.s)

    @pytest.mark.parametrize("model, expected", [
        ("ji-held",
         np.array([0.07674402, 0.07674402, 0.07674402, 0, 0]) * u.Pa * u.s),
        ("braginskii",
         np.array([0.07674402, 0.07671874, 0.07671874, 0, 0]) * u.Pa * u.s)
    ])
    def test_electron_viscosity_by_model(self, model, expected):
        ct2 = classical_transport(T_e=self.T_e,
                                  n_e=self.n_e,
                                  T_i=self.T_i,
                                  n_i=self.n_i,
                                  ion_particle=self.ion_particle,
                                  model=model)
        assert np.allclose(ct2.electron_viscosity(), expected,
                           atol=1e-6 * u.Pa * u.s)

    @pytest.mark.parametrize("model, expected", [
        ("ji-held",
         np.array([7.96226452, 7.96226452, 7.96226452, 0, 0]) * u.Pa * u.s),
        ("braginskii",
         np.array([7.91936173, 7.89528642, 7.89528642, 0, 0]) * u.Pa * u.s)
    ])
    def test_ion_viscosity_by_model(self, model, expected):
        ct2 = classical_transport(T_e=self.T_e,
                                  n_e=self.n_e,
                                  T_i=self.T_i,
                                  n_i=self.n_i,
                                  ion_particle=self.ion_particle,
                                  model=model)
        assert np.allclose(ct2.ion_viscosity(), expected,
                           atol=1e-6 * u.Pa * u.s)

    @pytest.mark.parametrize("model, expected", [
        ("ji-held", 5084253.556001088 * u.W / (u.K * u.m)),
        ("spitzer", 5082147.824377487 * u.W / (u.K * u.m)),
        ("braginskii", 5016895.3386957785 * u.W / (u.K * u.m))
    ])
    def test_electron_thermal_conductivity_by_model(self, model, expected):
        ct2 = classical_transport(T_e=self.T_e,
                                  n_e=self.n_e,
                                  T_i=self.T_i,
                                  n_i=self.n_i,
                                  ion_particle=self.ion_particle,
                                  model=model)
        assert np.allclose(ct2.electron_thermal_conductivity(), expected,
                           atol=1e-6 * u.W / (u.K * u.m))

    @pytest.mark.parametrize("model, expected", [
        ("ji-held", 134547.55528106514 * u.W / (u.K * u.m)),
        ("braginskii", 133052.21732349042 * u.W / (u.K * u.m))
    ])
    def test_ion_thermal_conductivity_by_model(self, model, expected):
        ct2 = classical_transport(T_e=self.T_e,
                                  n_e=self.n_e,
                                  T_i=self.T_i,
                                  n_i=self.n_i,
                                  ion_particle=self.ion_particle,
                                  model=model)
        assert np.allclose(ct2.ion_thermal_conductivity(), expected,
                           atol=1e-6 * u.W / (u.K * u.m))

    @pytest.mark.parametrize("key, expected", {
        'resistivity': [2.84304305e-08,
                        5.54447070e-08,
                        1.67853407e-12],
        'thermoelectric_conductivity': [7.11083999e-01,
                                        1.61011272e-09,
                                        2.66496639e-05],
        'electron_thermal_conductivity': [4.91374931e+06,
                                          2.28808496e-03,
                                          6.90324259e+01],
        'electron_viscosity': [7.51661800e-02,
                               5.23617668e-21,
                               2.09447067e-20,
                               1.61841341e-11,
                               3.23682681e-11],
        'ion_thermal_conductivity': [1.41709276e+05,
                                     4.20329493e-02,
                                     6.90323924e+01],
        'ion_viscosity': [8.43463595e+00,
                          8.84513731e-13,
                          3.53805159e-12,
                          2.54483240e-06,
                          5.08966116e-06]}.items())
    def test_dictionary(self, key, expected):
        calculated = self.all_variables[key]
        assert np.allclose(expected, calculated.si.value)


@pytest.mark.parametrize(["particle"], ['e', 'p'])
def test_nondim_thermal_conductivity_unrecognized_model(particle):
    with pytest.raises(ValueError):
        _nondim_thermal_conductivity(1, 1, particle,
                                     'standard model is best model',
                                     'parallel')


def test_nondim_resistivity_unrecognized_model():
    with pytest.raises(ValueError):
        _nondim_resistivity(1, 1, 'e', 'SURPRISE SUPERSYMMETRY', 'parallel')


def test_nondim_te_conductivity_unrecognized_model():
    with pytest.raises(ValueError):
        _nondim_te_conductivity(1, 1, 'e', 'this is not a model',
                                'parallel')


@pytest.mark.parametrize(["particle"], ['e', 'p'])
def test_nondim_viscosity_unrecognized_model(particle):
    with pytest.raises(ValueError):
        _nondim_viscosity(1, 1, particle, 'not a model', 'parallel')


# test class for _nondim_tc_e_braginskii function:
class Test__nondim_tc_e_braginskii:

    def setup_method(self):
        """set up some initial values for tests"""
        self.big_hall = 1000
        self.small_hall = 0

    # values from Braginskii '65
    @pytest.mark.parametrize("Z, field_orientation, expected", [
        (1, 'par', 3.16),  # eq (2.12), table 1
        (2, 'par', 4.9),  # eq (2.12), table 1
        (3, 'par', 6.1),  # eq (2.12), table 1
        (4, 'par', 6.9),  # eq (2.12), table 1
        (np.inf, 'par', 12.5),  # eq (2.12), table 1
    ])
    def test_known_values_par(self, Z, field_orientation, expected):
        """check some known values"""
        kappa_e_hat = _nondim_tc_e_braginskii(self.big_hall,
                                              Z,
                                              field_orientation)
        assert np.isclose(kappa_e_hat, expected, atol=1e-1)

    # values from Braginskii '65
    @pytest.mark.parametrize("Z, field_orientation, expected", [
        (1, 'perp', 4.66),  # eq (2.13), table 1
        (2, 'perp', 4.0),  # eq (2.13), table 1
        (3, 'perp', 3.7),  # eq (2.13), table 1
        (4, 'perp', 3.6),  # eq (2.13), table 1
        (np.inf, 'perp', 3.2),  # eq (2.13),table 1
    ])
    def test_known_values_perp(self, Z, field_orientation, expected):
        """check some known values"""
        kappa_e_hat = _nondim_tc_e_braginskii(self.big_hall,
                                              Z,
                                              field_orientation)
        assert np.isclose(kappa_e_hat * self.big_hall ** 2,
                          expected, atol=1e-1)

    @pytest.mark.parametrize("Z", [1, 2, 3, 4, np.inf])
    def test_unmagnetized(self, Z):
        """confirm perp -> par as B -> 0"""
        kappa_e_hat_par = _nondim_tc_e_braginskii(self.small_hall, Z, 'par')
        kappa_e_hat_perp = _nondim_tc_e_braginskii(self.small_hall, Z, 'perp')
        assert np.isclose(kappa_e_hat_par, kappa_e_hat_perp, rtol=1e-3)

    @pytest.mark.parametrize("Z", [1, 4])
    def test_cross_vs_ji_held(self, Z):
        """cross should roughly agree with ji-held"""
        kappa_e_hat_cross_brag = _nondim_tc_e_braginskii(self.big_hall,
                                                         Z,
                                                         'cross')
        kappa_e_hat_cross_jh = _nondim_tc_e_ji_held(self.big_hall,
                                                    Z,
                                                    'cross')
        assert np.isclose(kappa_e_hat_cross_brag,
                          kappa_e_hat_cross_jh,
                          rtol=2e-2)


# test class for _nondim_tc_i_braginskii function:
class Test__nondim_tc_i_braginskii:

    def setup_method(self):
        """set up some initial values for tests"""
        self.big_hall = 1000
        self.small_hall = 0

    def test_known_values_par(self):
        """check some known values"""
        kappa_i_hat = _nondim_tc_i_braginskii(self.big_hall,
                                              field_orientation='par')
        expected = 3.9  # Braginskii '65 eq (2.15)
        assert np.isclose(kappa_i_hat, expected, atol=1e-1)

    def test_known_values_perp(self):
        """check some known values"""
        kappa_i_hat = _nondim_tc_i_braginskii(self.big_hall,
                                              field_orientation='perp')
        expected = 2.0  # Braginskii '65 eq (2.16)
        assert np.isclose(kappa_i_hat * self.big_hall ** 2,
                          expected,
                          atol=1e-1)

    def test_unmagnetized(self):
        """confirm perp -> par as B -> 0"""
        kappa_i_hat_par = _nondim_tc_i_braginskii(self.small_hall, 'par')
        kappa_i_hat_perp = _nondim_tc_i_braginskii(self.small_hall, 'perp')
        assert np.isclose(kappa_i_hat_par, kappa_i_hat_perp, rtol=1e-3)

    def test_cross_vs_ji_held_K2(self):
        """confirm cross agrees with ji-held model when K=2"""
        kappa_i_hat_brag = _nondim_tc_i_braginskii(self.big_hall, 'cross')
        kappa_i_hat_jh = _nondim_tc_i_ji_held(self.big_hall, 1, 0, 100,
                                              'cross', K=2)
        assert np.isclose(kappa_i_hat_brag, kappa_i_hat_jh, rtol=2e-2)


# test class for _nondim_tec_braginskii function:
class Test__nondim_tec_braginskii:

    def setup_method(self):
        """set up some initial values for tests"""
        self.big_hall = 1000
        self.small_hall = 0

    # values from Braginskii '65
    @pytest.mark.parametrize("Z, field_orientation, expected", [
        (1, 'par', 0.71),  # eq (2.9), table 1
        (2, 'par', 0.9),  # eq (2.9), table 1
        (3, 'par', 1.0),  # eq (2.9), table 1
        (4, 'par', 1.1),  # eq (2.9), table 1
        (np.inf, 'par', 1.5),  # eq (2.9),table 1
    ])
    def test_known_values_par(self, Z, field_orientation, expected):
        """check some known values"""
        beta_hat = _nondim_tec_braginskii(self.big_hall,
                                          Z,
                                          field_orientation)
        assert np.isclose(beta_hat, expected, atol=1e-1)

    @pytest.mark.parametrize("Z", [1, 2, 3, 4, np.inf])
    def test_unmagnetized(self, Z):
        """confirm perp -> par as B -> 0"""
        beta_hat_par = _nondim_tec_braginskii(self.small_hall, Z, 'par')
        beta_hat_perp = _nondim_tec_braginskii(self.small_hall, Z, 'perp')
        assert np.isclose(beta_hat_par, beta_hat_perp, rtol=1e-3)

    @pytest.mark.parametrize("Z", [1, 4])
    def test_cross_vs_ji_held(self, Z):
        """cross should roughly agree with ji-held"""
        beta_hat_cross_brag = _nondim_tec_braginskii(self.big_hall, Z, 'cross')
        beta_hat_cross_jh = _nondim_tec_ji_held(self.big_hall, Z, 'cross')
        assert np.isclose(beta_hat_cross_brag, beta_hat_cross_jh, rtol=3e-2)


# test class for _nondim_resist_braginskii function:
class Test__nondim_resist_braginskii:

    def setup_method(self):
        """set up some initial values for tests"""
        self.big_hall = 1000
        self.small_hall = 0

    # values from Braginskii '65
    @pytest.mark.parametrize("Z, field_orientation, expected", [
        (1, 'par', 0.51),  # eq (2.8), table 1
        (2, 'par', 0.44),  # table 1
        (3, 'par', 0.40),  # eq (2.8), table 1
        (4, 'par', 0.38),  # eq (2.8), table 1
        (np.inf, 'par', 0.29),  # eq (2.8),table 1
    ])
    def test_known_values_par(self, Z, field_orientation, expected):
        """check some known values"""
        beta_hat = _nondim_resist_braginskii(self.big_hall,
                                             Z,
                                             field_orientation)
        assert np.isclose(beta_hat, expected, atol=1e-2)

    @pytest.mark.parametrize("Z", [1, 2, 3, 4, np.inf])
    def test_unmagnetized(self, Z):
        """confirm perp -> par as B -> 0"""
        alpha_hat_par = _nondim_resist_braginskii(self.small_hall, Z, 'par')
        alpha_hat_perp = _nondim_resist_braginskii(self.small_hall, Z, 'perp')
        assert np.isclose(alpha_hat_par, alpha_hat_perp, rtol=1e-3)

    @pytest.mark.parametrize("Z", [1, 4])
    def test_cross_vs_ji_held(self, Z):
        """cross should roughly agree with ji-held at hall 0.1"""
        alpha_hat_cross_brag = _nondim_resist_braginskii(0.1, Z, 'cross')
        alpha_hat_cross_jh = _nondim_resist_ji_held(0.1, Z, 'cross')
        assert np.isclose(alpha_hat_cross_brag, alpha_hat_cross_jh, rtol=5e-2)


# test class for _nondim_visc_i_braginskii function:
class Test__nondim_visc_i_braginskii:

    def setup_method(self):
        """set up some initial values for tests"""
        self.big_hall = 1000
        self.small_hall = 0

    # values from Braginskii '65, eqs. 2.22 to 2.24
    @pytest.mark.parametrize("expected, power", [
        (np.array((0.96, 0.3, 1.2, 0.5, 1.0)), np.array((0, 2, 2, 1, 1)))
    ])
    def test_known_values(self, expected, power):
        """check some known values"""
        eta_i_hat = _nondim_visc_i_braginskii(self.big_hall)
        eta_i_hat_with_powers = eta_i_hat * self.big_hall ** power
        assert np.allclose(eta_i_hat_with_powers, expected, atol=1e-2)

    def test_vs_ji_held_K2(self):
        """confirm agreement with ji-held model when K=2"""
        eta_i_hat_brag = _nondim_visc_i_braginskii(self.big_hall)
        eta_i_hat_jh = _nondim_visc_i_ji_held(self.big_hall, 1, 0, 100, K=2)
        for idx in [0, 1, 2, 3, 4]:
            assert np.isclose(eta_i_hat_brag[idx], eta_i_hat_jh[idx],
                              rtol=2e-2)


# test class for _nondim_visc_e_braginskii function:
class Test__nondim_visc_e_braginskii:

    def setup_method(self):
        """set up some initial values for tests"""
        self.big_hall = 1000
        self.small_hall = 0

    # values from Braginskii '65
    @pytest.mark.parametrize("Z, expected, idx", [
        (1, 0.73, 0),  # eq (2.25)
        (1, 0.51, 1),  # eq (2.26)
        (1, 2.04, 2),  # eq (2.26)
        (1, 0.5, 3),  # eq (2.27)
        (1, 1.0, 4),  # eq (2.27)
    ])
    def test_known_values(self, Z, expected, idx):
        """check some known values"""
        beta_hat = _nondim_visc_e_braginskii(self.big_hall, Z)
        if idx == 0:
            assert(np.isclose(beta_hat[idx], expected, atol=1e-2))
        elif idx == 1 or idx == 2:
            assert np.isclose(beta_hat[idx] * self.big_hall ** 2, expected,
                              atol=1e-2)
        elif idx == 3 or idx == 4:
            assert np.isclose(beta_hat[idx] * self.big_hall, expected,
                              atol=1e-1)


def test_fail__check_Z_nan():
    with pytest.raises(PhysicsError):
        _check_Z([1, 2, 3], 4)


@pytest.mark.parametrize("Z", [1, 2, 4, 16, np.inf])
def test__nondim_tc_e_spitzer(Z):
    """test _nondim_tc_e_spitzer function"""
    kappa = _nondim_tc_e_spitzer(Z)
    if Z == 1:
        kappa_check = 3.203
        rtol = 1e-3
    elif Z == 2 or Z == 4:
        kappa_check = _nondim_tc_e_braginskii(0, Z, 'par')
        rtol = 2e-2
    elif Z == 16:
        kappa_check = _nondim_tc_e_ji_held(0, Z, 'par')
        rtol = 2e-2
    elif Z == np.inf:
        kappa_check = _nondim_tc_e_ji_held(0, 1e6, 'par')
        rtol = 2e-2
    assert np.isclose(kappa, kappa_check, rtol=rtol)


@pytest.mark.parametrize("Z", [1, 2, 4, 16, np.inf])
def test__nondim_resist_spitzer(Z):
    """test _nondim_resist_spitzer function"""
    alpha = _nondim_resist_spitzer(Z, 'par')
    if Z == 1:
        alpha_check = 0.5064
        rtol = 1e-3
    elif Z == 2 or Z == 4 or Z == np.inf:
        alpha_check = _nondim_resist_braginskii(0, Z, 'par')
        rtol = 2e-2
    elif Z == 16:
        alpha_check = _nondim_resist_ji_held(0, Z, 'par')
        rtol = 2e-2
    assert np.isclose(alpha, alpha_check, rtol=rtol)


@pytest.mark.parametrize("Z", [1, 2, 4, 16, np.inf])
def test__nondim_tec_spitzer(Z):
    """test _nondim_tec_spitzer function"""
    beta = _nondim_tec_spitzer(Z)
    if Z == 1:
        beta_check = 0.699
        rtol = 1e-3
    elif Z == 2 or Z == 4 or Z == np.inf:
        beta_check = _nondim_tec_braginskii(0, Z, 'par')
        rtol = 2e-2
    elif Z == 16:
        beta_check = _nondim_tec_ji_held(0, Z, 'par')
        rtol = 2e-2
    assert np.isclose(beta, beta_check, rtol=rtol)


# approximated from Ji-Held '13 figures 1 and 2 (black circles)
@pytest.mark.parametrize("hall, Z, field_orientation, expected", [
    (0.0501, 1, 'perp', 3.187),
    (0.2522, 1, 'perp', 2.597),
    (1.004, 1, 'perp', 0.9942),
    (3.178, 1, 'perp', 0.2218),
    (10.03, 1, 'perp', 0.03216),
    (31.66, 1, 'perp', 0.003878),
    (100.6, 1, 'perp', 0.0004241),
    (315.7, 1, 'perp', 0.00004492),
    (1005, 1, 'perp', 0.000004543),
    (0.03175, 1, 'cross', 0.1899),
    (0.1001, 1, 'cross', 0.5648),
    (0.3166, 1, 'cross', 1.234),
    (1.267, 1, 'cross', 1.157),
    (4, 1, 'cross', 0.5359),
    (12.64, 1, 'cross', 0.1906),
    (40.04, 1, 'cross', 0.06191),
    (126.4, 1, 'cross', 0.01981),
    (401.6, 1, 'cross', 0.006282),
    (0.02494, 100, 'perp', 11.57),
    (0.09969, 100, 'perp', 6.707),
    (0.3987, 100, 'perp', 1.964),
    (1.586, 100, 'perp', 0.3524),
    (4.991, 100, 'perp', 0.06185),
    (15.85, 100, 'perp', 0.008857),
    (49.85, 100, 'perp', 0.001078),
    (158, 100, 'perp', 0.0001184),
    (499.9, 100, 'perp', 0.00001236),
    (0.0319, 100, 'cross', 3.68),
    (0.1271, 100, 'cross', 5.023),
    (0.502, 100, 'cross', 2.945),
    (1.595, 100, 'cross', 1.283),
    (5.017, 100, 'cross', 0.462),
    (15.95, 100, 'cross', 0.1534),
    (50.24, 100, 'cross', 0.04949),
    (158.1, 100, 'cross', 0.01572),
    (500.8, 100, 'cross', 0.004972),
])
def test__nondim_tc_e_ji_held(hall, Z, field_orientation, expected):
    """test _nondim_tc_e_ji_held function"""
    kappa_hat = _nondim_tc_e_ji_held(hall, Z, field_orientation)
    kappa_check = expected
    assert np.isclose(kappa_hat, kappa_check, rtol=2e-2)


# approximated from Ji-Held '13 figures 1 and 2 (black circles)
@pytest.mark.parametrize("hall, Z, field_orientation, expected", [
    (0.03939, 1, 'perp', 0.6959),
    (0.2498, 1, 'perp', 0.6216),
    (1.258, 1, 'perp', 0.3007),
    (6.321, 1, 'perp', 0.06303),
    (25.14, 1, 'perp', 0.01126),
    (100.3, 1, 'perp', 0.00161),
    (317.2, 1, 'perp', 0.0002877),
    (1006, 1, 'perp', 0.0000483),
    (3191, 1, 'perp', 0.000007741),
    (9958, 1, 'perp', 0.000001226),
    (0.02515, 1, 'cross', 0.02218),
    (0.06343, 1, 'cross', 0.0551),
    (0.1589, 1, 'cross', 0.1268),
    (0.5041, 1, 'cross', 0.2523),
    (2.006, 1, 'cross', 0.24),
    (6.321, 1, 'cross', 0.1335),
    (19.97, 1, 'cross', 0.05613),
    (50.26, 1, 'cross', 0.0253),
    (126.9, 1, 'cross', 0.01083),
    (317.5, 1, 'cross', 0.004495),
    (795.3, 1, 'cross', 0.001839),
    (0.03975, 100, 'perp', 1.335),
    (0.2522, 100, 'perp', 0.7647),
    (1.258, 100, 'perp', 0.2709),
    (6.345, 100, 'perp', 0.05833),
    (25.12, 100, 'perp', 0.01112),
    (100.2, 100, 'perp', 0.001649),
    (317.8, 100, 'perp', 0.0002915),
    (992.3, 100, 'perp', 0.00004875),
    (3170, 100, 'perp', 0.000007839),
    (9994, 100, 'perp', 0.000001213),
    (0.02507, 100, 'cross', 0.2022),
    (0.07935, 100, 'cross', 0.4037),
    (0.3155, 100, 'cross', 0.4764),
    (1.258, 100, 'cross', 0.3272),
    (3.958, 100, 'cross', 0.1795),
    (12.53, 100, 'cross', 0.08046),
    (39.62, 100, 'cross', 0.03088),
    (100.2, 100, 'cross', 0.01332),
    (250.3, 100, 'cross', 0.00562),
    (629.8, 100, 'cross', 0.002308),
])
def test__nondim_tec_ji_held(hall, Z, field_orientation, expected):
    """test _nondim_tec_ji_held function"""
    beta_hat = _nondim_tec_ji_held(hall, Z, field_orientation)
    beta_check = expected
    assert np.isclose(beta_hat, beta_check, rtol=2e-2)


# approximated from Ji-Held '13 figures 1 and 2 (black circles)
@pytest.mark.parametrize("hall, Z, field_orientation, expected", [
    (0.06317, 1, 'perp', 0.5064),
    (0.3966, 1, 'perp', 0.5316),
    (1.586, 1, 'perp', 0.619),
    (5.041, 1, 'perp', 0.7309),
    (15.8, 1, 'perp', 0.8343),
    (63.35, 1, 'perp', 0.92),
    (315.6, 1, 'perp', 0.9701),
    (1998, 1, 'perp', 0.9912),
    (0.02495, 1, 'cross', 0.005026),
    (0.06249, 1, 'cross', 0.01255),
    (0.1574, 1, 'cross', 0.03007),
    (0.4998, 1, 'cross', 0.07338),
    (1.995, 1, 'cross', 0.1254),
    (12.6, 1, 'cross', 0.1211),
    (62.96, 1, 'cross', 0.07421),
    (252.4, 1, 'cross', 0.03992),
    (998.4, 1, 'cross', 0.01908),
    (3194, 1, 'cross', 0.009749),
    (9963, 1, 'cross', 0.004812),
    (0.06333, 100, 'perp', 0.3144),
    (0.3137, 100, 'perp', 0.3894),
    (0.9954, 100, 'perp', 0.4979),
    (2.507, 100, 'perp', 0.6091),
    (6.324, 100, 'perp', 0.7221),
    (20.02, 100, 'perp', 0.8401),
    (79.68, 100, 'perp', 0.9275),
    (399.2, 100, 'perp', 0.9743),
    (2509, 100, 'perp', 0.9922),
    (0.02505, 100, 'cross', 0.02138),
    (0.07929, 100, 'cross', 0.05403),
    (0.3138, 100, 'cross', 0.1133),
    (1.581, 100, 'cross', 0.1693),
    (9.994, 100, 'cross', 0.1539),
    (49.88, 100, 'cross', 0.09238),
    (199.1, 100, 'cross', 0.04845),
    (786.3, 100, 'cross', 0.02278),
    (2504, 100, 'cross', 0.01152),
    (7879, 100, 'cross', 0.005652),
])
def test__nondim_resist_ji_held(hall, Z, field_orientation, expected):
    """test _nondim_resist_ji_held function"""
    alpha_hat = _nondim_resist_ji_held(hall, Z, field_orientation)
    alpha_check = expected
    assert np.isclose(alpha_hat, alpha_check, rtol=2e-2)


# approximated from Ji-Held '13 figure 3 (K = 80)
@pytest.mark.parametrize("hall, Z, index, expected", [
    (0.01968, 1, 2, 0.7368),
    (0.1338, 1, 2, 0.7171),
    (0.4766, 1, 2, 0.6003),
    (1.339, 1, 2, 0.3241),
    (4.479, 1, 2, 0.06964),
    (15.28, 1, 2, 0.008006),
    (48.82, 1, 2, 0.0008496),
    (94.89, 1, 2, 0.0002257),
    (0.01267, 1, 4, 0.009038),
    (0.03978, 1, 4, 0.02831),
    (0.1151, 1, 4, 0.08041),
    (0.2904, 1, 4, 0.1823),
    (0.8049, 1, 4, 0.3158),
    (1.77, 1, 4, 0.3083),
    (3.886, 1, 4, 0.2062),
    (16.68, 1, 4, 0.05845),
    (39.82, 1, 4, 0.02501),
    (77.82, 1, 4, 0.01288),
])
def test__nondim_visc_e_ji_held(hall, Z, index, expected):
    """test _nondim_visc_e_ji_held function"""
    alpha_hat = _nondim_visc_e_ji_held(hall, Z)
    alpha_check = expected
    assert np.isclose(alpha_hat[index], alpha_check, rtol=2e-2)


# approximated from Ji-Held '13 figure 7
# note r = sqrt(2) * x
# note kappa from figure = kappa_hat / sqrt(2)
@pytest.mark.parametrize("hall, Z, mu, theta, field_orientation, expected", [
    (0.01529, 1, 0, 100, 'perp', 3.99586042),
    (0.0589, 1, 0, 100, 'perp', 3.96828326),
    (0.22953, 1, 0, 100, 'perp', 3.34885772),
    (0.50077, 1, 0, 100, 'perp', 2.22385083),
    (1.29924, 1, 0, 100, 'perp', 0.76650375),
    (5.48856, 1, 0, 100, 'perp', 0.06337091),
    (25.99325, 1, 0, 100, 'perp', 0.00298328),
    (68.00953, 1, 0, 100, 'perp', 0.00042822),
    (120.53342, 1, 0, 100, 'perp', 0.00013739),
    (0.01865, 1, 0, 100, 'cross', 0.13661303),
    (0.04544, 1, 0, 100, 'cross', 0.32795613),
    (0.14199, 1, 0, 100, 'cross', 0.95317994),
    (0.38806, 1, 0, 100, 'cross', 1.73029029),
    (1.12996, 1, 0, 100, 'cross', 1.53230039),
    (2.96843, 1, 0, 100, 'cross', 0.77216061),
    (12.42528, 1, 0, 100, 'cross', 0.19968696),
    (77.11707, 1, 0, 100, 'cross', 0.03235721),
])
def test__nondim_tc_i_ji_held(hall, Z, mu, theta, field_orientation, expected):
    """test _nondim_tc_i_ji_held function"""
    kappa_hat = _nondim_tc_i_ji_held(hall, Z, mu, theta, field_orientation)
    kappa_check = expected
    assert np.isclose(kappa_hat, kappa_check, rtol=2e-2)


# approximated from Ji-Held '13 figure 7
# note r = sqrt(2) * x
# note eta from figure = eta_hat / sqrt(2)
@pytest.mark.parametrize("hall, Z, mu, theta, index, expected", [
    (0.01981, 1, 0, 100, 2, 0.96166522),
    (0.17423, 1, 0, 100, 2, 0.92206724),
    (0.66369, 1, 0, 100, 2, 0.6344162),
    (1.72958, 1, 0, 100, 2, 0.24890159),
    (10.09041, 1, 0, 100, 2, 0.01134199),
    (52.50975, 1, 0, 100, 2, 0.00042844),
    (0.01829, 1, 0, 100, 4, 0.01943837),
    (0.07845, 1, 0, 100, 4, 0.08251936),
    (0.35765, 1, 0, 100, 4, 0.31643028),
    (0.99985, 1, 0, 100, 4, 0.45346758),
    (4.35295, 1, 0, 100, 4, 0.21036427),
    (22.74055, 1, 0, 100, 4, 0.04358606),
    (80.42633, 1, 0, 100, 4, 0.01238144),
])
def test__nondim_visc_i_ji_held(hall, Z, mu, theta, index, expected):
    """test _nondim_visc_i_ji_held function"""
    kappa_hat = _nondim_visc_i_ji_held(hall, Z, mu, theta)
    kappa_check = expected
    assert np.isclose(kappa_hat[index], kappa_check, rtol=2e-2)
