"""
For calculating the plasma dispersion function :math:`Z(ζ)` and its
derivative :math:`Z′(ζ)`.
"""

__all__ = ["plasma_dispersion_func",
           "plasma_dispersion_func_deriv",
           "plasma_dispersion_1D_dist",
           "plasma_dispersion_1D_dist_arr",
           "plasma_dispersion_1D_dist_deriv",
           "plasma_dispersion_1D_dist_deriv_arr"]


import astropy.units as u
from astropy.constants.si import hbar
from astropy.constants import c
import numpy as np
from scipy.special import wofz as faddeeva_function
from scipy.integrate import quad
from collections.abc import Callable

from plasmapy.particles.particle_class import ParticleLike
from plasmapy.particles.atomic import particle_mass
from plasmapy.formulary.frequencies import plasma_frequency
from plasmapy.formulary.speeds import thermal_speed
from plasmapy.utils.decorators import (
    bind_lite_func,
    preserve_signature,
    validate_quantities,
)


@preserve_signature
def bohm_gross_lite(kWave, vth, wp):
    r"""
    
    Parameters
    ----------
    kWave : |array_like| of real values
        The corresponding wavenumber, in rad/m, of the electromagnetic
        wave propagating through the plasma.

    vth : `float`
        The 3D, most probable thermal speed, in m/s. (i.e. it includes
        the factor of :math:`\sqrt{2}`, see
        :ref:`thermal speed notes <thermal-speed-notes>`)

    wp : `float`
        The plasma frequency, in rad/s.
        
    Returns
    -------
    wl : `~astropy.units.Quantity`
        The frequency, in rad/s, of the longitudinal Langmuir wave propagating
        through the plasma.
    """
    wl = np.sqrt(wp ** 2 + 3 * (kWave * vth) ** 2)
    return wl


@bind_lite_func(bohm_gross_lite)
@validate_quantities(
    kWave={"none_shall_pass": True}
)
def bohm_gross(
    kWave: u.Quantity[u.rad / u.m],
    T: u.Quantity[u.K],
    n: u.Quantity[u.m**-3],
    particle: ParticleLike,
    z_mean: float | None = None,
) -> u.Quantity[u.rad / u.s]:
    r"""
    
    Parameters
    ----------
    kWave : `~astropy.units.Quantity`
        The corresponding wavenumber, in rad/m, of the electromagnetic
        wave propagating through the plasma.

    T : `~astropy.units.Quantity`
        The plasma temperature — this can be either the electron or the
        ion temperature, but should be consistent with density and
        particle.

    n : `~astropy.units.Quantity`
        The plasma density — this can be either the electron or the ion
        density, but should be consistent with temperature and particle.

    particle : |particle-like|
        The plasma particle species.

    z_mean : `float`
        The average ionization of the plasma. This is only required for
        calculating the ion permittivity.
        
    Returns
    -------
    wl : `~astropy.units.Quantity`
        The frequency, in rad/s, of the longitudinal Langmuir wave propagating
        through the plasma.
    """
    vth = thermal_speed(T=T, particle=particle, method="most_probable").value
    wp = plasma_frequency(n=n, particle=particle, Z=z_mean).value
    wl = bohm_gross_lite(kWave.value, vth, wp)
    return wl


def plasma_dispersion_func(
    zeta: complex | np.ndarray | u.Quantity[u.dimensionless_unscaled],
) -> complex | np.ndarray | u.Quantity[u.dimensionless_unscaled]:
    r"""
    Calculate the plasma dispersion function.

    The plasma dispersion function is defined as:

    .. math::
        Z(ζ) = π^{-0.5} \int_{-∞}^{+∞}
        \frac{e^{-x^2}}{x-ζ} dx

    where the argument is a complex number :cite:p:`fried:1961`. This function
    is analytically continued so it works in both the upper and lower halves
    of the complex plane in addition to the real line.

    Parameters
    ----------
    zeta : |array_like| or |Quantity|
        The real or complex value to be provided as an argument to the
        plasma dispersion function.

    Returns
    -------
    |array_like| or |Quantity|
        The real or complex value of plasma dispersion function
        evaluated at ``zeta``.

    Raises
    ------
    ~astropy.units.UnitsError
        If ``zeta`` is a |Quantity| but is not dimensionless.

    See Also
    --------
    `~plasmapy.dispersion.dispersion_functions.plasma_dispersion_func_deriv`

    Notes
    -----
    In plasma wave theory, the plasma dispersion function appears
    frequently when the background medium has a Maxwellian
    distribution function.  The argument of this function then refers
    to the ratio of a wave's phase velocity to a thermal velocity.

    Examples
    --------
    >>> from plasmapy.dispersion import plasma_dispersion_func
    >>> plasma_dispersion_func(0)
    np.complex128(1.77245385...j)
    >>> plasma_dispersion_func(1 + 1j)
    np.complex128(-0.36905845...+0.54014504...j)
    >>> plasma_dispersion_func([0.3, 0.7 + 2.3j])
    array([-0.56526333+1.61990085j, -0.09995023+0.37685142j])
    """
    try:
        return 1j * np.sqrt(np.pi) * faddeeva_function(zeta)
    except u.UnitTypeError as wrong_units:
        raise u.UnitsError(
            "The argument to plasma_dispersion_func "
            "must be dimensionless if it is a Quantity."
        ) from wrong_units
    except TypeError as wrong_type:
        raise TypeError(
            "The argument to plasma_dispersion_func should be a real or "
            "complex number or array, or a dimensionless Quantity."
        ) from wrong_type


def plasma_dispersion_func_deriv(
    zeta: complex | np.ndarray | u.Quantity[u.dimensionless_unscaled],
) -> complex | np.ndarray | u.Quantity[u.dimensionless_unscaled]:
    r"""
    Calculate the derivative of the plasma dispersion function.

    The derivative of the plasma dispersion function is:

    .. math::
        Z'(ζ) = π^{-1/2} \int_{-∞}^{+∞} \frac{e^{-x^2}}{(x-ζ)^2} dx

    where the argument :math:`ζ` is a complex number
    :cite:p:`fried:1961`. This function
    is analytically continued so it works in both the upper and lower halves
    of the complex plane in addition to the real line.

    Parameters
    ----------
    zeta : |array_like| or |Quantity|
        Argument of plasma dispersion function.

    Returns
    -------
    complex, `~numpy.ndarray`, or |Quantity|
        First derivative of plasma dispersion function.

    Raises
    ------
    ~astropy.units.UnitsError
        If the argument is a `~astropy.units.Quantity` but is not
        dimensionless.

    See Also
    --------
    `~plasmapy.dispersion.dispersion_functions.plasma_dispersion_func`

    Examples
    --------
    >>> plasma_dispersion_func_deriv(0)
    np.complex128(-2+0j)
    >>> plasma_dispersion_func_deriv(1j)
    np.complex128(-0.48425568771737604+0j)
    >>> plasma_dispersion_func_deriv(-1.52 + 0.47j)
    np.complex128(0.165871331...+0.4458797880...j)
    """
    try:
        return -2 * (1 + zeta * plasma_dispersion_func(zeta))
    except u.UnitsError as wrong_units:
        raise u.UnitsError(
            "The argument to plasma_dispersion_func_deriv "
            "must be dimensionless if it is a Quantity."
        ) from wrong_units
    except TypeError as wrong_type:
        raise TypeError(
            "The argument to plasma_dispersion_func_deriv "
            "must be one of the following types: complex, float, "
            "int, ndarray, or a dimensionless Quantity."
        ) from wrong_type



def plasma_dispersion_1D_dist(
    zeta: complex | u.Quantity[u.dimensionless_unscaled],
    dist: Callable[[float | u.Quantity[u.m / u.s]], float | u.Quantity[u.s / u.m]],
    kWave: u.Quantity[u.rad / u.m],
    vth: u.Quantity[u.m / u.s],
    wp: u.Quantity[u.rad / u.s],
    particle: ParticleLike = "e-",
) -> complex | np.ndarray | u.Quantity[u.dimensionless_unscaled]:
    r"""
    Calculate the plasma dispersion for a given 1D distribution function.

    The plasma dispersion function is defined as (see eq 15 in
    https://arxiv.org/abs/1305.6476 and https://doi.org/10.1063/1.4822332):

    .. math::
        g^{+}(\zeta) =
        \begin{cases}
            \frac{1}{\pi} \int_{-\infty}^{+\infty} \frac{f(x)}{x - \zeta} dx, \quad \Im(\zeta) > 0 \\
            \frac{1}{\pi} PV\int_{-\infty}^{+\infty} \frac{f(x)}{x - \zeta} dx + if(\zeta), \quad \Im(\zeta) = 0 \\
            \frac{1}{\pi} \int_{-\infty}^{+\infty} \frac{f(x)}{x - \zeta} dx + 2if(\zeta), \quad \Im(\zeta) < 0
        \end{cases}

    where the argument :math:`\zeta` is a complex number representing the
    phase velocity normalized by the thermal velocity, and where
    :math:`x=\frac{v}{v_{th}}` is the velocity in the distribution function
    normalized by the thermal velocity. The function has been
    analytically continued from the upper half of the complex plane to the
    lower half of the complex plane.

    Parameters
    ----------
    zeta : |Quantity|
        The real or complex value to be provided as an argument to the
        plasma dispersion function. This is the ratio of the wave's phase
        velocity to the thermal velocity.
        
    dist : function
        1D distribution function of the plasma velocity. The function should only
        have one argument, which is the velocity in m/s. The function
        should return probability density in units of velocity\ :sup:`-1`\ , 
        normalized so that :math:`\int_{-∞}^{+∞} f(v) dv = 1`. When considering
        an anisotropic distribution, dist should be the slice through the
        distribution function which is aligned with the k-vector. The function
        should be differentiable (analytic).

    kWave : `~astropy.units.Quantity`
        The corresponding wavenumber, in rad/m, of the electromagnetic
        wave propagating through the plasma.

    vth : `~astropy.units.Quantity`
        The 3D, most probable thermal speed, in m/s. (i.e. it includes
        the factor of :math:`\sqrt{2}`, see
        :ref:`thermal speed notes <thermal-speed-notes>`)

    wp : `~astropy.units.Quantity`
        The plasma frequency, in rad/s.
        
    particle : `str`, optional
        Representation of the particle species(e.g., ``'p+'`` for protons,
        ``'D+'`` for deuterium, or ``'He-4 +1'`` for singly ionized
        helium-4), which defaults to electrons.

    Returns
    -------
    |array_like| or |Quantity|
        The real or complex value of plasma dispersion function
        evaluated at ``zeta``.

    Raises
    ------
    ~astropy.units.UnitsError
        If ``zeta`` is a |Quantity| but is not dimensionless.

    See Also
    --------
    `~plasmapy.dispersion.dispersion_functions.plasma_dispersion_1D_dist_deriv`

    Examples
    --------
    
    """
    # normalizing distribution function by the thermal velocity so that
    # it is dimensionless in the same way as zeta.
    dist_norm = lambda x: (vth * dist(x * vth)).value
    # g_func is the whole function to be integrated, which is of the form
    # g(x) = f(x) / (x - x0)
    # where x0 is the location of the pole
    # and f_func is the f(x) where the denominator has been omitted for the
    # purposes of cauchy principal value integration using quad().
    try:
        pole = zeta.to(u.dimensionless_unscaled).value
    except AttributeError:
        pole = zeta

    f_func = dist_norm
    g_func = lambda x: f_func(x) / (x - pole)
    
    # integration on the real line
    if zeta.imag == 0:
        # A physically meaningful stand-off distance from the pole is defined
        # in terms of the group velocty of the longitudinal Langmuir wave.
        # The pole occurs when the EM wave frequency matches the plasma
        # frequency and so the dispersion relation blows up.
        wl = bohm_gross_lite(kWave, vth, wp)
        standoff = (wl / (kWave * vth)).to(u.dimensionless_unscaled).value
        # integration limits using stand-off distance
        pole_neg = (pole - standoff).real
        pole_pos = (pole + standoff).real
        # integral from -inf to pole
        neg_integral, neg_err = quad(func=g_func,
                                     a=-c.value,
                                     b=pole_neg,
                                     epsabs=1.49e-08,
                                     epsrel=1.49e-08,
                                     limit=50,
                                     points=None,
                                     weight=None,
                                     wvar=None,
                                     wopts=None,
                                     maxp1=50,
                                     limlst=50,
                                     complex_func=True)
        # integral from pole to +inf
        pos_integral, pos_err = quad(func=g_func,
                                     a=pole_pos,
                                     b=c.value,
                                     epsabs=1.49e-08,
                                     epsrel=1.49e-08,
                                     limit=50,
                                     points=None,
                                     weight=None,
                                     wvar=None,
                                     wopts=None,
                                     maxp1=50,
                                     limlst=50,
                                     complex_func=True)
        # Cauchy Principal Value integral across the pole
        cauchy_integral, cauch_err = quad(func=f_func,
                                          a=pole_neg,
                                          b=pole_pos,
                                          epsabs=1.49e-08,
                                          epsrel=1.49e-08,
                                          limit=50,
                                          points=None,
                                          weight="cauchy",
                                          wvar=pole.real,
                                          wopts=None,
                                          maxp1=50,
                                          limlst=50,
                                          complex_func=True)
        # the residue on a semi-circle Cauchy integral around the pole
        residue = np.pi * 1j * f_func(pole)
        # combine it all together
        total_integral = (neg_integral + pos_integral + cauchy_integral + residue)
        return total_integral
    elif zeta.imag > 0:
        # analytic extension to the upper half of the complex plane (growth)
        integral, integral_err = quad(func=g_func,
                                      a=-c.value,
                                      b=c.value,
                                      epsabs=1.49e-08,
                                      epsrel=1.49e-08,
                                      limit=50,
                                      points=None,
                                      weight=None,
                                      wvar=None,
                                      wopts=None,
                                      maxp1=50,
                                      limlst=50,
                                      complex_func=True)
        total_integral = integral
        return total_integral
    elif zeta.imag < 0:
        # analytic extension to the lower half of the complex plane (dampening)
        integral, integral_err = quad(func=g_func,
                                      a=-c.value,
                                      b=c.value,
                                      epsabs=1.49e-08,
                                      epsrel=1.49e-08,
                                      limit=50,
                                      points=None,
                                      weight=None,
                                      wvar=None,
                                      wopts=None,
                                      maxp1=50,
                                      limlst=50,
                                      complex_func=True)
        # the residue for analytic continuation
        residue = np.pi * 2j * f_func(pole)
        total_integral = (integral + residue)
        return total_integral * u.dimensionless_unscaled
    

def plasma_dispersion_1D_dist_arr(
    zetas: complex | np.ndarray | u.Quantity[u.dimensionless_unscaled],
    dist: Callable[[float | u.Quantity[u.m / u.s]], float | u.Quantity[u.s / u.m]],
    kWave: u.Quantity[u.rad / u.m],
    vth: u.Quantity[u.m / u.s],
    wp: u.Quantity[u.rad / u.s],
    particle: ParticleLike = "e-",
) -> complex | np.ndarray | u.Quantity[u.dimensionless_unscaled]:
    r"""
    Convenience function for passing an array of zetas to
    plasma_dispersion_1D_dist.

    The plasma dispersion function is defined as (see eq 15 in
    https://arxiv.org/abs/1305.6476 and https://doi.org/10.1063/1.4822332):

    .. math::
        g^{+}(\zeta) =
        \begin{cases}
            \frac{1}{\pi} \int_{-\infty}^{+\infty} \frac{f(x)}{x - \zeta} dx, \quad \Im(\zeta) > 0 \\
            \frac{1}{\pi} PV\int_{-\infty}^{+\infty} \frac{f(x)}{x - \zeta} dx + if(\zeta), \quad \Im(\zeta) = 0 \\
            \frac{1}{\pi} \int_{-\infty}^{+\infty} \frac{f(x)}{x - \zeta} dx + 2if(\zeta), \quad \Im(\zeta) < 0
        \end{cases}

    where the argument :math:`\zeta` is a complex number representing the
    phase velocity normalized by the thermal velocity, and where
    :math:`x=\frac{v}{v_{th}}` is the velocity in the distribution function
    normalized by the thermal velocity. The function has been
    analytically continued from the upper half of the complex plane to the
    lower half of the complex plane.

    Parameters
    ----------
    zeetas : |array_like|
        The real or complex value to be provided as an argument to the
        plasma dispersion function. This is the ratio of the wave's phase
        velocity to the thermal velocity.
        
    dist : function
        1D distribution function of the plasma velocity. The function should only
        have one argument, which is the velocity in m/s. The function
        should return probability density in units of velocity\ :sup:`-1`\ , 
        normalized so that :math:`\int_{-∞}^{+∞} f(v) dv = 1`. When considering
        an anisotropic distribution, dist should be the slice through the
        distribution function which is aligned with the k-vector. The function
        should be differentiable (analytic).

    kWave : `~astropy.units.Quantity`
        The corresponding wavenumber, in rad/m, of the electromagnetic
        wave propagating through the plasma.

    vth : `~astropy.units.Quantity`
        The 3D, most probable thermal speed, in m/s. (i.e. it includes
        the factor of :math:`\sqrt{2}`, see
        :ref:`thermal speed notes <thermal-speed-notes>`)

    wp : `~astropy.units.Quantity`
        The plasma frequency, in rad/s.
        
    particle : `str`, optional
        Representation of the particle species(e.g., ``'p+'`` for protons,
        ``'D+'`` for deuterium, or ``'He-4 +1'`` for singly ionized
        helium-4), which defaults to electrons.

    Returns
    -------
    |array_like| or |Quantity|
        The real or complex value of plasma dispersion function
        evaluated at ``zeta``.

    Raises
    ------
    ~astropy.units.UnitsError
        If ``zeta`` is a |Quantity| but is not dimensionless.

    See Also
    --------
    `~plasmapy.dispersion.dispersion_functions.plasma_dispersion_1D_dist`

    Examples
    --------
    
    """
    dispersions = np.array([plasma_dispersion_1D_dist(
        zeta=zeta,
        dist=dist,
        kWave=kWave,
        vth=vth,
        wp=wp,
        particle=particle) for zeta in zetas])
    return dispersions


def plasma_dispersion_1D_dist_deriv(
    zeta: complex | np.ndarray | u.Quantity[u.dimensionless_unscaled],
    dist: Callable[[float | u.Quantity[u.m / u.s]], float | u.Quantity[u.s / u.m]],
    kWave: u.Quantity[u.rad / u.m],
    vth: u.Quantity[u.m / u.s],
    wp: u.Quantity[u.rad / u.s],
    particle: ParticleLike = "e-",
) -> complex | np.ndarray | u.Quantity[u.dimensionless_unscaled]:
    r"""
    Calculate the derivative of the plasma dispersion function generalized
    for arbitrary (well-behaved) 1D distribution functions.

    The derivative of the plasma dispersion function is:

    .. math::
        Z'(ζ) = \frac{∂Z}{∂ζ}
        
    This results in (see eqs 9 and 15 in
    https://arxiv.org/abs/1305.6476 and https://doi.org/10.1063/1.4822332):
    .. math::
        g'^{+}(\zeta) =
        \begin{cases}
            \frac{1}{\pi} \int_{-\infty}^{+\infty} \frac{\partial f(x) / \partial x}{x - \zeta} dx, \quad \Im(\zeta) > 0 \\
            \frac{1}{\pi} PV\int_{-\infty}^{+\infty} \frac{\partial f(x) / \partial x}{x - \zeta} dx + if'(\zeta), \quad \Im(\zeta) = 0 \\
            \frac{1}{\pi} \int_{-\infty}^{+\infty} \frac{\partial f(x) \partial x}{x - \zeta} dx + 2if'(\zeta), \quad \Im(\zeta) < 0
        \end{cases}

    where the argument :math:`\zeta` is a complex number representing the
    phase velocity normalized by the thermal velocity, and where
    :math:`x=\frac{v}{v_{th}}` is the velocity in the distribution function
    normalized by the thermal velocity.
    The function has been analytically continued from the upper half of the
    complex plane to the lower half of the complex plane.
    
    The partial derivative of the distribution function is approximated through
    a symmetric finite difference using the Compton shift as a physically
    characteristic step size:
        
    .. math::
        f'(x) = \frac{\partial f(x)}{\partial x} \approx \frac{f(x + \Delta x) - f(x - \Delta x)}{2 \Delta x}
        
    where the step size is the momentum due to the Compton shift, converted
    into a velocity and then normalized by the thermal velocity:
    
    .. math::
        \Delta x = \frac{\hbar k}{2 m v_{th}}
        

    Parameters
    ----------
    zeta : |Quantity|
        The real or complex value to be provided as an argument to the
        plasma dispersion function. This is the ratio of the wave's phase
        velocity to the thermal velocity.
        
    dist : function
        1D distribution function of the plasma velocity. The function should only
        have one argument, which is the velocity in m/s. The function
        should return probability density in units of velocity\ :sup:`-1`\ , 
        normalized so that :math:`\int_{-∞}^{+∞} f(v) dv = 1`. When considering
        an anisotropic distribution, dist should be the slice through the
        distribution function which is aligned with the k-vector. The function
        should be differentiable (analytic).
    
    kWave : `~astropy.units.Quantity`
        The corresponding wavenumber, in rad/m, of the electromagnetic
        wave propagating through the plasma.

    vth : `~astropy.units.Quantity`
        The 3D, most probable thermal speed, in m/s. (i.e. it includes
        the factor of :math:`\sqrt{2}`, see
        :ref:`thermal speed notes <thermal-speed-notes>`)

    wp : `~astropy.units.Quantity`
        The plasma frequency, in rad/s.
        
    particle : `str`, optional
        Representation of the particle species(e.g., ``'p+'`` for protons,
        ``'D+'`` for deuterium, or ``'He-4 +1'`` for singly ionized
        helium-4), which defaults to electrons.

    Returns
    -------
    complex, `~numpy.ndarray`, or |Quantity|
        First derivative of plasma dispersion function.

    Raises
    ------
    ~astropy.units.UnitsError
        If the argument is a `~astropy.units.Quantity` but is not
        dimensionless.

    See Also
    --------
    `~plasmapy.dispersion.dispersion_functions.plasma_dispersion_1D_dist`

    Examples
    --------
    
    """
    # zeta = omega / (kWave * vth)
    # so group velocity of the EM wave normalized by vth
    # not to be confused with group velocity of the langmuir wave wl
    
    # which implies that x is velocity normalized by vth to stay
    # consistent with the integral in eq (16) of G. Gregori PRE (2003)

    # x = v / vth so it is the velocity normalized by the thermal
    # velocity, so that it is in the same dimensionless units as zeta.
    # step size is based on photon momentum transferred to the particle (Compton shift)
    mass_particle = particle_mass(particle)
    diff_step = ((hbar * kWave / (2 * mass_particle)) / vth).to(u.dimensionless_unscaled, equivalencies=u.dimensionless_angles())
    dist_norm = lambda x: (vth * dist(x * vth)).value
    deriv_dist_func = lambda x: (dist_norm(x + diff_step) - dist_norm(x - diff_step)) / (2 * diff_step)
    
    # g_func is the whole function to be integrated, which is of the form
    # g(x) = f(x) / (x - x0)
    # where x0 is the location of the pole
    # and f_func is the f(x) where the denominator has been omitted for the
    # purposes of cauchy principal value integration using quad().
    try:
        pole = zeta.to(u.dimensionless_unscaled).value
    except AttributeError:
        pole = zeta
    f_func = deriv_dist_func
    g_func = lambda x: f_func(x) / (x - pole)
    
    # integration on the real line
    if zeta.imag == 0:
        # A physically meaningful stand-off distance from the pole is defined
        # in terms of the group velocty of the longitudinal Langmuir wave.
        # The pole occurs when the EM wave frequency matches the plasma
        # frequency and so the dispersion relation blows up.
        wl = bohm_gross_lite(kWave, vth, wp)
        standoff = (wl / (kWave * vth)).to(u.dimensionless_unscaled).value
        # integration limits using stand-off distance
        pole_neg = (pole - standoff).real
        pole_pos = (pole + standoff).real
        # integral from -inf to pole
        neg_integral, neg_err = quad(func=g_func,
                                     a=-c.value,
                                     b=pole_neg,
                                     epsabs=1.49e-08,
                                     epsrel=1.49e-08,
                                     limit=50,
                                     points=None,
                                     weight=None,
                                     wvar=None,
                                     wopts=None,
                                     maxp1=50,
                                     limlst=50,
                                     complex_func=True)
        # integral from pole to +inf
        pos_integral, pos_err = quad(func=g_func,
                                     a=pole_pos,
                                     b=c.value,
                                     epsabs=1.49e-08,
                                     epsrel=1.49e-08,
                                     limit=50,
                                     points=None,
                                     weight=None,
                                     wvar=None,
                                     wopts=None,
                                     maxp1=50,
                                     limlst=50,
                                     complex_func=True)
        # Cauchy Principal Value integral across the pole
        cauchy_integral, cauchy_err = quad(func=f_func,
                                           a=pole_neg,
                                           b=pole_pos,
                                           epsabs=1.49e-08,
                                           epsrel=1.49e-08,
                                           limit=50,
                                           points=None,
                                           weight="cauchy",
                                           wvar=pole.real,
                                           wopts=None,
                                           maxp1=50,
                                           limlst=50,
                                           complex_func=True)
        # the residue on a semi-circle Cauchy integral around the pole
        residue = np.pi * 1j * f_func(pole)
        # combine it all together
        total_integral = (neg_integral + pos_integral + cauchy_integral + residue)
        return total_integral
    elif zeta.imag > 0:
        # analytic extension to the upper half of the complex plane (growth)
        integral, integral_err = quad(func=g_func,
                                      a=-c.value,
                                      b=c.value,
                                      epsabs=1.49e-08,
                                      epsrel=1.49e-08,
                                      limit=50,
                                      points=None,
                                      weight=None,
                                      wvar=None,
                                      wopts=None,
                                      maxp1=50,
                                      limlst=50,
                                      complex_func=True)
        total_integral = integral
        return total_integral
    elif zeta.imag < 0:
        # analytic extension to the lower half of the complex plane (dampening)
        integral, integral_err = quad(func=g_func,
                                      a=-c.value,
                                      b=c.value,
                                      epsabs=1.49e-08,
                                      epsrel=1.49e-08,
                                      limit=50,
                                      points=None,
                                      weight=None,
                                      wvar=None,
                                      wopts=None,
                                      maxp1=50,
                                      limlst=50,
                                      complex_func=True)
        # the residue for analytic continuation
        residue = np.pi * 2j * f_func(pole)
        total_integral = (integral + residue)
        return total_integral * u.dimensionless_unscaled


def plasma_dispersion_1D_dist_deriv_arr(
    zetas: complex | np.ndarray | u.Quantity[u.dimensionless_unscaled],
    dist: Callable[[float | u.Quantity[u.m / u.s]], float | u.Quantity[u.s / u.m]],
    kWave: u.Quantity[u.rad / u.m],
    vth: u.Quantity[u.m / u.s],
    wp: u.Quantity[u.rad / u.s],
    particle: ParticleLike = "e-",
) -> complex | np.ndarray | u.Quantity[u.dimensionless_unscaled]:
    r"""
    Convenience function for passing an array of zetas to
    plasma_dispersion_1D_dist_deriv.

    The derivative of the plasma dispersion function is:

    .. math::
        Z'(ζ) = \frac{∂Z}{∂ζ}
        
    This results in (see eqs 9 and 15 in
    https://arxiv.org/abs/1305.6476 and https://doi.org/10.1063/1.4822332):
    .. math::
        g'^{+}(\zeta) =
        \begin{cases}
            \frac{1}{\pi} \int_{-\infty}^{+\infty} \frac{\partial f(x) / \partial x}{x - \zeta} dx, \quad \Im(\zeta) > 0 \\
            \frac{1}{\pi} PV\int_{-\infty}^{+\infty} \frac{\partial f(x) / \partial x}{x - \zeta} dx + if'(\zeta), \quad \Im(\zeta) = 0 \\
            \frac{1}{\pi} \int_{-\infty}^{+\infty} \frac{\partial f(x) \partial x}{x - \zeta} dx + 2if'(\zeta), \quad \Im(\zeta) < 0
        \end{cases}

    where the argument :math:`\zeta` is a complex number representing the
    phase velocity normalized by the thermal velocity, and where
    :math:`x=\frac{v}{v_{th}}` is the velocity in the distribution function
    normalized by the thermal velocity.
    The function has been analytically continued from the upper half of the
    complex plane to the lower half of the complex plane.
    
    The partial derivative of the distribution function is approximated through
    a symmetric finite difference using the Compton shift as a physically
    characteristic step size:
        
    .. math::
        f'(x) = \frac{\partial f(x)}{\partial x} \approx \frac{f(x + \Delta x) - f(x - \Delta x)}{2 \Delta x}
        
    where the step size is the momentum due to the Compton shift, converted
    into a velocity and then normalized by the thermal velocity:
    
    .. math::
        \Delta x = \frac{\hbar k}{2 m v_{th}}
        

    Parameters
    ----------
    zeta : |Quantity|
        The real or complex value to be provided as an argument to the
        plasma dispersion function. This is the ratio of the wave's phase
        velocity to the thermal velocity.
        
    dist : function
        1D distribution function of the plasma velocity. The function should only
        have one argument, which is the velocity in m/s. The function
        should return probability density in units of velocity\ :sup:`-1`\ , 
        normalized so that :math:`\int_{-∞}^{+∞} f(v) dv = 1`. When considering
        an anisotropic distribution, dist should be the slice through the
        distribution function which is aligned with the k-vector. The function
        should be differentiable (analytic).
    
    kWave : `~astropy.units.Quantity`
        The corresponding wavenumber, in rad/m, of the electromagnetic
        wave propagating through the plasma.

    vth : `~astropy.units.Quantity`
        The 3D, most probable thermal speed, in m/s. (i.e. it includes
        the factor of :math:`\sqrt{2}`, see
        :ref:`thermal speed notes <thermal-speed-notes>`)

    wp : `~astropy.units.Quantity`
        The plasma frequency, in rad/s.
        
    particle : `str`, optional
        Representation of the particle species(e.g., ``'p+'`` for protons,
        ``'D+'`` for deuterium, or ``'He-4 +1'`` for singly ionized
        helium-4), which defaults to electrons.

    Returns
    -------
    complex, `~numpy.ndarray`, or |Quantity|
        First derivative of plasma dispersion function.

    Raises
    ------
    ~astropy.units.UnitsError
        If the argument is a `~astropy.units.Quantity` but is not
        dimensionless.

    See Also
    --------
    `~plasmapy.dispersion.dispersion_functions.plasma_dispersion_1D_dist_deriv`

    Examples
    --------
    
    """
    dispersions = np.array([plasma_dispersion_1D_dist_deriv(
        zeta=zeta,
        dist=dist,
        kWave=kWave,
        vth=vth,
        wp=wp,
        particle=particle) for zeta in zetas])
    return dispersions