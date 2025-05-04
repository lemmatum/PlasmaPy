"""
For calculating the plasma dispersion function :math:`Z(ζ)` and its
derivative :math:`Z′(ζ)`.
"""

__all__ = ["plasma_dispersion_func",
           "plasma_dispersion_func_deriv",
           "plasma_dispersion_1D_dist",
           "plasma_dispersion_1D_dist_deriv"]


import astropy.units as u
import numpy as np
from scipy.special import wofz as faddeeva_function
from scipy.integrate import quad


def plasma_dispersion_func(
    zeta: complex | np.ndarray | u.Quantity[u.dimensionless_unscaled],
) -> complex | np.ndarray | u.Quantity[u.dimensionless_unscaled]:
    r"""
    Calculate the plasma dispersion function.

    The plasma dispersion function is defined as:

    .. math::
        Z(ζ) = π^{-0.5} \int_{-∞}^{+∞}
        \frac{e^{-x^2}}{x-ζ} dx

    where the argument is a complex number :cite:p:`fried:1961`.

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
    :cite:p:`fried:1961`.

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
    zeta: complex | np.ndarray | u.Quantity[u.dimensionless_unscaled],
) -> complex | np.ndarray | u.Quantity[u.dimensionless_unscaled]:
    r"""
    Calculate the plasma dispersion for a given 1D distribution function.

    The plasma dispersion function is defined as:

    .. math::
        Z(ζ) = π^{-0.5} \int_{-∞}^{+∞}
        \frac{∂f(x)/∂x}{x-ζ} dx

    where the argument is a complex number :cite:p:`fried:1961`.

    Parameters
    ----------
    zeta : |array_like| or |Quantity|
        The real or complex value to be provided as an argument to the
        plasma dispersion function. This is the ratio of the wave's phase
        velocity to the thermal velocity.

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
    >>> from plasmapy.dispersion import plasma_dispersion_1D_dist
    >>> plasma_dispersion_1D_dist(0)
    np.complex128(1.77245385...j)
    >>> plasma_dispersion_1D_dist(1 + 1j)
    np.complex128(-0.36905845...+0.54014504...j)
    >>> plasma_dispersion_1D_dist([0.3, 0.7 + 2.3j])
    array([-0.56526333+1.61990085j, -0.09995023+0.37685142j])
    """
    try:
        # derivative of 1D distribution function to be integrated
        deriv_dist_func = np.diff()
        # whole function to be integrated
        func_integ = 
        # integration limits
        pole = 
        standoff = 
        pole_neg = pole - standoff
        pole_pos = pole + standoff
        # integral from -inf to pole
        neg_integral = quad(func=func_integ,
                            a=-np.inf,
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
        pos_integral = quad(func=func_integ,
                            a=pole_pos,
                            b=np.inf,
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
        cauchy_integral = quad(func=func_integ,
                               a=pole_neg,
                               b=pole_pos,
                               epsabs=1.49e-08,
                               epsrel=1.49e-08,
                               limit=50,
                               points=None,
                               weight="cauchy",
                               wvar=None,
                               wopts=None,
                               maxp1=50,
                               limlst=50,
                               complex_func=True)
        # combine it all together
        total_integral = neg_integral + pos_integral + cauchy_integral
        return total_integral
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
        

def plasma_dispersion_1D_dist_deriv(
    zeta: complex | np.ndarray | u.Quantity[u.dimensionless_unscaled],
) -> complex | np.ndarray | u.Quantity[u.dimensionless_unscaled]:
    r"""
    Calculate the derivative of the plasma dispersion function generalized
    for arbitrary (well-behaved)1D distribution functions.

    The derivative of the plasma dispersion function is:

    .. math::
        Z'(ζ) = \frac{∂Z}{∂ζ}

    where the argument :math:`ζ` is a complex number
    :cite:p:`fried:1961`.

    Parameters
    ----------
    zeta : |array_like| or |Quantity|
        The real or complex value to be provided as an argument to the
        plasma dispersion function. This is the ratio of the wave's phase
        velocity to the thermal velocity.

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
    >>> plasma_dispersion_1D_dist_deriv(0)
    np.complex128(-2+0j)
    >>> plasma_dispersion_1D_dist_deriv(1j)
    np.complex128(-0.48425568771737604+0j)
    >>> plasma_dispersion_1D_dist_deriv(-1.52 + 0.47j)
    np.complex128(0.165871331...+0.4458797880...j)
    """
    try:
        # locally computing the numerical derivative
        z_deriv = 
        return z_deriv
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