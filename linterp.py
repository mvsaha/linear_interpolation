import numpy as np
import numba


@numba.njit
def find_next_finite(arr, i):
    i += 1
    while i < arr.size:
        if np.isfinite(arr[i]):
            return i
        i += 1
    return arr.size


sig = [(numba.float64[:], numba.float64[:], numba.float64[:],
        numba.float64[:], numba.float64[:], numba.float64[:]), 
      
       (numba.float32[:], numba.float32[:], numba.float32[:],
        numba.float32[:], numba.float32[:], numba.float32[:])]


@numba.guvectorize(sig, '(n),(n),(m),(),()->(m)', nopython=True)
def array_linear_interpolation_1d(x, y, x_new, _extrap, _fill_val, out):
    """Find the interpolated values of y at the positions in x_new.
    
    Arguments
    ---------
    x : Strictly increasing 1d numpy array
        Positions of reference values. ('Known' data)
    y : 1d array of values to interpolate, identical in shape to `x`
        Values at reference positions. ('Known' data)
    x_new : Strictly increasing 1d numpy array
        The positions where we want to interpolate `y` values at. This must
        be increasing along the last dimension.
    out : [None] | 1d array
        Optional array to place interpolated values into. Must be the same size
        as `x`. If this is None then a new array will be created and returned.
    extrap : [0] | 1 | 2
        Flag indicating how to fill outside of the range (min(x), max(x)):
        0 - Fill with `fill_val`, by default this is NaN
        1 - Fill with the closest value (e.g. `y` value at `min(x)` where
            `x_new` values are less than min(`x`) or `y` value at `max(x)`
            where `x_new` values are greater than `max(x)`)
        2 - Use linear extrapolation. If only one non-NaN value is present
            in `y`, then that value will be used.
        3 - Raise a ValueError if any of `x_new` are outside of the range
            of `x`
    fill_val : [np.nan] | scalar
        The value to assign to extrapolated values when extrap is set to 0.
        This argument is ignored in other cases.
    
    """
    extrap = _extrap[0]
    fill_val = _fill_val[0]
    
    if not (extrap == 0 or extrap == 1 or extrap == 2 or extrap == 3):
        raise ValueError('`extrap` must be either 0, 1, 2, or 3.')
    
    n = x.size
    
    if out is None:
        y_new = np.empty(x_new.size)
    else:
        y_new = out
    
    if y_new.size == 0:
        return #y_new
    
    n_new = y_new.size
    
    if np.isnan(y[0]):
        i0 = find_next_finite(y, 0)
    else:
        i0 = 0
    
    if i0 == n:  # No finite y values
        if extrap == 0:
            y_new[:] = fill_val
        else:
            y_new[:] = np.nan
        return #y_new
    
    x0, y0 = x[i0], y[i0]
    i1 = find_next_finite(y, i0)
    
    # Case: There is only one non-NaN value in `y`
    if i1 == n:
        if extrap == 0:
            j = 0
            while j < n_new:
                if x_new[j] >= x0:
                    break
                y_new[j] = fill_val
                j += 1
            while j < n_new:
                if x_new[j] > x0:
                    break
                y_new[j] = y0
                j += 1
            while j < n_new:
                y_new[j] = fill_val
                j += 1
        
        elif extrap < 3:
            for j in range(0, n_new):
                y_new[j] = y0
        
        else:  # extrap == 3:
            for j in range(0, n_new):
                if x_new[j] != x0:
                    raise ValueError('`x_new` values outside of range '
                        '(x[0], x[-1]) are not allowed when extrap=3.')
                y_new[j] = y0
        
        return #y_new
    
    # Extrapolate all x_new values less than x0
    x1, y1 = x[i1], y[i1]  # We know that y1 is finite here
    j = 0
    if extrap == 0:
        while j < n_new and x_new[j] < x0:
            y_new[j] = fill_val
            j += 1
    elif extrap == 1:
        while j < n_new and x_new[j] < x0:
            y_new[j] = y0
            j += 1
    elif extrap == 2:
        dx, dy = x1 - x0, y1 - y0
        while j < n_new and x_new[j] < x0:
            y_new[j] = y0 + (dy * (x_new[j] - x0) / dx)
            j += 1
    else:
        if x_new[j] < x0:
            raise ValueError('`x_new` values outside of range '
                    '(x[0], x[-1]) are not allowed when extrap=3.')
    
    # Edge case xn == x0
    while j < n_new and x_new[j] == x0:
        y_new[j] = y0
        j += 1
    
    # Interpolation
    dx, dy = x1 - x0, y1 - y0  # Only update the differences when needed
    for j in range(j, n_new):
        xn = x_new[j]
        while xn > x1:
            i0 = i1
            i1 = find_next_finite(y, i1)
            if i1 == n:
                break
            x0, y0 = x1, y1
            x1, y1 = x[i1], y[i1]
            if not x0 < x1:
                raise ValueError("`x` must be strictly increasing without NaNs.")
            dx, dy = x1 - x0, y1 - y0
        
        if i1 == n:
            break  # x0, y0, x1, y1 still point to valid points for extrap
        
        y_new[j] = y0 + (dy * (xn - x0) / dx)
    
    # Edge case xn == x1
    while j < n_new and x_new[j] == x1:
        y_new[j] = y1
        j += 1
    
    if j == n_new:
        return #y_new
    
    # Extrapolate all x_new values greater than x1
    if extrap == 0:
        while j < n_new:
            y_new[j] = fill_val
            j += 1
    elif extrap == 1:
        while j < n_new:
            y_new[j] = y1
            j += 1
    elif extrap < 3:
        while j < n_new:
            y_new[j] = y0 + (dy * (x_new[j] - x0) / dx)
            j += 1
    else:
        if x_new[j] > x1:
            raise ValueError('`x_new` values outside of range '
                    '(x[0], x[-1]) are not allowed when extrap=3.')
    
    return #y_new


def linear_interpolation(x, y, x_new, y_new=None, extrap=0, fill_val=np.nan):
    """Linear interpolate multiple along the highest dimension of x and y.
    
    Either x or y are 'broadcast' if the other is larger along lower
    dimensions.
    
    Arguments
    ---------
    x : 1+d array
        The last dimension of x gives the locations of the original measurements.
        The lower dimension sizes (i.e. x.shape[:-1]) must match those of y.
        The values of x must be strictly increasing along the last dimension.
    y : 1+d array, y.shape[:-1] = x.shape[:-1] = x_new.shape[:-1]
        The values at the locations in x.
    x_new : 1+d array
        The locations we want to interpolate y values. This must be increasing.
    y_new : [None] | array such that shape[:-1] == x_new.shape[:-1]
        If an array is supplied then the result of interpolation is written
        to this output. If this is None then the interpolation output is
        returned from this function.
    extrap : [0] | int
        The extrapolation behavior:
        0 - Fill all extrapolated values with the fill value (NaN by default).
        1 - Extrapolate constant end values.
        2 - Linear extrapolation from two end values.
        3 - Raise an error if extrapolation is encountered.
    fill_val : [np.nan] | scalar
        The value to fill missing interpolation values with. This only applies
        if extrap == 0.
    
    Returns
    -------
    y_new - The values of y interpolated/extrapolated at the positions given
    in `x_new`. If `y_new` was specified in the input then this function
    will simply return the same array (that is mutated).
    
    Notes
    -----
    An error will be raised if `x` or `x_new` are not strictly increasing.
    `x` and `x_new` are not allowed to have NaN values. An error will be raised
    if NaNs are encountered in either array but intermediate results may
    be written to `out` before this happens.
    
    `y` is allowed to have NaN values. If all of the values in `y` are NaN then
    an array of NaNs will be returned. If there is only one non-NaN in `y` then
    the return values depend on the extrapolation method used.
    """
    ret_y = True if y_new is None else False
    
    y_new = array_linear_interpolation_1d(x, y, x_new, extrap, fill_val, y_new)
    
    return y_new if ret_y else None