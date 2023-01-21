# -*- coding: utf-8 -*-
"""
________________TITLE________________
PHYS20161 - Assignment 2 - Nuclear Decay
-------------------------------------
This script performs main function is to detemine the best fit for two
parameters (halflives of Sr and Rb), according to a known nuclear activity
function and two sets of data.

The General Schematic is as follows:

1) Read, Filter (anomalous and nan) and sort 3 column data (x, y, uncertainty)
files.
    - Plotting this progress
    - Notifying of any data points that have been altered
2) From starting guesses, determine a set of the two parameters that have the
lowest associated chi value.
    - Plotting a 2D graph of the chi values against various values of the
      parameters
3) Determine the 1 sigma uncertainty on these fitted parameters
    - plotting the + 1 sigma ellipse on the contour plot
4) Plot a graph of the future predicted activity, taking a certain time value
to print the predicted activity for.
    - It will save a copy of this the filtered data, and the predicted trend,
      as a csv.

The data reading is designed to be versatile, but the code will terminate if
there is an issue.
It should be noted here as well, that the "resolution_input" of
"minimising" function is related to the runtime**2 {O(2)}, increasing this
value is likely to increase the runtime drastically, with limited increase in
accuracy.

Last Updated: 9/12/2022
@author: Vadan Khan UID: 10823198

"""
# %% Import Statements
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.optimize import fmin

# %% Declaring Constants
FILENAME1 = 'Nuclear_data_1.csv'
FILENAME2 = 'Nuclear_data_2.csv'
DELIMITER = ','
COMMENT_MARKER = '%'

INITIAL_RB_DECAY_CONSTANT = 0.0005
INITIAL_SR_DECAY_CONSTANT = 0.005

INITIAL_SR_MOLES = 10**-6
AVOGADRO_CONSTANT = const.Avogadro

# %% Function Definitions


def number_of_atoms(moles_input):
    """
    Determines number of atoms, from input moles.

    Parameters
    ----------
    moles_input : float


    Returns
    -------
    float

    """
    return moles_input * AVOGADRO_CONSTANT


def decay_constant_seconds2hrs(decay_constant_input):
    """
    Determines Decay Constant in hours, given in seconds

    Parameters
    ----------
    decay_constant_input : float


    Returns
    -------
    float

    """
    return decay_constant_input * 3600


def decay_constant_hrs2seconds(decay_constant_input):
    """
    Determines Decay Constant in seconds, given in hours

    Parameters
    ----------
    decay_constant_input : float


    Returns
    -------
    float

    """
    return decay_constant_input / 3600


def read_data(data_input_name, delimiter_input, comment_marker_input):
    """
    Reads 3 column data file (csv or txt), with a given delimiter and
    comment marker. Converts to numpy array, with 3 columns.
    Then sorts the data according to the first column. This will remove lines
    of data that are not numbers, indicated those that have been eliminated.

    Parameters
    ----------
    data_input_name : string
    delimiter_input : string
    comment_marker_input : string

    Returns
    -------
    numpy_array[floats]

    """
    print("\nReading File: ", data_input_name)
    print("==================================================")
    try:
        data_intake = np.genfromtxt(
            data_input_name, delimiter=delimiter_input,
            comments=comment_marker_input)
    except ValueError:
        return 1
    except NameError:
        return 1
    except TypeError:
        return 1
    except IOError:
        return 1
    index = 0
    eliminated_lines = 0
    initial_length = len(data_intake[:, 0])
    for line in data_intake:
        if np.isnan(line[0]) or np.isnan(line[1]) or np.isnan(line[2]):
            print('Deleted line {0}: '.format(index + 1 + eliminated_lines))
            print(line)
            data_intake = np.delete(data_intake, index, 0)
            index -= 1
            eliminated_lines += 1
        index += 1
    if eliminated_lines == 0:
        print("[no lines eliminated]")
    print("==================================================")
    print('Initial Array Length: {0}'.format(initial_length))
    print('Final Array Length: {0}'.format(len(data_intake[:, 0])))
    print('Data Read with {0} lines removed'.format(eliminated_lines))
    print("==================================================")
    return data_intake


def create_plot(data_input, title_input, x_axis_input, y_axis_input):
    """
    Creates a plot of input 3 column data (x axis, y axis, uncertainty)
    with an inputted title, x_axis and y_axis name.

    Parameters
    ----------
    data_input : numpy array, [float, float, float]
        Data should be in order independent variable, dependent variable,
        uncertainty
    title_input : string
    x_axis_input : string
    y_axis_input : string

    Returns
    -------
    0

    """

    fig = plt.figure()

    axes = fig.add_subplot(111)

    axes.set_title(title_input)
    axes.set_xlabel(x_axis_input)
    axes.set_ylabel(y_axis_input)
    axes.errorbar(data_input[:, 0], data_input[:, 1],
                  data_input[:, 2], fmt='.', color='k')
    plt.savefig(title_input + '.png', dpi=777)
    plt.show()
    return 0


def comparison_plot(data_input, function_input, function_parameter_1,
                    function_parameter_2, function_inital_atoms, plot_title,
                    plot_x_axis, plot_y_axis):
    """
    Creates a plot of input 3 column data (x axis, y axis, uncertainty),
    and also plots a function on the same axes.
    with an inputted title, x_axis and y_axis name.

    Parameters
    ----------
    data_input : numpy array, [float, float, float]
        Data should be in order independent variable, dependent variable,
        uncertainty
    plot_title : string
    plot_x_axis : string
    plot_y_axis : string
    function_parameter_1: float
    function_parameter_2: float
    function_inital_atoms: float

    Returns
    -------
    0

    """
    x_values = np.linspace(data_input[:, 0][np.argmin(data_input[:, 0])],
                           data_input[:, 0][np.argmax(data_input[:, 0])], 777)
    fig = plt.figure()

    axes = fig.add_subplot(111)

    axes.set_title(plot_title)
    axes.set_xlabel(plot_x_axis)
    axes.set_ylabel(plot_y_axis)
    axes.errorbar(data_input[:, 0], data_input[:,
                                               1], data_input[:, 2], fmt='.',
                  color='k', label='Data Points')
    axes.plot(x_values, function_input(x_values, function_parameter_2,
                                       function_parameter_1,
                                       function_inital_atoms), color='r',
              label='Trend Prediction')
    plt.savefig(plot_title + '.png', dpi=777)
    plt.legend()
    plt.show()
    return 0


def combine_data_vertically(top_data, bottom_data):
    """
    Combines 2 numpy arrays, adding the bottom_data to the bottom of the first.
    Then sorts the data according to the first column. This will terminate
    if the arrays do not have the same width.

    Parameters
    ----------
    top_data : numpy_array
    bottom_data : numpy_array

    Returns
    -------
    sorted_array : numpy_array

    or if unsucessful:
        1 : float

    """
    if len(top_data[0, :]) == len(bottom_data[0, :]):
        data_output = np.vstack((top_data, bottom_data))
        sorted_array = data_output[data_output[:, 0].argsort()]
        return sorted_array
    print("Arrays are not same width!")
    return 1


def function(t_input, decay_constant_out, decay_constant_in, n_initial_in):
    """
    Function of amount of intermediate isotope, when initially mother isotope
    is present.

    NOTE THAT IT TAKES t_input AND DECAY_CONSTANTS IN HOURS
    OUTPUTS IN TBq

    Parameters
    ----------
    t_input : float
    decay_constant_out : float
    decay_constant_in : float
    n_initial_in : float

    Returns
    -------
    float

    """
    answer = n_initial_in * ((decay_constant_out * decay_constant_in) /
                             (decay_constant_out-decay_constant_in))*1/3600 * \
        (np.exp(-decay_constant_in * t_input)
         - np.exp(-decay_constant_out * t_input))*10**(-12)

    return answer


def filter_outliers(data_set, function_input, function_parameter_1,
                    function_parameter_2, function_inital_atoms,
                    outlier_threshold):
    """
    Expected data in (x_value, y_value, uncertainty) column array.
    will compare data to a prediciton function. Will eliminate lines of data
    that are above a set "outlier_threshold" away from the prediction,
    notifying what is eliminated.

    Parameters
    ----------
    data_set : array
    function_input : function
    function_parameter_1 : float
    function_parameter_2 : float
    function_inital_atoms : float
    outlier_threshold : float

    Returns
    -------
    filtered_array[floats]

    """
    print("\nFiltering File: ")
    print("==================================================")

    prediction_range = function_input(data_set[:, 0], function_parameter_2,
                                      function_parameter_1,
                                      function_inital_atoms)
    dummyerrs = np.zeros(len(data_set[:, 0]))
    comparison_set = np.column_stack((data_set[:, 0], prediction_range,
                                      dummyerrs))
    initial_length = len(data_set)
    differences = abs(data_set[:, 1] - comparison_set[:, 1])
    outliers = np.where(differences > outlier_threshold)

    if len(outliers[0]) == 0:
        print("[no lines eliminated]")
        filtered_array = data_set
    elif len(outliers[0]) > 0:
        filtered_array = np.delete(data_set, outliers[0], axis=0)
        for i in outliers[0]:
            print('Data line {0} removed: '.format(i), data_set[i, :])
            print('Compared to: ', comparison_set[i, :])
    else:
        print("filtering error")
    print("==================================================")
    print('Initial Array Length: {0}'.format(initial_length))
    print('Final Array Length: {0}'.format(len(filtered_array[:, 0])))
    print('Number of anomlies identified and removed: {0}'.format(
        len(outliers[0])))
    print("==================================================")

    return filtered_array


def fix_zero_uncertainties(observation_data):
    """
    This will look for any uncertainties in a data set.
    If there is a zero uncertainty, it will replace it with the previous
    uncertainty (if first data point will use next uncertainty).
    This will also print out the changes to uncertainty.

    Parameters
    ----------
    observation_data : array[x_values, y_values, uncertainties]

    Returns
    -------
    observation_data : array[x_values, y_values, uncertainties]

    """
    zero_uncertainties_location = np.where(observation_data[:, 2] == 0)
    if len(zero_uncertainties_location[0]) > 0:
        print("\nData set contains uncertainties that = 0! Correcting:")
        print("==================================================")
        print("initial uncertainties", observation_data[:, 2])
        for zero_loc in zero_uncertainties_location[0]:
            if zero_loc.size > 0:
                observation_data[zero_loc,
                                 2] = observation_data[zero_loc - 1, 2]
            elif zero_loc.size == 0:
                observation_data[zero_loc,
                                 2] = observation_data[zero_loc + 1, 2]
            else:
                print("fixing 0s error")
        print("==================================================")
        print("final uncertainties", observation_data[:, 2])
        print("==================================================")
    return observation_data


def chi_square(observation, observation_uncertainty, prediction):
    """
    Returns the chi squared.

    Parameters
    ----------
    observation : numpy array of floats
    observation_uncertainty : numpy array of floats
    prediction : numpy array of floats


    Returns
    -------
    float
    """

    return np.sum((observation - prediction)**2 / observation_uncertainty**2)


def mesh_arrays(x_array, y_array):
    """
    Returns two meshed arrays of size len(x_array)
    by len(y_array)

    Parameters
    ----------
    x_array array[floats]
    y_array array[floats]


    Returns
    -------
    array (mesh)

    """
    x_array_mesh, y_array_mesh = np.meshgrid(x_array, y_array)
    return x_array_mesh, y_array_mesh


def minimising(function_input, parameter_1_start, parameter_2_start,
               range_input_1, range_input_2,
               observation_data, n_input, resolution_input):
    """
    Takes in an initial guess to parameters for a certain function,
    and a bounds on the ranges of parameter values to try. Will optimise the
    parameters for a function to best fit a set of data, with the number of
    parameters plotted for each parameter = resolution input.

    NOTE: runtime is proportional to resolution_input O(n^2), or
    resolution_input^2. Runtime can quickly increase due to this variable.

    Parameters
    ----------
    function_input : function
    parameter_1_start : float
    parameter_2_start : float
    range_input_1_lower : float
    range_input_1_upper : float
    range_input_2_lower : float
    range_input_2_upper : float
    observation_data : numpy_array (x_values, y_values, uncertainties)
    n_input : float
    resolution_input : int

    Returns
    -------
    minimum_chi : float
    fitted_parameter_1 : float
    fitted_parameter_2 : float
    x_values_mesh : numpy_array (mesh)
    y_values_mesh : numpy_array (mesh)
    parameter_values_mesh : numpy_array (mesh)

    """
    def chi_value(xyvalues):
        x_value = xyvalues[0]
        y_value = xyvalues[1]
        chi = chi_square(observation_data[:, 1], observation_data[:, 2],
                         function_input(observation_data[:, 0], x_value,
                                        y_value, n_input))
        return chi

    print("\nDetermining Optimimum Parameter Value: ")
    print("==================================================")

    fit_results = fmin(chi_value, (parameter_1_start, parameter_2_start),
                       full_output=True)

    [fitted_parameter_1, fitted_parameter_2] = fit_results[0]
    minimum_chi = fit_results[1]
    print("==================================================")

    parameter_range_1 = np.linspace(
        fitted_parameter_1 - range_input_1,
        fitted_parameter_1 + range_input_1,
        resolution_input)
    parameter_range_2 = np.linspace(
        fitted_parameter_2 - range_input_2,
        fitted_parameter_2 + range_input_2,
        resolution_input)

    x_values_mesh, y_values_mesh = mesh_arrays(
        parameter_range_1, parameter_range_2)

    parameter_values_mesh = np.empty((0, len(parameter_range_1)))

    for y_value in parameter_range_2:  # select y value
        x_line = np.empty((0, len(parameter_range_1)))
        for x_value in parameter_range_1:  # vary x and create line
            if x_value != y_value:
                point = chi_square(observation_data[:, 1],
                                   observation_data[:, 2],
                                   function_input(observation_data[:, 0],
                                                  y_value, x_value, n_input))
            elif x_value == y_value:

                point = chi_square(observation_data[:, 1],
                                   observation_data[:, 2],
                                   function_input(observation_data[:, 0],
                                                  y_value,
                                                  parameter_range_1[x_value+1],
                                                  n_input))
            else:
                print("forming chis mesh error")
            x_line = np.append(x_line, point)
        parameter_values_mesh = np.vstack((parameter_values_mesh, x_line))

    return minimum_chi, fitted_parameter_1, fitted_parameter_2, x_values_mesh,\
        y_values_mesh, parameter_values_mesh


def ellipse_identify(x_input_mesh, y_input_mesh, chi_mesh_input,
                     chi_minimum_input, tolerance):
    """
    From an inputted plot of chi values against x and y, this will look for
    values of chi that are 1 above the minimum to find the ellipse.
    Then it will detemine the highest x and y values of this ellipse.
    The tolerance of being 1 above can be adjusted, as this code will count
    values that are:
        1 - tolerance < x < 1 + tolerance
    above the minimum.


    Parameters
    ----------
    x_input_mesh : numpy_array (mesh)
    y_input_mesh : numpy_array (mesh)
    chi_mesh_input : numpy_array (mesh)
    chi_minimum_input : float
    tolerance : float

    Returns
    -------
    minimum_x : float
    maximum_x : float
    minimum_y : float
    maximum_y : float

    """
    print("\nDetermining Uncertainty: ")
    print("==================================================")

    ellipse_values_index = np.where(
        (chi_mesh_input < chi_minimum_input + 1 + tolerance)
        & (chi_mesh_input > chi_minimum_input + 1 - tolerance))
    x_values = x_input_mesh[ellipse_values_index]
    y_values = y_input_mesh[ellipse_values_index]
    minimum_x = x_values[np.argmin(x_values)]
    minimum_y = y_values[np.argmin(y_values)]
    maximum_x = x_values[np.argmax(x_values)]
    maximum_y = y_values[np.argmax(y_values)]
    print('Minimum Sr, Maximum Sr : {0:.5f}, {1:.5f}'
          .format(
              minimum_x, maximum_x))
    print('Minimum Rb Maximum Rb : {0:.5f}, {1:.5f}'.format(
        minimum_y, maximum_y))
    print("==================================================")
    return minimum_x, maximum_x, minimum_y, maximum_y


def uncertainty_from_range(minimum_input, maxiumum_input):
    """
    returns uncertainty, given an upper and lower bound.

    Parameters
    ----------
    minimum_input : float
    maxiumum_input : float


    Returns
    -------
    float

    """
    return (maxiumum_input - minimum_input) / 2


def contour_plot(parameter_1_name_input, parameter_2_name_input,
                 fitted_parameter_1_input, fitted_parameter_2_input,
                 minimum_chi_input, x_mesh_input,
                 y_mesh_input, chi_values_input, min_parameter_1,
                 max_parameter_1, min_parameter_2, max_parameter_2,
                 contour_input):
    """
    A function that will create a contour plot of 2 parameter minimised chi
    fitting. Requires the minimum chi value and associated parameters, the
    upper and lower bounds of the parameters, and a premade x_mesh, y_mesh
    and mesh with the chi_value at each point. Will take inputted names of axes
    and number of contour lines.

    Parameters
    ----------
    paramater_1_name_input : string
    parameter_2_name_input : string
    fitted_parameter_1_input : float
    fitted_parameter_2_input : float
    minimum_chi_input : float
    x_mesh_input : numpy_array (mesh)
    y_mesh_input : numpy_array (mesh)
    chi_values_input : numpy_array (mesh)
    min_parameter_1 : float
    max_parameter_1 : float
    min_parameter_2 : float
    max_parameter_2 : float
    contour_input : int

    Returns
    -------
    0

    """
    intial_figure = plt.figure()

    plot = intial_figure.add_subplot(111)

    plot.set_title('Chi Values Depending on Parameters')
    plot.set_xlabel(parameter_1_name_input)
    plot.set_ylabel(parameter_2_name_input)

    plot.axhline(y=max_parameter_2, color='k', linestyle='--', alpha=0.7)
    plot.axhline(y=min_parameter_2, color='k', linestyle='--', alpha=0.7)
    plot.axvline(x=max_parameter_1, color='k', linestyle='--', alpha=0.7)
    plot.axvline(x=min_parameter_1, color='k', linestyle='--', alpha=0.7)

    main_plot = plot.contourf(
        x_mesh_input, y_mesh_input, chi_values_input,
        contour_input, cmap='Blues')

    plot.scatter(fitted_parameter_1_input, fitted_parameter_2_input,
                 color='green',
                 label=r'$\chi^2_{{\mathrm{{min.}}}} = ${0:1.5g}'
                 .format(minimum_chi_input))

    sigma_levels = [minimum_chi_input + 1, minimum_chi_input + 2,
                    minimum_chi_input + 3]

    contour_levels = plot.contour(x_mesh_input, y_mesh_input,
                                  chi_values_input,
                                  levels=sigma_levels,
                                  colors=('red', 'orange', 'yellow'))

    plt.legend()

    contour_levels.clabel(fmt='%1.5g', colors='k', fontsize=8)
    plt.colorbar(main_plot, shrink=0.8, location='left', pad=0.14)

    plt.tight_layout()
    plt.savefig('Chi_Value_Contour_Plot.png', dpi=777)
    plt.show()

    return 0


def reduced_chi_squared(chi_squared_input, data_points, parameters):
    """
    returns reduced chi squared, given number of parameters and data points,
    and an inital total chi value.

    Parameters
    ----------
    chi_squared_input : float
    data_points : int
    parameters : int


    Returns
    -------
    float

    """
    return chi_squared_input / (data_points - parameters)


def decay_constant2half_life(decay_constant_input):
    """
    returns half life in seconds, given decay constant in s^-1

    Parameters
    ----------
    decay_constant_input : float


    Returns
    -------
    float

    """
    return np.log(2)/decay_constant_input


def halflife_secs2mins(half_life_seconds_input):
    """
    returns halflife in minutes, if given in seconds

    Parameters
    ----------
    half_life_seconds_input : float


    Returns
    -------
    float

    """
    return half_life_seconds_input / 60


def printing_fitting_results(decay_constant_in_input, decay_constant_out_input,
                             uncertainty_in_input, uncertainty_out_input,
                             chi_minimum_input, data_input):
    """
    This will take the fitted results for parameters and uncertainties, and
    convert them to desired units and print them. It will also print the
    reduced chi value.

    Parameters
    ----------
    decay_constant_in_input : float
    decay_constant_out_input : float
    uncertainty_in_input : float
    uncertainty_out_input : float
    chi_minimum_input : float
    data_input : numpy_array[floats]

    Returns
    -------
    0
    """
    reduced_chi_result = reduced_chi_squared(chi_minimum_input,
                                             len(data_input[:, 0]), 2)

    decay_constant_sr_seconds = decay_constant_hrs2seconds(
        decay_constant_in_input)
    decay_constant_rb_seconds = decay_constant_hrs2seconds(
        decay_constant_out_input)
    decay_uncertainty_sr_seconds = decay_constant_hrs2seconds(
        uncertainty_in_input)
    decay_uncertainty_rb_seconds = decay_constant_hrs2seconds(
        uncertainty_out_input)
    halflife_sr_seconds = decay_constant2half_life(decay_constant_sr_seconds)
    halflife_rb_seconds = decay_constant2half_life(decay_constant_rb_seconds)
    halflife_sr_minutes = halflife_secs2mins(halflife_sr_seconds)
    halflife_rb_minutes = halflife_secs2mins(halflife_rb_seconds)
    uncertainty_halflife_sr_minutes = (uncertainty_in_input /
                                       decay_constant_in_input) * \
        halflife_sr_minutes
    uncertainty_halflife_rb_minutes = (uncertainty_out_input
                                       / decay_constant_out_input) * \
        halflife_rb_minutes

    print("\n\nFINAL FITTING RESULTS: ")
    print("==================================================")
    print('Reduced Chi value is: {0: .2f}, with Total Chi : {1: .7f}'.format(
        reduced_chi_result, chi_minimum_input))
    print("==================================================")
    print('{0}: {1:4.7f} +- {4:1.3g} hrs^-1\n{2}: {3:4.7f} +- {5:1.3g} hrs^-1'
          .format('Sr decay constant', decay_constant_in_input,
                  'Rb decay constant', decay_constant_out_input,
                  uncertainty_in_input, uncertainty_out_input))
    print('{0}: {1:1.3g} +- {4:1.0g} s^-1\n{2}: {3:1.3g} +- {5:1.0g} s^-1'
          .format('Sr decay constant',
                  decay_constant_sr_seconds,
                  'Rb decay constant', decay_constant_rb_seconds,
                  decay_uncertainty_sr_seconds,
                  decay_uncertainty_rb_seconds))
    print("==================================================")
    print('{0}: {1:1.3g} +- {4:1.0g} mins\n{2}: {3:1.3g} +- {5:1.0g} mins'
          .format('Sr halflife',
                  halflife_sr_minutes,
                  'Rb halflife', halflife_rb_minutes,
                  uncertainty_halflife_sr_minutes,
                  uncertainty_halflife_rb_minutes))
    print("==================================================")
    return 0


def frac_propagated_error_on_function(uncertainty_in_input,
                                      uncertainty_out_input,
                                      decay_constant_in_input,
                                      decay_constant_out_input,
                                      time_input):
    """
    Returns the fractional error on the activity for the equation that
    calculates activity of an intermediate isotope, if there are uncertainties
    in the two decay constant parameters.

    This expects decay constants in hours^-1

    Parameters
    ----------
    uncertainty_in_input : float
    uncertainty_out_input : float
    decay_constant_in_input : float
    decay_constant_out_input : float
    time_input : float

    Returns
    -------
    error : float

    """

    first_term = 1/decay_constant_out_input - 1 / \
        (decay_constant_out_input-decay_constant_in_input) \
        + time_input*(np.exp(-decay_constant_out_input * time_input) /
                      (np.exp(-decay_constant_in_input*time_input)
                       - np.exp(-decay_constant_out_input*time_input)))
    second_term = 1/decay_constant_in_input + 1 / \
        (decay_constant_out_input-decay_constant_in_input) \
        - time_input*(np.exp(-decay_constant_in_input * time_input)
                      / (np.exp(-decay_constant_in_input*time_input)
                         - np.exp(-decay_constant_out_input*time_input)))

    error = np.sqrt((first_term*uncertainty_out_input)**2 +
                    (second_term*uncertainty_in_input)**2)
    return error


def future_prediction(decay_constant_in_input, decay_constant_out_input,
                      uncertainty_in_input, uncertainty_out_input, atoms_input,
                      data_input, time_value_input, resolution_trend):
    """
    This will take the fitted parameters and plot a predicted future trend.
    It will also print a certain prediction for a given time, where the graph
    will have a width twice this.

    It will save a copy of the prediction as a csv, with resolution of the
    graph and adjustable variable "resolution_trend"

    This expects the parameters to be in units of hrs^-1

    Parameters
    ----------
    ecay_constant_in_input : float
    decay_constant_out_input : float
    uncertainty_in_input : float
    uncertainty_out_input : float
    atoms_input : float
    data_input : float
    time_value_input : float
    resolution_trend : float

    Returns
    -------
    0
    """
    prediction = function(
        time_value_input, decay_constant_out_input, decay_constant_in_input,
        atoms_input)
    uncertainty_prediction = prediction*frac_propagated_error_on_function(
        uncertainty_in_input, uncertainty_out_input, decay_constant_in_input,
        decay_constant_out_input, time_value_input)
    print("\nPrediction Results: ")
    print("==================================================")
    print('At time {0: .2f}hrs, the fitted parameters predict an activity'
          '{1: 1.3g} +- {2: 1.0g} TBq'.format(
              time_value_input, prediction, uncertainty_prediction))
    print("==================================================")

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.set_title('Predicted Future Trend')
    axes.set_xlabel('Time (hrs)')
    axes.set_ylabel('Activty (TBq)')
    data_times = data_input[:, 0]
    time_range = np.linspace(
        data_times[np.argmin(data_times)], 2*time_value_input,
        resolution_trend)
    axes.plot(time_range, function(time_range, decay_constant_out_input,
                                   decay_constant_in_input, atoms_input),
              color='r')
    axes.scatter(time_value_input, prediction, color='k',
                 label='Prediction at {0:1.3g} hrs = {1:1.3g} TBq'
                 .format(time_value_input, prediction),
                 marker='x')
    plt.savefig('Predicted_Future_Trend.png', dpi=777)
    plt.legend()
    plt.show()

    function_array = np.array(function(time_range, decay_constant_out_input,
                                       decay_constant_in_input, atoms_input))
    function_data = np.column_stack((time_range, function_array))
    np.savetxt("Filtered_data.csv", data_input, delimiter=",")
    np.savetxt("Future Prediction.csv", function_data, delimiter=",")

    return 0

# %% Main


def main():
    """
    Main function. Will run all main_code, and if a 1 is returned, it will
    notify of a reading file error.
    """
    return_code = main_code()
    if return_code == 1:
        print("Error Reading File")
        sys.exit()
        return 1
    return 0


def main_code():
    """
    Contains all executed code. Should return 1 in the instance of a reading
    error.

    """
    # %% Read Data
    data_raw1 = read_data(FILENAME1, DELIMITER, COMMENT_MARKER)
    if isinstance(data_raw1, int):
        if data_raw1 == 1:
            return 1
    data_raw2 = read_data(FILENAME2, DELIMITER, COMMENT_MARKER)
    if isinstance(data_raw2, int):
        if data_raw2 == 1:
            return 1
    adjusted_rb_constant = \
        decay_constant_seconds2hrs(INITIAL_RB_DECAY_CONSTANT)
    adjusted_sr_constant = \
        decay_constant_seconds2hrs(INITIAL_SR_DECAY_CONSTANT)
    initial_atoms = number_of_atoms(INITIAL_SR_MOLES)

    # %%    initial data plots
    data_raw = combine_data_vertically(data_raw1, data_raw2)
    create_plot(data_raw, 'Plotted Data', 'Time (hrs)', 'Activity (TBq)')

    comparison_plot(data_raw, function, adjusted_sr_constant,
                    adjusted_rb_constant, initial_atoms,
                    'Data vs Initial Parameter Prediction',
                    'Time (hrs)', 'Activity (TBq)')

    # %%    Filtering

    data_raw = filter_outliers(data_raw, function, adjusted_sr_constant,
                               adjusted_rb_constant, initial_atoms, 100)

    comparison_plot(data_raw, function, adjusted_sr_constant,
                    adjusted_rb_constant, initial_atoms,
                    'Filtered Data vs Initial Parameter Prediction',
                    'Time (hrs)', 'Activity (TBq)')

    data_raw = fix_zero_uncertainties(data_raw)

    # %%    Minimising
    minimum_chi_value, minimised_sr, minimised_rb, x_mesh_result, \
        y_mesh_result, chi_mesh = minimising(
            function, adjusted_sr_constant, adjusted_rb_constant, 1, 0.05,
            data_raw, initial_atoms, 300)

    # %%    Uncertainties on Parameters
    ellipse_minimum_sr, ellipse_maximum_sr, ellipse_minimum_rb, \
        ellipse_maximum_rb = ellipse_identify(
            x_mesh_result, y_mesh_result, chi_mesh, minimum_chi_value, 0.01)

    uncertainty_sr = uncertainty_from_range(
        ellipse_minimum_sr, ellipse_maximum_sr)
    uncertainty_rb = uncertainty_from_range(
        ellipse_minimum_rb, ellipse_maximum_rb)

    # %%    Fitting Results
    contour_plot('Sr decay constant ($Hrs^{-1}$)',
                 'Rb decay constant ($Hrs^{-1}$)',
                 minimised_sr, minimised_rb,
                 minimum_chi_value, x_mesh_result,
                 y_mesh_result, chi_mesh, ellipse_minimum_sr,
                 ellipse_maximum_sr,
                 ellipse_minimum_rb, ellipse_maximum_rb, 30)

    printing_fitting_results(minimised_sr, minimised_rb,
                             uncertainty_sr, uncertainty_rb,
                             minimum_chi_value, data_raw)

    # %%    Future Prediction
    future_prediction(minimised_sr, minimised_rb,
                      uncertainty_sr, uncertainty_rb, initial_atoms,
                      data_raw, 1.5, 777)

    return 0


# %% Main Execution
if __name__ == "__main__":
    main()
