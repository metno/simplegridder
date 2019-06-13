#!/usr/bin/env python3
################################################################
# read_aeolus_l2a_data.py
#
# read binary ESA L2A files of the ADM Aeolus mission
#
# this file is part of the pyaerocom package
#
#################################################################
# Created 20190104 by Jan Griesfeller for Met Norway
#
# Last changed: See git log
#################################################################

# Copyright (C) 2019 met.no
# Contact information:
# Norwegian Meteorological Institute
# Box 43 Blindern
# 0313 OSLO
# NORWAY
# E-mail: jan.griesfeller@met.no
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301, USA

"""
object to grid L2 satellite data usually coming as a point cloud

Look at file README.md for more details

"""
import os
import glob
import numpy as np

import logging
import time
import geopy.distance


# import coda

class ReadL2Data:
    """Interface for reading point clouds of satellite L2 data

    IMPORTANT:
    This module requires the coda package to be installed in the local python distribution.
    The coda package can be obtained from http://stcorp.nl/coda/

    If you are using anaconda, it can be installed using the following command:
    conda install -c stcorp coda


    Attributes
    ----------
    data : numpy array of dtype np.float64 initially of shape (10000,8)
        data point array

    Parameters
    ----------

    """
    _FILEMASK = '*.nc'
    __version__ = "0.1"
    DATASET_NAME = 'L2'
    DATASET_PATH = ''
    # Flag if the dataset contains all years or not
    DATASET_IS_YEARLY = False

    SUPPORTED_SUFFIXES = []
    SUPPORTED_SUFFIXES.append('.DBL')
    SUPPORTED_SUFFIXES.append('.nc')

    SUPPORTED_ARCHIVE_SUFFIXES = []
    SUPPORTED_ARCHIVE_SUFFIXES.append('.TGZ')
    SUPPORTED_ARCHIVE_SUFFIXES.append('.tgz')
    SUPPORTED_ARCHIVE_SUFFIXES.append('.tar')
    SUPPORTED_ARCHIVE_SUFFIXES.append('.tar.gz')

    GLOBAL_ATTRIBUTES = {}
    FILE_DIR = ''
    FILE_MASK = '*.nc'

    # variable names
    # dimension data
    _LATITUDENAME = 'latitude'
    _LONGITUDENAME = 'longitude'
    _ALTITUDENAME = 'altitude'

    _EC355NAME = 'ec355aer'
    _BS355NAME = 'bs355aer'
    _TIME_NAME = 'time'
    _NO2NAME = 'nitrogendioxide_tropospheric_column'
    _QANAME = 'qa_index'

    _LATBOUNDSNAME = 'lat_bnds'
    _LATBOUNDSSIZE = 4
    _LONBOUNDSNAME = 'lon_bnds'
    _LONBOUNDSSIZE = 4
    _O3NAME = 'ozone'

    COORDINATE_NAMES = [_LATITUDENAME, _LONGITUDENAME, _ALTITUDENAME, _LATBOUNDSNAME, _LONBOUNDSNAME]

    #     Example
    #     7.2.Cells in a
    #     non - rectangular
    #     grid
    #
    #
    # dimensions:
    # imax = 128;
    # jmax = 64;
    # nv = 4;
    # variables:
    # float
    # lat(jmax, imax);
    # lat: long_name = "latitude";
    # lat: units = "degrees_north";
    # lat: bounds = "lat_bnds";
    # float
    # lon(jmax, imax);
    # lon: long_name = "longitude";
    # lon: units = "degrees_east";
    # lon: bounds = "lon_bnds";
    # float
    # lat_bnds(jmax, imax, nv);
    # float
    # lon_bnds(jmax, imax, nv);
    #
    # The
    # boundary
    # variables
    # lat_bnds and lon_bnds
    # associate
    # a
    # gridpoint(j, i)
    # with the cell determined by the vertices (lat_bnds(j, i, n), lon_bnds(j, i, n)), n=0, .., 3.The gridpoint location, (lat(j, i), lon(j, i)), should be contained within this region.

    # variable names for the different retrievals

    _TIMEINDEX = 0
    _LATINDEX = 1
    _LONINDEX = 2
    _ALTITUDEINDEX = 3
    # for distance calculations we need the location in radians
    # so store these for speed in self.data
    # the following indexes indicate the column where that is stored
    _RADLATINDEX = 4
    _RADLONINDEX = 5
    _DISTINDEX = 6

    _DATAINDEX01 = 7
    _DATAINDEX02 = 8
    _DATAINDEX03 = 9
    _DATAINDEX04 = 10
    _QAINDEX = 11
    _LATBOUNDINDEX = 12
    _LONBOUNDINDEX = _LATBOUNDINDEX + _LATBOUNDSSIZE + 1

    _COLNO = 12 + _LATBOUNDSSIZE + _LONBOUNDSSIZE + 2

    _ROWNO = 1000000
    _CHUNKSIZE = 100000
    _HEIGHTSTEPNO = 24

    GROUP_DELIMITER = '/'

    # create a dict with the aerocom variable name as key and the index number in the
    # resulting numpy array as value.
    INDEX_DICT = {}
    INDEX_DICT.update({_LATITUDENAME: _LATINDEX})
    INDEX_DICT.update({_LONGITUDENAME: _LONINDEX})
    INDEX_DICT.update({_ALTITUDENAME: _ALTITUDEINDEX})
    INDEX_DICT.update({_TIME_NAME: _TIMEINDEX})
    INDEX_DICT.update({_NO2NAME: _DATAINDEX01})
    INDEX_DICT.update({_QANAME: _QAINDEX})
    INDEX_DICT.update({_LATBOUNDSNAME: _LATBOUNDINDEX})
    INDEX_DICT.update({_LONBOUNDSNAME: _LONBOUNDINDEX})
    INDEX_DICT.update({_EC355NAME: _DATAINDEX02})
    INDEX_DICT.update({_BS355NAME: _DATAINDEX03})

    # dixtionary to store array sizes of an element in self.data
    SIZE_DICT = {}
    SIZE_DICT.update({_LATBOUNDSNAME: _LATBOUNDSSIZE})
    SIZE_DICT.update({_LONBOUNDSNAME: _LONBOUNDSSIZE})

    # NaN values are variable specific
    NAN_DICT = {}
    NAN_DICT.update({_LATITUDENAME: -1.E-6})
    NAN_DICT.update({_LONGITUDENAME: -1.E-6})
    NAN_DICT.update({_ALTITUDENAME: -1.})
    NAN_DICT.update({_EC355NAME: -1.E6})
    NAN_DICT.update({_BS355NAME: -1.E6})
    # NAN_DICT.update({_LODNAME: -1.})
    # NAN_DICT.update({_SRNAME: -1.})

    # the following defines necessary quality flags for a value to make it into the used data set
    # the flag needs to have a HIGHER or EQUAL value than the one listed here
    # The valuse are taken form the product readme file
    QUALITY_FLAGS = {}
    QUALITY_FLAGS.update({_NO2NAME: 0.75})
    # QUALITY_FLAGS.update({_NO2NAME: 0.5}) #cloudy
    QUALITY_FLAGS.update({_O3NAME: 0.7})
    QUALITY_FLAGS.update({_EC355NAME: 0.7})
    QUALITY_FLAGS.update({_BS355NAME: 0.7})

    # PROVIDES_VARIABLES = list(RETRIEVAL_READ_PARAMETERS['sca']['metadata'].keys())
    # PROVIDES_VARIABLES.extend(RETRIEVAL_READ_PARAMETERS['sca']['vars'].keys())

    # max distance between point on the earth's surface for a match
    # in meters
    MAX_DISTANCE = 50000.
    EARTH_RADIUS = geopy.distance.EARTH_RADIUS
    NANVAL_META = -1.E-6
    NANVAL_DATA = -1.E6

    # these are the variable specific attributes written into a netcdf file
    NETCDF_VAR_ATTRIBUTES = {}
    NETCDF_VAR_ATTRIBUTES['latitude'] = {}
    # NETCDF_VAR_ATTRIBUTES['latitude']['_FillValue'] = {}
    NETCDF_VAR_ATTRIBUTES['latitude']['long_name'] = 'latitude'
    NETCDF_VAR_ATTRIBUTES['latitude']['standard_name'] = 'latitude'
    NETCDF_VAR_ATTRIBUTES['latitude']['units'] = 'degrees north'
    NETCDF_VAR_ATTRIBUTES['latitude']['bounds'] = 'lat_bnds'
    NETCDF_VAR_ATTRIBUTES['latitude']['axis'] = 'Y'
    NETCDF_VAR_ATTRIBUTES['longitude'] = {}
    # NETCDF_VAR_ATTRIBUTES['longitude']['_FillValue'] = {}
    NETCDF_VAR_ATTRIBUTES['longitude']['long_name'] = 'longitude'
    NETCDF_VAR_ATTRIBUTES['longitude']['standard_name'] = 'longitude'
    NETCDF_VAR_ATTRIBUTES['longitude']['units'] = 'degrees_east'
    NETCDF_VAR_ATTRIBUTES['longitude']['bounds'] = 'lon_bnds'
    NETCDF_VAR_ATTRIBUTES['longitude']['axis'] = 'X'
    NETCDF_VAR_ATTRIBUTES['altitude'] = {}
    # NETCDF_VAR_ATTRIBUTES['altitude']['_FillValue'] = {}
    NETCDF_VAR_ATTRIBUTES['altitude']['long_name'] = 'altitude'
    NETCDF_VAR_ATTRIBUTES['altitude']['standard_name'] = 'altitude'
    NETCDF_VAR_ATTRIBUTES['altitude']['units'] = 'm'
    NETCDF_VAR_ATTRIBUTES['bs355aer'] = {}
    NETCDF_VAR_ATTRIBUTES['bs355aer']['_FillValue'] = np.nan
    NETCDF_VAR_ATTRIBUTES['bs355aer']['long_name'] = 'backscatter @ 355nm'
    # NETCDF_VAR_ATTRIBUTES['bs355aer']['standard_name'] = 'volume_extinction_coefficient_in_air_due_to_ambient_aerosol_particles'
    NETCDF_VAR_ATTRIBUTES['bs355aer']['units'] = '1'

    NETCDF_VAR_ATTRIBUTES['ec355aer'] = {}
    NETCDF_VAR_ATTRIBUTES['ec355aer']['_FillValue'] = np.nan
    NETCDF_VAR_ATTRIBUTES['ec355aer']['long_name'] = 'extinction @ 355nm'
    NETCDF_VAR_ATTRIBUTES['ec355aer'][
        'standard_name'] = 'volume_extinction_coefficient_in_air_due_to_ambient_aerosol_particles'
    NETCDF_VAR_ATTRIBUTES['ec355aer']['units'] = '1/Mm'

    NETCDF_VAR_ATTRIBUTES['nitrogendioxide_tropospheric_column'] = {}
    NETCDF_VAR_ATTRIBUTES['nitrogendioxide_tropospheric_column']['_FillValue'] = np.nan
    NETCDF_VAR_ATTRIBUTES['nitrogendioxide_tropospheric_column']['long_name'] = \
        'Tropospheric vertical column of nitrogen dioxide'
    NETCDF_VAR_ATTRIBUTES['nitrogendioxide_tropospheric_column'][
        'standard_name'] = 'troposphere_mole_content_of_nitrogen_dioxide'
    NETCDF_VAR_ATTRIBUTES['nitrogendioxide_tropospheric_column']['units'] = 'mol m-2'
    NETCDF_VAR_ATTRIBUTES['nitrogendioxide_tropospheric_column']['coordinates'] = 'longitude latitude'

    NETCDF_VAR_ATTRIBUTES['nitrogendioxide_tropospheric_column_mean'] = \
        NETCDF_VAR_ATTRIBUTES['nitrogendioxide_tropospheric_column']

    NETCDF_VAR_ATTRIBUTES['nitrogendioxide_tropospheric_column_numobs'] = {}
    NETCDF_VAR_ATTRIBUTES['nitrogendioxide_tropospheric_column_numobs']['_FillValue'] = np.nan
    NETCDF_VAR_ATTRIBUTES['nitrogendioxide_tropospheric_column_numobs']['long_name'] = \
        'number of observations'
    # NETCDF_VAR_ATTRIBUTES['nitrogendioxide_tropospheric_column_numobs'][
    #     'standard_name'] = 'troposphere_mole_content_of_nitrogen_dioxide'
    NETCDF_VAR_ATTRIBUTES['nitrogendioxide_tropospheric_column_numobs']['units'] = '1'
    NETCDF_VAR_ATTRIBUTES['nitrogendioxide_tropospheric_column_numobs']['coordinates'] = 'longitude latitude'

    NETCDF_VAR_ATTRIBUTES['ozone_tropospheric_column'] = {}
    NETCDF_VAR_ATTRIBUTES['ozone_tropospheric_column']['_FillValue'] = np.nan
    NETCDF_VAR_ATTRIBUTES['ozone_tropospheric_column']['long_name'] = \
        'average tropospheric ozone column based on CCD algorithm'
    NETCDF_VAR_ATTRIBUTES['ozone_tropospheric_column'][
        'standard_name'] = 'troposphere_mole_content_of_ozone'
    NETCDF_VAR_ATTRIBUTES['ozone_tropospheric_column']['units'] = 'mol m-2'
    NETCDF_VAR_ATTRIBUTES['ozone_tropospheric_column']['coordinates'] = 'longitude latitude'

    NETCDF_VAR_ATTRIBUTES['qa_index'] = {}
    NETCDF_VAR_ATTRIBUTES['qa_index']['_FillValue'] = np.nan
    NETCDF_VAR_ATTRIBUTES['qa_index']['long_name'] = 'data quality value'
    NETCDF_VAR_ATTRIBUTES['qa_index']['comment'] = \
        'A continuous quality descriptor, varying between 0(no data) and 1 (full quality data). Recommend to ignore data with qa_value < 0.5'
    NETCDF_VAR_ATTRIBUTES['qa_index']['units'] = '1'
    NETCDF_VAR_ATTRIBUTES['qa_index']['coordinates'] = 'longitude latitude'

    TEX_UNITS = {}
    TEX_UNITS['ec355aer'] = r'$10^{-6} \cdot m^{-1}$'
    TEX_UNITS['bs355aer'] = ''

    CODA_READ_PARAMETERS = {}
    CODA_READ_PARAMETERS['s5p'] = {}
    CODA_READ_PARAMETERS['s5p']['metadata'] = {}
    CODA_READ_PARAMETERS['s5p']['vars'] = {}

    CODA_READ_PARAMETERS['s5p']['metadata'][_TIME_NAME] = 'PRODUCT/time_utc'
    CODA_READ_PARAMETERS['s5p']['metadata'][_LATITUDENAME] = 'PRODUCT/latitude'
    CODA_READ_PARAMETERS['s5p']['metadata'][_LONGITUDENAME] = 'PRODUCT/longitude'
    CODA_READ_PARAMETERS['s5p']['metadata'][_LONBOUNDSNAME] = 'PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds'
    CODA_READ_PARAMETERS['s5p']['metadata'][_LATBOUNDSNAME] = 'PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds'
    CODA_READ_PARAMETERS['s5p']['metadata'][_QANAME] = 'PRODUCT/qa_value'
    CODA_READ_PARAMETERS['s5p']['vars'][_NO2NAME] = 'PRODUCT/nitrogendioxide_tropospheric_column'
    CODA_READ_PARAMETERS['s5p']['vars'][_O3NAME] = 'PRODUCT/ozone_tropospheric_vertical_column'

    SUPPORTED_DATASETS = []
    SUPPORTED_DATASETS.append('s5p')
    SUPPORTED_DATASETS.append('aeolus')

    DATASET_READ = ''

    def __init__(self, index_pointer=0, loglevel=logging.INFO, verbose=False):
        self.verbose = verbose
        self.metadata = {}
        self.data = []
        self.data_for_gridding = {}
        self.gridded_data = {}
        self.global_attributes = {}
        self.index = len(self.metadata)
        self.files = []
        self.files_read = []
        self.index_pointer = index_pointer
        # that's the flag to indicate if the location of a data point in self.data has been
        # stored in rads in self.data already
        # trades RAM for speed
        self.rads_in_array_flag = False

        if loglevel is not None:
            self.logger = logging.getLogger(__name__)
            if self.logger.hasHandlers():
                # Logger is already configured, remove all handlers
                self.logger.handlers = []
            # self.logger = logging.getLogger('pyaerocom')
            default_formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(default_formatter)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(loglevel)
            self.logger.debug('init')

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.metadata[float(self.index)]

    def __str__(self):
        stat_names = []
        for key in self.metadata:
            stat_names.append(self.metadata[key]['station name'])

        return ','.join(stat_names)

    ###################################################################################

    def ndarr2data(self, file_data):
        """small helper routine to put the data read by the read_file method into
        the ndarray of self.data"""

        # start_read = time.perf_counter()
        # return all data points
        num_points = len(file_data)
        if self.index_pointer == 0:
            self.data = file_data
            self._ROWNO = num_points
            self.index_pointer = num_points

        else:
            # append to self.data
            # add another array chunk to self.data
            self.data = np.append(self.data, np.zeros([num_points, self._COLNO], dtype=np.float_),
                                  axis=0)
            self._ROWNO = num_points
            # copy the data
            self.data[self.index_pointer:, :] = file_data
            self.index_pointer = self.index_pointer + num_points

            # end_time = time.perf_counter()
            # elapsed_sec = end_time - start_read
            # temp = 'time for single file read seconds: {:.3f}'.format(elapsed_sec)
            # self.logger.warning(temp)

    ###################################################################################
    def read_file(self, filename, vars_to_read=None, read_dataset='s5p', return_as='dict',
                  loglevel=None, apply_quality_flag=True):
        """method to read the file partially

        Parameters
        ----------
        filename : str
            absolute path to filename to read
        vars_to_read : list
            list of str with variable names to read; defaults to ['od355aer']
        verbose : Bool
            set to True to increase verbosity

        Returns
        --------
        Either:
            dictionary (default):
                keys are 'time', 'latitude', 'longitude', 'altitude' and the variable names
                'ec355aer', 'bs355aer', 'sr', 'lod' if the whole file is read
                'time' is a 1d array, while the other dict values are a another dict with the
                time as keys (the same ret['time']) and a numpy array as values. These values represent the profile.
                Note 1: latitude and longitude are height dependent due to the tilt of the measurement.
                Note 2: negative values indicate a NaN

            2d ndarray of type float:
                representing a 'point cloud' with all points
                    column 1: time in seconds since the Unix epoch with ms accuracy (same time for every height
                    in a profile)
                    column 2: latitude
                    column 3: longitude
                    column 4: altitude

                    Note: negative values are put to np.nan already

                    The indexes are noted in read_aeolus_l2a_data.ReadAeolusL2aData.<index_name>
                    e.g. the time index is named read_aeolus_l2a_data.ReadAeolusL2aData._TIMEINDEX
                    have a look at the example to access the values


        Example
        -------
        >>> import simplegridder
        >>> obj = simplegridder.ReadL2Data(verbose=True)
        >>> import os
        >>> filename = '/lustre/storeB/project/fou/kl/vals5p/tmp/S5P_OFFL_L2__NO2____20181201T011851_20181201T030021_05869_01_010202_20181207T030115.nc'


        >>> # read returning a ndarray
        >>> filedata_numpy = obj.read_file(filename, vars_to_read=['ec355aer'], return_as='numpy')
        >>> time_as_numpy_datetime64 = filedata_numpy[0,obj._TIMEINDEX].astype('datetime64[s]')
        >>> print('time: {}'.format(time_as_numpy_datetime64))
        >>> print('latitude: {}'.format(filedata_numpy[1,obj._LATINDEX]))
        >>> # read returning a dictionary
        >>> filedata = obj.read_file(filename, vars_to_read=['ec355aer'])
        >>> print('time: {}'.format(filedata['time'][0].astype('datetime64[s]')))
        >>> print('all latitudes of 1st time step: {}'.format(filedata['latitude'][filedata['time'][0]]))
        """

        import time
        import coda

        # coda uses 2000-01-01T00:00:00 as epoch unfortunately.
        # so calculate the difference in seconds to the Unix epoch
        seconds_to_add = np.datetime64('2000-01-01T00:00:00') - np.datetime64('1970-01-01T00:00:00')
        seconds_to_add = seconds_to_add.astype(np.float_)

        # the same can be achieved using pandas, but we stick to numpy here
        # base_time = pd.DatetimeIndex(['2000-01-01'])
        # seconds_to_add = (base_time.view('int64') // pd.Timedelta(1, unit='s'))[0]

        start = time.perf_counter()
        file_data = {}

        self.logger.info('reading file {}'.format(filename))
        # read file
        product = coda.open(filename)

        # if isinstance(read_dataset, str):
        #     read_dataset = [read_dataset]
        for retrieval in self.SUPPORTED_DATASETS:
            if retrieval not in read_dataset:
                continue

            vars_to_read_in = None
            vars_to_read_in = vars_to_read.copy()
            if vars_to_read is None:
                # read all variables
                vars_to_read_in = list(self.CODA_READ_PARAMETERS[retrieval]['vars'].keys())
            vars_to_read_in.extend(list(self.CODA_READ_PARAMETERS[retrieval]['metadata'].keys()))
            # get rid of duplicates
            vars_to_read_in = list(set(vars_to_read_in))

            # read data time
            # do that differently since its only stored once per profile
            coda_groups_to_read = (
                self.CODA_READ_PARAMETERS[retrieval]['metadata'][self._TIME_NAME].split(self.GROUP_DELIMITER))

            # this works only for aeolus
            if read_dataset == 'aeolus':
                # this is for ESA Aeolus DBL files
                file_data[self._TIME_NAME] = coda.fetch(product,
                                                        coda_groups_to_read[0],
                                                        -1,
                                                        coda_groups_to_read[1])
                # epoch is 1 January 2000 at ESA
                # so add offset to move that to 1 January 1970
                # and save it into a np.datetime64[ms] object

                file_data[self._TIME_NAME] = \
                    ((file_data[self._TIME_NAME] + seconds_to_add) * 1.E3).astype(np.int).astype('datetime64[ms]')

                # read data in a simple dictionary
                for var in vars_to_read_in:
                    # time has been read already
                    if var == self._TIME_NAME:
                        continue
                    self.logger.info('reading var: {}'.format(var))
                    try:
                        groups = self.CODA_READ_PARAMETERS[retrieval]['vars'][var].split(self.GROUP_DELIMITER)
                    except KeyError:
                        groups = self.CODA_READ_PARAMETERS[retrieval]['metadata'][var].split(self.GROUP_DELIMITER)

                    if len(groups) == 3:
                        file_data[var] = {}
                        for idx, key in enumerate(file_data[self._TIME_NAME]):
                            file_data[var][key] = coda.fetch(product,
                                                             groups[0],
                                                             idx,
                                                             groups[1],
                                                             -1,
                                                             groups[2])

                    elif len(groups) == 2:
                        for idx, key in enumerate(file_data[self._TIME_NAME]):
                            file_data[var][key] = coda.fetch(product,
                                                             groups[0],
                                                             -1,
                                                             groups[1])
                    else:
                        for idx, key in enumerate(file_data[self._TIME_NAME]):
                            file_data[var][key] = coda.fetch(product,
                                                             groups[0])

            elif read_dataset == 's5p':
                # This is for Sentinel 5p TEMIS netcdf files

                # in this case the time comes as a string!
                time_data_temp = coda.fetch(product,
                                            coda_groups_to_read[0],
                                            coda_groups_to_read[1])
                # time_data_temp comes as shape[1,<time step number] and looks like this
                # 2018-12-01T01:40:26.055000Z
                # since np.datetime64 will remove the understanding of the last Z, we remove that
                # also in the list comprehension doing the conversion to np.datetime64

                file_data[self._TIME_NAME] = \
                    np.array([np.datetime64(time_data_temp[0, x][:-1]) for x in range(len(time_data_temp[0]))])

                # read data in a simple dictionary
                for var in vars_to_read_in:
                    # time has been read already
                    if var == self._TIME_NAME:
                        continue
                    self.logger.info('reading var: {}'.format(var))
                    try:
                        groups = self.CODA_READ_PARAMETERS[retrieval]['vars'][var].split(self.GROUP_DELIMITER)
                    except KeyError:
                        groups = self.CODA_READ_PARAMETERS[retrieval]['metadata'][var].split(self.GROUP_DELIMITER)

                    # the data comes as record and not as array as at aeolus
                    file_data[var] = {}

                    if len(groups) == 3:
                        file_data[var] = np.squeeze(coda.fetch(product,
                                                               groups[0],
                                                               groups[1],
                                                               groups[2]))

                    elif len(groups) == 2:
                        file_data[var] = np.squeeze(coda.fetch(product,
                                                               groups[0],
                                                               groups[1]))
                    elif len(groups) == 4:

                        file_data[var] = np.squeeze(coda.fetch(product,
                                                               groups[0],
                                                               groups[1],
                                                               groups[2],
                                                               groups[3]))
                    else:
                        file_data[var] = np.squeeze(coda.fetch(product,
                                                               groups[0]))

            if return_as == 'numpy':
                # return as one multidimensional numpy array that can be put into self.data directly
                # (column wise because the column numbers do not match)
                index_pointer = 0
                data = np.empty([self._ROWNO, self._COLNO], dtype=np.float_)
                if read_dataset == 'aeolus':

                    for idx, _time in enumerate(file_data['time'].astype(np.float_) / 1000.):
                        # file_data['time'].astype(np.float_) is milliseconds after the (Unix) epoch
                        # but we want to save the time as seconds since the epoch
                        for _index in range(len(file_data['latitude'][file_data['time'][idx]])):
                            # this works because all variables have to have the same size
                            # (aka same number of height levels)
                            # This loop could be avoided using numpy index slicing
                            # do that in case we need more optimisations
                            data[index_pointer, self._TIMEINDEX] = _time
                            for var in vars_to_read_in:
                                # time is the index, so skip it here
                                if var == self._TIME_NAME:
                                    continue
                                # logitudes are 0 based for Aeolus, but -18- based for model data
                                # adjust Aeolus to model data
                                if var == self._LONGITUDENAME:
                                    data[index_pointer, self.INDEX_DICT[var]] = \
                                        file_data[var][file_data['time'][idx]][_index]
                                    if file_data[var][file_data['time'][idx]][_index] > 180.:
                                        data[index_pointer, self.INDEX_DICT[var]] = \
                                            file_data[var][file_data['time'][idx]][_index] - 360.

                                else:
                                    data[index_pointer, self.INDEX_DICT[var]] = \
                                        file_data[var][file_data['time'][idx]][_index]
                                # put negative values to np.nan if the variable is not a metadata variable
                                if data[index_pointer, self.INDEX_DICT[var]] == self.NAN_DICT[var]:
                                    data[index_pointer, self.INDEX_DICT[var]] = np.nan

                            index_pointer += 1
                            if index_pointer >= self._ROWNO:
                                # add another array chunk to self.data
                                data = np.append(data, np.empty([self._CHUNKSIZE, self._COLNO], dtype=np.float_),
                                                 axis=0)
                                self._ROWNO += self._CHUNKSIZE

                elif read_dataset == 's5p':
                    # as this point file_data['time'] has ns time resolution
                    # degrade that to the pyaerocom internal ms
                    for idx, _time in enumerate(file_data['time'].astype('datetime64[ms]').astype(np.float_) / 1000.):
                        # file_data['time'].astype(np.float_) is milliseconds after the (Unix) epoch
                        # but we want to save the time as seconds since the epoch

                        # loop over the number of ground pixels
                        for _index in range(file_data[self._LATITUDENAME].shape[1]):
                            # check first if the quality flag requirement is met

                            if apply_quality_flag and file_data[self._QANAME][idx, _index] < self.QUALITY_FLAGS[
                                vars_to_read[0]]:
                                # if _index < 100:
                                #     print(file_data[self._QANAME][idx,_index])
                                continue

                            data[index_pointer, self._TIMEINDEX] = _time
                            # loop over the variables
                            for var in vars_to_read_in:
                                # time is the index, so skip it here
                                if var == self._TIME_NAME:
                                    continue
                                # the bounds need to be treated special
                                elif var in self.SIZE_DICT:
                                    # elif var == self._LONBOUNDSNAME or var == self._LATBOUNDSNAME:
                                    #     print('bla')
                                    data[index_pointer,
                                    self.INDEX_DICT[var]:self.INDEX_DICT[var] + self.SIZE_DICT[var]] = \
                                        file_data[var][idx, _index]

                                else:
                                    data[index_pointer, self.INDEX_DICT[var]] = file_data[var][idx, _index]

                            index_pointer += 1
                            if index_pointer >= self._ROWNO:
                                # add another array chunk to self.data
                                data = np.append(data, np.empty([self._CHUNKSIZE, self._COLNO], dtype=np.float_),
                                                 axis=0)
                                self._ROWNO += self._CHUNKSIZE

                # return only the needed elements...
                file_data = data[0:index_pointer]

        coda.close(product)
        end_time = time.perf_counter()
        elapsed_sec = end_time - start
        temp = 'time for single file read [s]: {:.3f}'.format(elapsed_sec)
        self.logger.info(temp)
        # self.logger.info('{} points read'.format(index_pointer))
        self.DATASET_READ = read_dataset
        self.files_read.append(filename)
        return file_data

    ###################################################################################

    def read(self, base_dir=None, vars_to_read=['ec355aer'], locs=None, backend='geopy', verbose=False):
        """method to read all files in self.files into self.data and self.metadata
        At this point the data format is NOT the same as for the ungridded base class


        Example
        -------
        >>> import logging
        >>> import read_aeolus_l2a_data
        >>> obj = read_aeolus_l2a_data.ReadAeolusL2aData(loglevel=logging.DEBUG)
        >>> obj.read(vars_to_read=['ec355aer'])
        >>> locations = [(49.093,8.428,0.),(58.388, 8.252, 0.)]
        >>> obj.read(locs=locations,vars_to_read=['ec355aer'],verbose=True)
        >>> obj.read(verbose=True)
        """

        import time

        start = time.perf_counter()
        self.files = self.get_file_list()
        after_file_search_time = time.perf_counter()
        elapsed_sec = after_file_search_time - start
        temp = 'time for file find: {:.3f}'.format(elapsed_sec)
        self.logger.info(temp)

        for idx, _file in enumerate(sorted(self.files)):
            file_data = self.read_file(_file, vars_to_read=vars_to_read, return_as='numpy')
            # the metadata dict is left empty for L2 data
            # the location in the data set is time step dependant!
            self.ndarr2data(file_data)

        end_time = time.perf_counter()
        elapsed_sec = end_time - start
        temp = 'overall time for file read [s]: {:.3f}'.format(elapsed_sec)
        self.logger.info(temp)
        self.logger.info('size of data object: {}'.format(self.index_pointer))

    ###################################################################################

    ###################################################################################

    def select_bbox(self, bbox=None):
        """method to return all points of self.data laying within a certain latitude and longitude range

        This method will likely never be used by a user, but serves as helper method for the colocate method

        EXAMPLE
        =======
        >>> import logging
        >>> import read_aeolus_l2a_data
        >>> obj = read_aeolus_l2a_data.ReadAeolusL2aData(loglevel=logging.DEBUG)
        >>> obj.read(vars_to_read=['ec355aer'])
        >>> bbox = (-62.,-61.,7.,8.)
        >>> result = obj.select_bbox(bbox)
        >>> import numpy as np
        >>> print('min distance: {:.3f} km'.format(np.nanmin(obj.data[:, obj._DISTINDEX])))
        >>> print('max distance: {:.3f} km'.format(np.nanmax(obj.data[:, obj._DISTINDEX])))


        """
        start = time.perf_counter()

        # ret_data = np.empty([self._ROWNO, self._COLNO], dtype=np.float_)
        # index_counter = 0
        # cut_flag = True

        if bbox is not None:
            logging.info(bbox)
            lat_min = bbox[0]
            lat_max = bbox[1]
            lon_min = bbox[2]
            lon_max = bbox[3]

            # remove NaNs at this point
            matching_indexes = np.where(np.isfinite(self.data[:, self._LATINDEX]))
            ret_data = self.data[matching_indexes[0], :]

            # np.where can unfortunately only work with a single criterion
            matching_indexes = np.where(ret_data[:, self._LATINDEX] <= lat_max)
            ret_data = ret_data[matching_indexes[0], :]
            # logging.warning('len after lat_max: {}'.format(len(ret_data)))
            matching_indexes = np.where(ret_data[:, self._LATINDEX] >= lat_min)
            ret_data = ret_data[matching_indexes[0], :]
            # logging.warning('len after lat_min: {}'.format(len(ret_data)))
            matching_indexes = np.where(ret_data[:, self._LONINDEX] <= lon_max)
            ret_data = ret_data[matching_indexes[0], :]
            # logging.warning('len after lon_max: {}'.format(len(ret_data)))
            matching_indexes = np.where(ret_data[:, self._LONINDEX] >= lon_min)
            ret_data = ret_data[matching_indexes[0], :]
            # logging.warning('len after lon_min: {}'.format(len(ret_data)))
            # matching_length = len(matching_indexes[0])

            # end_time = time.perf_counter()
            # elapsed_sec = end_time - start
            # temp = 'time for single station bbox calc [s]: {:.3f}'.format(elapsed_sec)
            # self.logger.info(temp)
            # log the found times
            # unique_times = np.unique(self.data[matching_indexes,self._TIMEINDEX]).astype('datetime64[s]')
            # self.logger.info('matching times:')
            # self.logger.info(unique_times)
            # if len(ret_data) == 0:
            #     data_lat_min = np.nanmin(self.data[:,self._LATINDEX])
            #     data_lat_max = np.nanmax(self.data[:,self._LATINDEX])
            #     data_lon_min = np.nanmin(self.data[:,self._LONINDEX])
            #     data_lon_max = np.nanmax(self.data[:,self._LONINDEX])
            #     logging.info('[lat_min, lat_max, lon_min, lon_max in data]: '.format([data_lat_min, data_lat_max, data_lon_min, data_lon_max]))
            return ret_data

    ###################################################################################
    def get_file_list(self, basedir=None):
        """search for files to read

        Example
        -------
        >>> import read_aeolus_l2a_data
        >>> obj = read_aeolus_l2a_data.ReadAeolusL2aData(verbose=True)
        >>> files = obj.get_file_list()
        """

        self.logger.info('searching for data files. This might take a while...')
        if basedir is None:
            files = glob.glob(os.path.join(self.DATASET_PATH, '**',
                                           self._FILEMASK),
                              recursive=True)
        else:
            files = glob.glob(os.path.join(basedir, '**',
                                           self._FILEMASK),
                              recursive=True)

        return files

    ##################################################################################################

    def to_netcdf_simple(self, netcdf_filename='/home/jang/tmp/to_netcdf_simple.nc',
                         global_attributes=None, vars_to_read=None,
                         data_to_write=None, gridded=False):

        """method to store the file contents in a very basic netcdf file
        >>> import read_aeolus_l2a_data
        >>> obj = read_aeolus_l2a_data.ReadAeolusL2aData(verbose=True)
        >>> import os
        >>> os.environ['CODA_DEFINITION']='/lustre/storeA/project/aerocom/aerocom1/ADM_CALIPSO_TEST/'
        >>> filename = '/lustre/storeB/project/fou/kl/admaeolus/data.rev.2A02/AE_OPER_ALD_U_N_2A_20181201T033526026_005423993_001590_0001.DBL'
        >>> # read returning a ndarray
        >>> filedata_numpy = obj.read_file(filename, vars_to_read=['ec355aer'], return_as='numpy')
        >>> obj.ndarr2data(filedata_numpy)
        >>> obj.to_netcdf_simple()

        Parameters:
        ----------
            global_attributes : dict
            dictionary with things to put into the global attributes of a netcdf file

        """

        start_time = time.perf_counter()
        import xarray as xr
        import pandas as pd
        import numpy as np

        vars_to_read_in = vars_to_read.copy()
        if isinstance(vars_to_read_in, str):
            vars_to_read_in = [vars_to_read_in]

        if not gridded:

            if data_to_write is None:
                _data = self.data
            else:
                _data = data_to_write
    
            vars_to_read_in.extend(list(self.CODA_READ_PARAMETERS[self.DATASET_READ]['metadata'].keys()))
    
            datetimedata = pd.to_datetime(_data[:, self._TIMEINDEX].astype('datetime64[s]'))
            # pointnumber = np.arange(0, len(datetimedata))
            bounds_dim_name = 'bounds'
            point_dim_name = 'point'
            ds = xr.Dataset()
    
            # time is a special variable that needs special treatment
            ds['time'] = (point_dim_name), datetimedata
            for var in vars_to_read_in:
                if var == self._TIME_NAME:
                    continue
                # 1D data
                if var not in self.SIZE_DICT:
                    ds[var] = (point_dim_name), _data[:, self.INDEX_DICT[var]]
                else:
                    # 2D data: here: bounds
                    ds[var] = ((point_dim_name, bounds_dim_name),
                               _data[:, self.INDEX_DICT[var]:self.INDEX_DICT[var] + self.SIZE_DICT[var]])
    
                # remove _FillVar attribute for coordinate variables as CF requires it
                if var in self.COORDINATE_NAMES:
                    ds[var].encoding['_FillValue'] = None
    
                # add predifined attributes
                try:
                    for attrib in self.NETCDF_VAR_ATTRIBUTES[var]:
                        ds[var].attrs[attrib] = self.NETCDF_VAR_ATTRIBUTES[var][attrib]
    
                except KeyError:
                    pass
    
        else:
            # write gridded data to netcdf
            if data_to_write is None:
                _data = self.gridded_data
            else:
                _data = data_to_write

            bounds_dim_name = 'bounds'
            time_dim_name = 'time'
            lat_dim_name = 'latitude'
            lon_dim_name = 'longitude'

            ds = xr.Dataset()

            # coordinate variables need special treatment

            ds[time_dim_name] = (time_dim_name), [np.datetime64(_data[time_dim_name], 'D')]
            ds[lat_dim_name] = (lat_dim_name), _data[lat_dim_name],
            ds[lon_dim_name] = (lon_dim_name), _data[lon_dim_name]

            for var in vars_to_read_in:
                if var == self._TIME_NAME:
                    continue
                # 1D data
                # 3D data
                ds[var+'_mean'] = (time_dim_name, lat_dim_name, lon_dim_name), np.reshape(_data[var]['mean'],(len(ds[time_dim_name]),len(_data[lat_dim_name]), len(_data[lon_dim_name])))
                ds[var+'_numobs'] = (time_dim_name, lat_dim_name, lon_dim_name), np.reshape(_data[var]['numobs'],(len(ds[time_dim_name]),len(_data[lat_dim_name]), len(_data[lon_dim_name])))

                # remove _FillVar attribute for coordinate variables as CF requires it

            vars_to_read_in.extend([time_dim_name, lat_dim_name, lon_dim_name])

            for var in ds:
                # add predifined attributes
                try:
                    for attrib in self.NETCDF_VAR_ATTRIBUTES[var]:
                        ds[var].attrs[attrib] = self.NETCDF_VAR_ATTRIBUTES[var][attrib]

                except KeyError:
                    pass

            for var in ds.coords:
                if var in self.COORDINATE_NAMES:
                    ds[var].encoding['_FillValue'] = None

                # add predifined attributes
                try:
                    for attrib in self.NETCDF_VAR_ATTRIBUTES[var]:
                        ds[var].attrs[attrib] = self.NETCDF_VAR_ATTRIBUTES[var][attrib]

                except KeyError:
                    pass

    # add potential global attributes
        try:
            for name in global_attributes:
                ds.attrs[name] = global_attributes[name]
        except:
            pass

        ds.to_netcdf(netcdf_filename)

        end_time = time.perf_counter()
        elapsed_sec = end_time - start_time
        temp = 'time for netcdf write [s]: {:.3f}'.format(elapsed_sec)
        self.logger.info(temp)
        temp = 'file written: {}'.format(netcdf_filename)
        self.logger.info(temp)

    ###################################################################################
    def to_grid(self, _data=None, vars=None, gridtype='1x1', engine='python'):
        """simple gridding algorithm that only takes the pixel middle points into account

        All the data points in self.data or _data are considered!



        """
        import numpy as np

        vars_to_read_in = vars.copy()
        if isinstance(vars_to_read_in, str):
            vars_to_read_in = [vars_to_read_in]

        if _data is None:
            _data = self.data

        if engine == 'python':
            start_time = time.perf_counter()
            grid_data_prot = {}
            # define ouput grid
            if gridtype == '1x1':
                # global 1x1 degree grid
                pass
                temp='starting simple gridding for 1x1 degree grid...'
                self.logger.info(temp)
                max_grid_dist_lon = 1.
                max_grid_dist_lat = 1.
                grid_lats = np.arange(-89.5, 90.5, max_grid_dist_lat)
                grid_lons = np.arange(-179.5, 180.5, max_grid_dist_lon)

                grid_array_prot = np.full((grid_lats.size, grid_lons.size), np.nan)
                # organise the data in a nested python dict like dict_data[grid_lat][grid_lon]=np.ndarray
                for grid_lat in grid_lats:
                    grid_data_prot[grid_lat] = {}
                    for grid_lon in grid_lons:
                        grid_data_prot[grid_lat] = {}

                end_time = time.perf_counter()
                elapsed_sec = end_time - start_time
                temp = 'time for global 1x1 gridding with python data types [s] init: {:.3f}'.format(elapsed_sec)
                self.logger.info(temp)

                #predifine the output data dict
                data_for_gridding = {}
                gridded_var_data = {}
                for var in vars:
                    data_for_gridding[var] = grid_data_prot.copy()
                    gridded_var_data['latitude']=grid_lats
                    gridded_var_data['longitude']=grid_lons
                    gridded_var_data['time']=np.mean(_data[:,self._TIMEINDEX]).astype('datetime64[s]')

                    gridded_var_data[var] = {}
                    gridded_var_data[var]['mean'] = grid_array_prot.copy()
                    gridded_var_data[var]['stddev'] = grid_array_prot.copy()
                    gridded_var_data[var]['numobs'] = grid_array_prot.copy()

                # Loop through the output grid and collect data
                # store that in data_for_gridding[var]
                for lat_idx, grid_lat in enumerate(grid_lats):
                    diff_lat = np.absolute((_data[:, self._LATINDEX] - grid_lat))
                    lat_match_indexes = np.squeeze(np.where(diff_lat <= max_grid_dist_lat))
                    if lat_match_indexes.size == 0:
                        continue

                    for lon_idx, grid_lon in enumerate(grid_lons):
                        diff_lon = np.absolute((_data[lat_match_indexes,self._LONINDEX] - grid_lon))
                        lon_match_indexes = np.squeeze(np.where(diff_lon <= max_grid_dist_lon))
                        if lon_match_indexes.size == 0:
                            continue

                        for var in vars:
                            data_for_gridding[var][grid_lat][grid_lon] = \
                                np.array(_data[lat_match_indexes[lon_match_indexes],self.INDEX_DICT[var]])
                            gridded_var_data[var]['mean'][lat_idx,lon_idx] = \
                                np.nanmean(_data[lat_match_indexes[lon_match_indexes],self.INDEX_DICT[var]])
                            gridded_var_data[var]['stddev'][lat_idx,lon_idx] = \
                                np.nanstd(_data[lat_match_indexes[lon_match_indexes],self.INDEX_DICT[var]])
                            gridded_var_data[var]['numobs'][lat_idx,lon_idx] = \
                                _data[lat_match_indexes[lon_match_indexes],self.INDEX_DICT[var]].size



                # now go through self.data and select the appropriate data points


                self.data_for_gridding = data_for_gridding
                self.gridded_data = gridded_var_data
                end_time = time.perf_counter()
                elapsed_sec = end_time - start_time
                temp = 'time for global 1x1 gridding with python data types [s]: {:.3f}'.format(elapsed_sec)
                self.logger.info(temp)
                return 1

            elif gridtype == '1x1_emep':
                # 1 by on degree grid on emep domain
                pass

            pass

        pass
    ###################################################################################

    def plot_map(self, plotfilename, bbox=None, himalaya_flag=None, title=None,
                 vars=['nitrogendioxide_tropospheric_column']):
        """small routine to plot self.gridded_data

        >>> import matplotlib.pyplot as plt
        >>> from mpl_toolkits.basemap import Basemap
        >>> lats=obj.data[:,obj._LATINDEX]
        >>> lons=obj.data[:,obj._LONINDEX]
        >>> m = Basemap(projection='merc',lon_0=0)
        >>> x, y = m(lons,lats)
        >>> m.drawmapboundary(fill_color='#99ffff')
        >>> m.fillcontinents(color='#cc9966',lake_color='#99ffff')
        >>> m.scatter(x,y,3,marker='o',color='k')
        >>> plt.show()
        """

        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap
        import numpy as np

        # positions of some peaks:

        # Everest: 27.988056, 86.925278
        # K2: 35.8825, 76.513333
        # Kangchenjunga: 27.7025, 88.146667
        # Lhotse: 27.961667, 86.933333
        # Makalu: 27.889167, 87.088611
        # Cho Oyu: 28.094167, 86.660833
        # Dhaulagiri: 28.698333, 83.4875
        # Manaslu: 28.549444, 84.561944
        # Nanga Parbat: 35.2375, 74.589167
        # Annapurna Massif: 28.596111, 83.820278
        # Gasherbrum I: 35.724444, 76.696389
        # Broad Peak: 35.811667, 76.565
        # Gasherbrum II:35.758333, 76.653333
        # Shishapangma:28.352222, 85.779722
        himalaya_data = {}
        himalaya_data['Everest'] = (27.988056, 86.925278)
        himalaya_data['K2'] = (35.8825, 76.513333)
        himalaya_data['Kangchenjunga'] = (27.7025, 88.146667)
        himalaya_data['Lhotse'] = (27.961667, 86.933333)
        himalaya_data['Makalu'] = (27.889167, 87.088611)
        himalaya_data['Cho Oyu'] = (28.094167, 86.660833)
        himalaya_data['Dhaulagiri'] = (28.698333, 83.4875)
        himalaya_data['Manaslu'] = (28.549444, 84.561944)
        himalaya_data['Nanga Parbat'] = (35.2375, 74.589167)
        himalaya_data['Annapurna Massif'] = (28.596111, 83.820278)
        himalaya_data['Gasherbrum I'] = (35.724444, 76.696389)
        himalaya_data['Broad Peak'] = (35.811667, 76.565)
        himalaya_data['Gasherbrum II'] = (35.758333, 76.653333)
        himalaya_data['Shishapangma'] = (28.352222, 85.779722)
        if bbox:
            pass

            lat_low = bbox[0]
            lat_high = bbox[1]
            lon_low = bbox[2]
            lon_high = bbox[3]
        else:
            lat_low = -90.
            lat_high = 90.
            lon_low = -180.
            lon_high = 180.

        for var in vars:
            lats = self.gridded_data['latitude']
            lons = self.gridded_data['longitude']



            m = Basemap(projection='cyl', llcrnrlat=lat_low, urcrnrlat=lat_high,
                        llcrnrlon=lon_low, urcrnrlon=lon_high, resolution='c', fix_aspect=False)

            x, y = m(lons, lats)
            # m.drawmapboundary(fill_color='#99ffff')
            # m.fillcontinents(color='#cc9966', lake_color='#99ffff')
            # plot = m.scatter(x, y, 4, marker='o', color='r', )
            # plot = iplt.pcolormesh(time_sub_cube[:, :], cmap = colormap, vmin=0., vmax=max(colorbar_levels))

            my_cmap = plt.get_cmap('rainbow')
            my_cmap.set_under('white')
            cs = m.pcolormesh(x, y, self.gridded_data[var]['mean'], cmap=my_cmap)
            m.colorbar(cs, extend='min')

            m.drawmeridians(np.arange(-180, 220, 40), labels=[0, 0, 0, 1], fontsize=10)
            m.drawparallels(np.arange(-90, 120, 30), labels=[1, 1, 0, 0], fontsize=10)
            # axis = plt.axis([LatsToPlot.min(), LatsToPlot.max(), LonsToPlot.min(), LonsToPlot.max()])
            # ax = plot.axes
            m.drawcoastlines()
            # m.etopo()
            # m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 1500, verbose= True)
            # m.arcgisimage(service='World_Shaded_Relief', xpixels = 1500, verbose= True)
            if bbox is not None:
                m.drawcountries()
                # m.drawrivers()

            if himalaya_flag:
                for peak in himalaya_data:
                    x,y=m(himalaya_data[peak][1], himalaya_data[peak][0])
                    plot=m.plot(x, y, 4, marker='.', color='b')

            if title:
                plt.title(title,fontsize='small')


            plt.savefig(plotfilename, dpi=300)
            plt.close()


    ###################################################################################


    if __name__ == '__main__':
        import logging

        import argparse
        options = {}
        default_topo_file = '/lustre/storeB/project/fou/kl/admaeolus/EMEP.topo/MACC14_topo_v1.nc'
        default_gridded_out_file = './gridded.nc'

        parser = argparse.ArgumentParser(
            description='command line interface to simplegridder.py\n\n\n')
        parser.add_argument("--file",
                            help="file(s) to read", nargs="+")
        parser.add_argument("-v", "--verbose", help="switch on verbosity",
                            action='store_true')
        parser.add_argument("--listpaths", help="list the file contents.", action='store_true')
        parser.add_argument("--readpaths", help="read listed rootpaths of coda supported file. Can be comma separated",
                            default='mph,sca_optical_properties')
        parser.add_argument("-o", "--outfile", help="output file")
        parser.add_argument("--outdir", help="output directory; the filename will be extended with the string '.nc'")
        parser.add_argument("--plotdir", help="directories where the plots will be put; defaults to './'",
                            default='./')
        parser.add_argument("--logfile", help="logfile; defaults to /home/jang/tmp/aeolus2netcdf.log",
                            default="/home/jang/tmp/aeolus2netcdf.log")
        parser.add_argument("-O", "--overwrite", help="overwrite output file", action='store_true')
        parser.add_argument("--emep", help="flag to limit the read data to the cal/val model domain",
                            action='store_true')
        parser.add_argument("--himalayas", help="flag to limit the read data to himalayas", action='store_true')
        parser.add_argument("--codadef", help="set path of CODA_DEFINITION env variable",
                            default='/lustre/storeA/project/aerocom/aerocom1/ADM_CALIPSO_TEST/')
        parser.add_argument("--latmin", help="min latitude to return", default=np.float_(30.))
        parser.add_argument("--latmax", help="max latitude to return", default=np.float_(76.))
        parser.add_argument("--lonmin", help="min longitude to return", default=np.float_(-30.))
        parser.add_argument("--lonmax", help="max longitude to return", default=np.float_(45.))
        parser.add_argument("--dir", help="work on all files below this directory",
                            default='/lustre/storeB/project/fou/kl/admaeolus/data.rev.2A02/download/AE_OPER_ALD_U_N_2A_*')
        parser.add_argument("--filemask", help="file mask to find data files",
                            default='*AE_OPER_ALD_U_N_2A_*')
        parser.add_argument("--tempdir", help="directory for temporary files",
                            default=os.path.join(os.environ['HOME'], 'tmp'))
        parser.add_argument("--plotmap", help="flag to plot a map of the data points; files will be put in outdir",
                            action='store_true')
        parser.add_argument("--plotprofile", help="flag to plot the profiles; files will be put in outdir",
                            action='store_true')
        parser.add_argument("--variables",
                            help="comma separated list of variables to write; default: ec355aer,bs355aer",
                            default='ec355aer')
        parser.add_argument("--retrieval", help="retrieval to read; supported: sca, ica, mca; default: sca",
                            default='sca')
        parser.add_argument("--netcdfcolocate", help="flag to add colocation with a netcdf file",
                            action='store_true')
        parser.add_argument("--modeloutdir",
                            help="directory for colocated model files; will have a similar filename as aeolus input file",
                            default=os.path.join(os.environ['HOME'], 'tmp'))
        parser.add_argument("--topofile", help="topography file; defaults to {}.".format(default_topo_file),
                            default=default_topo_file)
        parser.add_argument("--gridfile", help="grid data and write it to given output file (in netcdf).")

        args = parser.parse_args()

        if args.netcdfcolocate:
            options['netcdfcolocate'] = True
        else:
            options['netcdfcolocate'] = False

        if args.filemask:
            options['filemask'] = args.filemask

        if args.gridfile:
            options['gridfile'] = args.gridfile

        if args.retrieval:
            options['retrieval'] = args.retrieval

        if args.modeloutdir:
            options['modeloutdir'] = args.modeloutdir

        if args.logfile:
            options['logfile'] = args.logfile
            logging.basicConfig(filename=options['logfile'], level=logging.INFO)

        if args.dir:
            options['dir'] = args.dir

        if args.outdir:
            options['outdir'] = args.outdir

        if args.plotdir:
            options['plotdir'] = args.plotdir
        else:
            options['plotdir'] = './'

        if args.plotmap:
            options['plotmap'] = True
        else:
            options['plotmap'] = False

        if args.plotprofile:
            options['plotprofile'] = True
        else:
            options['plotprofile'] = False

        if args.tempdir:
            options['tempdir'] = args.tempdir

        if args.latmin:
            options['latmin'] = np.float_(args.latmin)

        if args.latmax:
            options['latmax'] = np.float_(args.latmax)

        if args.lonmin:
            options['lonmin'] = np.float_(args.lonmin)

        if args.lonmax:
            options['lonmax'] = np.float_(args.lonmax)

        if args.emep:
            options['emepflag'] = args.emep
            options['latmin'] = np.float(30.)
            options['latmax'] = np.float(76.)
            options['lonmin'] = np.float(-30.)
            options['lonmax'] = np.float(45.)
        else:
            options['emepflag'] = False

        if args.himalayas:
            options['himalayas'] = args.himalayas
            options['latmin'] = np.float(10.)
            options['latmax'] = np.float(50.)
            options['lonmin'] = np.float(60.)
            options['lonmax'] = np.float(110.)
        else:
            options['himalayas'] = False

        if args.readpaths:
            options['readpaths'] = args.readpaths.split(',')

        if args.variables:
            options['variables'] = args.variables.split(',')

        if args.file:
            options['files'] = args.file

        if args.listpaths:
            options['listpaths'] = True
        else:
            options['listpaths'] = False

        if args.verbose:
            options['verbose'] = True
        else:
            options['verbose'] = False

        if args.overwrite:
            options['overwrite'] = True
        else:
            options['overwrite'] = False

        if args.outfile:
            options['outfile'] = args.outfile

        if args.codadef:
            options['codadef'] = args.codadef

        if args.topofile:
            options['topofile'] = args.topofile

        # import read_data_fieldaeolus_l2a_data
        import os
        # os.environ['CODA_DEFINITION'] = options['codadef']
        import coda
        import sys
        import glob
        import pathlib
        import tarfile
        import simplegridder

        bbox = None
        obj = simplegridder.ReadL2Data(verbose=True)
        non_archive_files = []
        temp_files_dir = {}

        if 'files' not in options:
            options['files'] = glob.glob(options['dir'] + '/**/' + options['filemask'], recursive=True)
        else:
            # check if the supplied files are archives or not
            # if they are: extract files to tmp dir and work with them then
            pass
            obj.logger.info('checking if supplied files are archives...')
            for filename in options['files']:
                obj.logger.info('file: {}'.format(filename))
                suffix = pathlib.Path(filename).suffix
                if suffix in SUPPORTED_ARCHIVE_SUFFIXES:
                    temp = 'opening archive file; using {} as temp dir.'.format(options['tempdir'])
                    obj.logger.info(temp)
                    # untar archive files first
                    tarhandle = tarfile.open(filename)
                    files_in_tar = tarhandle.getnames()
                    for file_in_tar in files_in_tar:
                        if pathlib.Path(file_in_tar).suffix in SUPPORTED_SUFFIXES:
                            # extract file to tmp path
                            member = tarhandle.getmember(file_in_tar)
                            temp = 'extracting file {}...'.format(member.name)
                            obj.logger.info(temp)
                            tarhandle.extract(member, path=options['tempdir'], set_attrs=False)
                            extract_file = os.path.join(options['tempdir'], member.name)
                            non_archive_files.append(extract_file)
                            temp_files_dir[extract_file] = True
                    tarhandle.close()
                    temp_file_flag = True
                else:
                    non_archive_files.append(filename)

        options['files'] = non_archive_files

        for filename in options['files']:
            obj.logger.info('reading file: {}'.format(filename))
            suffix = pathlib.Path(filename).suffix

            if suffix not in SUPPORTED_SUFFIXES:
                print('ignoring file {}'.format(filename))
                continue
            if options['listpaths']:
                coda_handle = coda.open(filename)
                root_field_names = coda.get_field_names(coda_handle)
                for field in root_field_names:
                    print(field)
                coda.close(coda_handle)
            else:
                # read sca retrieval data
                vars_to_read = options['variables'].copy()
                # filedata_numpy = obj.read_file(filename, vars_to_read=vars_to_read, return_as='numpy',
                #                                read_dataset=options['retrieval'])

                filedata_numpy = obj.read_file(filename, vars_to_read=vars_to_read, return_as='numpy')

                obj.ndarr2data(filedata_numpy)
                # read additional data
                # ancilliary_data = obj.read_data_fields(filename, fields_to_read=['mph'])
                if filename in temp_files_dir:
                    obj.logger.info('removing temp file {}'.format(filename))
                    os.remove(filename)

        # apply emep options for cal / val
        if options['emepflag']:
            bbox = [options['latmin'], options['latmax'], options['lonmin'], options['lonmax']]
            tmp_data = obj.select_bbox(bbox)
            if len(tmp_data) > 0:
                obj.data = tmp_data
                obj.logger.info('file {} contains {} points in emep area! '.format(filename, len(tmp_data)))
            else:
                obj.logger.info('file {} contains no data in emep area! '.format(filename))
                obj = None
                # continue

        if options['himalayas']:
            bbox = [options['latmin'], options['latmax'], options['lonmin'], options['lonmax']]
            tmp_data = obj.select_bbox(bbox)
            if len(tmp_data) > 0:
                obj.data = tmp_data
                obj.logger.info('file {} contains {} points in himalaya area! '.format(filename, len(tmp_data)))
            else:
                obj.logger.info('file {} contains no data in himalaya area! '.format(filename))
                obj = None
                # continue

        # single outfile
        if 'outfile' in options:
            if len(options['files']) == 1:
                # write netcdf
                if os.path.exists(options['outfile']):
                    if options['overwrite']:
                        # obj.to_netcdf_simple(options['outfile'], global_attributes=ancilliary_data['mph'])
                        obj.to_netcdf_simple(options['outfile'], vars_to_read=vars_to_read)
                    else:
                        sys.stderr.write('Error: path {} exists'.format(options['outfile']))
                else:
                    # obj.to_netcdf_simple(options['outfile'], global_attributes=ancilliary_data['mph'])
                    obj.to_netcdf_simple(options['outfile'], vars_to_read=vars_to_read)
            else:
                sys.stderr.write("error: multiple input files, but only on output file given\n"
                                 "Please use the --outdir option instead\n")

        # grid data
        if 'gridfile' in options:
            result_flag = obj.to_grid(vars=vars_to_read)
            global_attributes = {}
            global_attributes['input files']=','.join(obj.files_read)
            global_attributes['info']='file created by simplegridder (https://github.com/metno/simplegridder) at '+\
                                      np.datetime64('now').astype('str')

            obj.to_netcdf_simple(options['gridfile'],
                                 vars_to_read=vars_to_read,
                                 global_attributes=global_attributes,
                                 gridded=True)

        # outdir
        if 'outdir' in options:
            outfile_name = os.path.join(options['outdir'], os.path.basename(filename) + '.nc')
            obj.logger.info('writing file {}'.format(outfile_name))
            global_attributes = ancilliary_data['mph']
            global_attributes['Aeolus_Retrieval'] = obj.RETRIEVAL_READ
            obj.to_netcdf_simple(outfile_name, global_attributes=global_attributes,
                                 vars_to_read=vars_to_read)

        # work with emep data and do some colocation
        if options['netcdfcolocate']:
            start_time = time.perf_counter()

            netcdf_indir = '/lustre/storeB/project/fou/kl/admaeolus/EMEPmodel'
            import xarray as xr
            # read topography since that needs to be added to the ground following height of the model
            obj.logger.info('reading topography file {}'.format(options['topofile']))
            topo_data = xr.open_dataset(options['topofile'])

            # truncate Aeolus times to hour

            aeolus_times_rounded = obj.data[:, obj._TIMEINDEX].astype('datetime64[s]').astype('datetime64[h]')
            aeolus_times = obj.data[:, obj._TIMEINDEX].astype('datetime64[s]')
            unique_aeolus_times, unique_aeolus_time_indexes = np.unique(aeolus_times, return_index=True)
            aeolus_profile_no = len(unique_aeolus_times)
            # aeolus_profile_no = int(len(aeolus_times)/obj._HEIGHTSTEPNO)
            last_netcdf_file = ''
            for time_idx in range(len(unique_aeolus_time_indexes)):
                ae_year, ae_month, ae_dummy = \
                    aeolus_times[unique_aeolus_time_indexes[time_idx]].astype('str').split('-')
                ae_day, ae_dummy = ae_dummy.split('T')
                netcdf_infile = 'CWF_12ST-{}{}{}_hourInst.nc'.format(ae_year, ae_month, ae_day)
                netcdf_infile = os.path.join(netcdf_indir, netcdf_infile)
                if not os.path.exists(netcdf_infile):
                    obj.logger.info('file does not exist: {}. skipping colocation ...'.format(netcdf_infile))
                    continue
                # read netcdf file if it has not yet been loaded
                if netcdf_infile != last_netcdf_file:
                    obj.logger.info('reading and co-locating on model file {}'.format(netcdf_infile))
                    last_netcdf_file = netcdf_infile
                    nc_data = xr.open_dataset(netcdf_infile)
                    nc_times = nc_data.time.data.astype('datetime64[h]')
                    nc_latitudes = nc_data['lat'].data
                    nc_longitudes = nc_data['lon'].data
                    nc_lev_no = len(nc_data['lev'])
                    nc_colocated_data = np.zeros([aeolus_profile_no * nc_lev_no, obj._COLNO], dtype=np.float_)

                # locate current rounded Aeolus time in netcdf file
                nc_ts_no = np.where(nc_times == unique_aeolus_times[time_idx].astype('datetime64[h]'))
                if len(nc_ts_no) != 1:
                    # something is wrong here!
                    pass

                # locate current profile's location index in lats and lons
                # Has to be done on original aeolus data
                for aeolus_profile_index in range(aeolus_profile_no):
                    data_idx = unique_aeolus_time_indexes[aeolus_profile_index]
                    try:
                        data_idx_end = unique_aeolus_time_indexes[aeolus_profile_index + 1]
                    except:
                        data_idx_end = len(aeolus_times)

                    data_idx_arr = np.arange(data_idx_end - data_idx) + data_idx

                    aeolus_lat = np.nanmean(obj.data[data_idx_arr, obj._LATINDEX])
                    aeolus_lon = np.nanmean(obj.data[data_idx_arr, obj._LONINDEX])
                    aeolus_altitudes = obj.data[data_idx_arr, obj._ALTITUDEINDEX]
                    diff_dummy = nc_latitudes - aeolus_lat
                    min_lat_index = np.argmin(np.abs(diff_dummy))
                    diff_dummy = nc_longitudes - aeolus_lon
                    min_lon_index = np.argmin(np.abs(diff_dummy))

                    nc_data_idx = aeolus_profile_index * nc_lev_no
                    nc_index_arr = np.arange(nc_lev_no) + nc_data_idx
                    nc_colocated_data[nc_index_arr, obj._EC355INDEX] = \
                        nc_data['EXT_350nm'].data[nc_ts_no, :, min_lat_index, min_lon_index]
                    # nc_data['EXT_350nm'].data[nc_ts_no,:,min_lat_index,min_lon_index].reshape(nc_lev_no)
                    nc_colocated_data[nc_index_arr, obj._ALTITUDEINDEX] = \
                        nc_data['Z_MID'].data[nc_ts_no, :, min_lat_index, min_lon_index] + \
                        topo_data['topography'].data[0, min_lat_index, min_lon_index]
                    nc_colocated_data[nc_index_arr, obj._LATINDEX] = \
                        nc_data['lat'].data[min_lat_index]
                    nc_colocated_data[nc_index_arr, obj._LONINDEX] = \
                        nc_data['lon'].data[min_lon_index]
                    # nc_data['Z_MID'].data[nc_ts_no,:,min_lat_index,min_lon_index].reshape(nc_lev_no)
                    nc_colocated_data[nc_index_arr, obj._TIMEINDEX] = \
                        obj.data[data_idx, obj._TIMEINDEX]

            end_time = time.perf_counter()
            elapsed_sec = end_time - start_time
            temp = 'time for colocation all time steps [s]: {:.3f}'.format(elapsed_sec)
            if 'nc_colocated_data' in locals():
                obj.logger.info(temp)
                obj.logger.info('{} is colocated model output directory'.format(options['modeloutdir']))
                model_file_name = os.path.join(options['modeloutdir'],
                                               os.path.basename(filename) + '.colocated.nc')
                obj.to_netcdf_simple(model_file_name, data_to_write=nc_colocated_data)
            pass

        # plot the profile
        if options['plotprofile']:
            plotfilename = os.path.join(options['outdir'], os.path.basename(filename)
                                        + '.' + options['retrieval'] + '.profile.png')
            obj.logger.info('profile plot file: {}'.format(plotfilename))
            # title = '{} {}'.format(options['retrieval'], os.path.basename(filename))
            title = '{}'.format(os.path.basename(filename))
            obj.plot_profile_v2(plotfilename, title=title,
                                retrieval_name=options['retrieval'],
                                plot_range=(0., 200.))

        # plot the map
        if options['plotmap']:
            if len(obj.gridded_data.keys()) == 0:
                #gridding has not been done yet
                result_flag = obj.to_grid(vars=vars_to_read)

            if len(obj.files_read) == 1:
                #single file read
                plotmapfilename = os.path.join(options['plotdir'], '_'.join([options['variables'][0],
                                                                             os.path.basename(obj.files_read[0])]) + '.map.png')
                title='\n'.join([options['variables'][0],os.path.basename(obj.files_read[0])])
                obj.plot_map(plotmapfilename, bbox=bbox, title=title)
            else:
                # archive file read
                plot_date = np.datetime64(obj.gridded_data['time'], 'D').astype('str')
                for var in options['variables']:

                    plotmapfilename = os.path.join(options['plotdir'], '_'.join([var, plot_date]) + '.map.png')
                    obj.logger.info('map plot file: {}'.format(plotmapfilename))
                    title='_'.join([var, plot_date])
                    # title = os.path.basename(filename)

                    obj.plot_map(plotmapfilename, bbox=bbox, title=title)
                    # obj.plot_location_map(plotmapfilename)



