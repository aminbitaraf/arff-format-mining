import pandas as pd
import numpy as np


class Mining:
    # features name with type
    columns = dict()
    # our main data
    df = None
    # all of type we support
    types = ['Numeric', 'Binary', 'Nominal']

    def __init__(self, file):
        self.f = open(file, 'r').readlines()
        # get name of data
        self.name = self.f[0][11:-2]
        # create csv file for save data as csv
        self.csv_file = open(self.name + '.csv', 'w')
        self.__features = pd.DataFrame(columns=['name', 'type', 'domain'])
        self.__numeric_features_details = pd.DataFrame(columns=['name', 'mean', 'median', 'mode', 'variance', 'std'])
        self.__read_file()
        self.__feature_details()
        self.__add_num_features_details()

    # --------get variables as csv--------
    def get_features(self):
        return self.__features.to_csv(index=False)

    def get_numeric_features_details(self):
        return self.__numeric_features_details.to_csv(index=False)

    # --------math functions--------
    def mean(self, column):
        return self.df[column].mean()

    def median(self, column):
        return self.df[column].median()

    def mode(self, column):
        return self.df[column].mode()[0]

    def variance(self, column):
        return self.df[column].var()

    def standard_deviation(self, column):
        return self.df[column].std()

    # --------helper functions--------
    def min_domain(self, column):
        return self.df[column].min()

    def max_domain(self, column):
        return self.df[column].max()

    # --------private functions--------
    # separate different types by numeric,nominal,binary
    def __find_column(self, line):
        # type of attribute is Numeric for default
        attr_type = 'Numeric'
        # if end of line was '}' mean it's object(binary or nominal)
        if line[-2] == '}':
            # if object has "{'0', '1'}" form mean it's binary
            attr_type = 'Binary' if line[-11:-1] == "{'0', '1'}" or line[-10:-1] == "{'0','1'}" else 'Nominal'

        # the 12 character of first is --@attribute '-- and we want word of between '
        attr_name = line[12:12 + line[12:].find('\'')]
        self.__save_column(attr_name, attr_type)

    # save column name with type
    def __save_column(self, attr_name, attr_type):
        self.columns[attr_name] = attr_type

    # save __features details(name,type,domain(if type was numeric))
    def __feature_details(self):
        for key in self.columns:
            # only numeric values have domain
            domain = f'{self.min_domain(key)}-{self.max_domain(key)}' if self.columns[key] == 'Numeric' else 'None'
            self.__features = self.__features.append(
                {'name': key, 'type': self.columns[key], 'domain': domain}, ignore_index=True, sort=False)

    # save numeric __features details(mean,median,mode,variance,standard deviation)
    def __add_num_features_details(self):
        for key in self.__features[self.__features['type'] == 'Numeric']['name']:
            self.__numeric_features_details = self.__numeric_features_details.append(
                {'name': key, 'mean': self.mean(key), 'median': self.median(key),
                 'mode': self.mode(key), 'variance': self.variance(key),
                 'std': self.standard_deviation(key)}, ignore_index=True, sort=False)

    # get attributes as column and convert data to csv file
    def __read_file(self):
        for line in self.f:
            if line[0] == '@':
                if line[:10] == "@attribute":
                    self.__find_column(line)
                elif line[:5] == "@data":
                    # save attributes as column to csv file when we get all of attributes
                    self.csv_file.write(','.join(list(self.columns.keys())) + '\n')
                else:
                    continue
            else:
                self.csv_file.write(line)
        self.df = pd.read_csv(self.name + '.csv')

    # calculate the dissimilarity matrix by all type
    def dissimilarity_matrix(self, limit=0):
        len_df = len(self.df)
        # do all of data if didn't set limit (if limit is 0)
        limit = len_df if limit == 0 or len_df < limit else limit
        nominal_matrix = None
        binary_matrix = None
        numeric_matrix = None
        for _type in self.types:
            # get only data is _type (ex: numeric,nominal,binary) per time
            data_frame = self.df[self.__features[self.__features['type'] == _type]['name']][:limit]
            if _type == 'Numeric':
                numeric_matrix = self.numeric_dissimilarity_matrix(data_frame)
            elif _type == 'Nominal':
                nominal_matrix = self.nominal_or_binary_dissimilarity_matrix(data_frame)
            elif _type == 'Binary':
                binary_matrix = self.nominal_or_binary_dissimilarity_matrix(data_frame)
        return pd.DataFrame((nominal_matrix + binary_matrix + numeric_matrix) / 3).to_csv(index=False, header=False)

    # get dissimilarity matrix for numeric values
    @staticmethod
    def numeric_dissimilarity_matrix(data_frame):
        # normalization data ((current data - minimum of data) / (maximum of data - minimum of data))
        data_frame = ((data_frame - data_frame.min()) / (data_frame.max() - data_frame.min()))
        len_df = len(data_frame)
        # create matrix with zero value
        matrix = np.zeros((len_df, len_df))
        for i in range(0, len_df - 1):
            for j in range(i + 1, len_df):
                # calculate Euclidean Distance
                matrix[j, i] = np.sqrt(np.sum(np.square(np.abs(data_frame.iloc[i] - data_frame.iloc[j]))))
        return matrix

    # get dissimilarity matrix for normal or binary values
    @staticmethod
    def nominal_or_binary_dissimilarity_matrix(data_frame):
        len_df = len(data_frame)
        # create matrix with zero value
        matrix = np.zeros((len_df, len_df))
        for i in range(0, len_df - 1):
            for j in range(i + 1, len_df):
                # return true when data wasn't equal
                matrix[j, i] = np.mean((data_frame.iloc[i] != data_frame.iloc[j]).astype(int))
        return matrix

    def __del__(self):
        self.csv_file.close()


LIMIT_ROW_FOR_DISSIMILARITY_MATRIX = 30
PATH_OF_INPUT_FILE = './KDDTest-21.arff.txt'
PATH_OF_OUTPUT_FILE = './output.txt'

if __name__ == '__main__':
    # send path of data file
    data = Mining(PATH_OF_INPUT_FILE)

    # path of output data
    output_file = open(PATH_OF_OUTPUT_FILE, 'w')
    output_file.write(data.get_features() +
                      '\n' +
                      data.dissimilarity_matrix(limit=LIMIT_ROW_FOR_DISSIMILARITY_MATRIX) +
                      '\n' +
                      data.get_numeric_features_details())
    print("Well Done!")
