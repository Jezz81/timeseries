import datetime as dt
import pandas as pd
import os
import numpy as np


class TimeSeries:

    '''
    initiated with a dictionary
    key = filename or source directory
    values = dictionary

    header: None or header row for xlsx or txt files
    delimiter: None or single char. If none must pass dictionary of column cutoffs
    date: date column
    century,year,month,day: either column headers or column cutoffs if no header
    data: dictionary of column cutoffs. key becomes column header
    keep_cols: Columns to keep, all others are discarded
    prepend_filename: Prepend the 0th split('.') of the base filename to column headers
    '''

    def __init__(self, data_sources):
        self.__data_sources = data_sources
        self.__dfs = []
        self.__start_date = None
        self.__end_date = None
        self.__global_df = self.__process_data_sources()

    def __process_data_sources(self):
        for ds, schema in self.__data_sources.items():
            if os.path.exists(ds):
                if os.path.isdir(ds):
                    files = os.listdir(ds)
                    tmp_dfs = []
                    for f in files:
                        full_path = os.path.join(ds, f)
                        if os.path.isfile(full_path):
                            print('reading {0}'.format(full_path))
                            if schema['format'] == 'xlsx':
                                df = self.__read_xlsx_file(full_path, schema)
                                tmp_dfs.append(df)
                            elif schema['format'] == 'txt':
                                df = self.__read_txt_file(full_path, schema)
                                tmp_dfs.append(df)
                            else:
                                raise Exception('Format {0} is not recognised'.format(schema['format']))
                    df = pd.concat(tmp_dfs)
                    self.__dfs.append(df)

                if os.path.isfile(ds):
                    print('reading {0}'.format(ds))
                    if schema['format'] == 'xlsx':
                        df = self.__read_xlsx_file(ds, schema)
                        self.__dfs.append(df)
                    elif schema['format'] == 'txt':
                        df = self.__read_txt_file(ds, schema)
                        self.__dfs.append(df)
                    else:
                        raise Exception('Format {0} is not recognised'.format(schema['format']))
            else:
                raise FileNotFoundError('{0} not found'.format(ds))

        all_dates = []
        for df in self.__dfs:
            all_dates.extend(df.index.values.tolist())

        start_date = min(all_dates)
        end_date = max(all_dates)
        global_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))

        for df in self.__dfs:
            global_df = global_df.join(df, rsuffix='_R')

        return global_df

    @staticmethod
    def __read_xlsx_file(filename, schema):
        header_row = schema.get('header', 0)
        sheet_name = schema.get('sheet_name', 'Data')
        date_col = schema['date']

        df = pd.read_excel(filename, sheet_name=sheet_name, header=header_row)
        df.set_index(date_col, inplace=True)

        fname_prepend = os.path.basename(filename)
        fname_prepend = ''.join(fname_prepend.split('.')[0:-2])
        if schema.get('prepend_filename', False):
            df.rename(columns=lambda x: fname_prepend + x, inplace=True)

        return df

    @staticmethod
    def __read_txt_file(file_name, schema):
        header_row = schema.get('header', None)
        delimiter = schema.get('delimiter', None)
        thousands = schema.get('thousands', ',')
        date_col = schema.get('date', None)
        century = schema.get('century', None)
        year = schema.get('year', None)
        month = schema.get('month', None)
        day = schema.get('day', None)
        data = schema.get('data', None)
        keep_cols = schema.get('keep_cols', [])
        df = pd.read_csv(file_name, header=header_row, index_col=None, delimiter=delimiter, thousands=thousands,
                         engine='python')

        if header_row == None and delimiter:
            raise ValueError('Header row must be specified if delimiter is not None')

        if header_row and not date_col:
            raise ValueError('Header row must contain a date col if header not None')

        if header_row and data:
            raise ValueError('Data must be none if header not None. Specify keep_cols if required')

        if date_col:
            df[date_col] = df[date_col].apply(TimeSeries.__set_date_longform)
            df.rename(columns={date_col: 'date'}, inplace=True)
        else:
            if header_row:
                df['date'] = df.apply(TimeSeries.__get_date, axis=1)
            else:
                if not delimiter:
                    if century:
                        df['century'] = df.apply(TimeSeries.__get_part, args=(century[0], century[1]), axis=1)
                    if year:
                        df['year'] = df.apply(TimeSeries.__get_part, args=(year[0], year[1]), axis=1)
                    if month:
                        df['month'] = df.apply(TimeSeries.__get_part, args=(month[0], month[1]), axis=1)
                    if day:
                        df['day'] = df.apply(TimeSeries.__get_part, args=(day[0], day[1]), axis=1)
                    df.dropna(inplace=True)
                    df['date'] = df.apply(TimeSeries.__get_date, axis=1)
                else:
                    if isinstance(century, str):
                        df.rename(columns={century, 'century'}, inplace=True)
                    else:
                        raise ValueError('century must be a string if delimiter is None')
                    if isinstance(year, str):
                        df.rename(columns={year: 'year'}, inplace=True)
                    else:
                        raise ValueError('year must be a string if delimiter is None')
                    if isinstance(month, str):
                        df.rename(columns={month: 'month'}, inplace=True)
                    else:
                        raise ValueError('month must be a string if delimiter is None')
                    if isinstance(day, str):
                        df.rename(columns={day: 'day'}, inplace=True)
                    else:
                        raise ValueError('day must be a string if delimiter is None')
                    df.dropna(inplace=True)
                    df['date'] = df.apply(TimeSeries.__get_date, axis=1)
                    df.drop(columns=['century', 'year', 'month', 'day'], inplace=True)
                if data:
                    for k, v in data.items():
                        df[k] = df.apply(TimeSeries.__get_part, args=(v[0], v[1]), axis=1)

        df.dropna(inplace=True)
        if header_row:
            df = df.rename(columns=lambda x: x.strip())

        df.reset_index(drop=True, inplace=True)
        df.set_index('date', inplace=True)
        fname_prepend = os.path.basename(file_name)

        fname_prepend = ''.join(fname_prepend.split('.')[0])
        if schema.get('prepend_filename', False):
            df.rename(columns=lambda x: fname_prepend + x, inplace=True)

        drop_cols = []
        if data:
            for col in df.columns:
                if col not in data:
                    drop_cols.append(col)
        else:
            if keep_cols:
                for col in df.columns:
                    if col not in keep_cols:
                        drop_cols.append(col)

        df.drop(columns=drop_cols, inplace=True)

        return df

    @property
    def return_df(self):
        return self.__global_df

    @staticmethod
    def __get_part(row, start, end):
        test_val = row[0][start:end]
        test_val = test_val.strip()
        if test_val == '':
            test_val = np.nan
        return test_val

    @staticmethod
    def __get_date(row):
        century = row.get('century', 2000)
        try:
            return dt.datetime(century + int(row['year']), int(row['month']), int(row['day']))
        except ValueError:
            return None

    @staticmethod
    def __set_date_longform(date_val):
        fmts = ['%b %d, %Y', '%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%b-%d %H:%M']
        for fmt in fmts:
            try:
                return dt.datetime.strptime(date_val.strip(), fmt)
            except ValueError:
                pass


# if __name__ == '__main__':
#
#
#     data_sources= {'USDJPY.FOREX.txt': {'header': 0, 'date': 'Date', 'header': 0,'delimiter':'\t','format': 'txt'},
#                     'AUDUSD.xlsx': {'header': 0, 'date': 'Date', 'sheet_name': 'Data','format': 'xlsx'},
#                     'Solar': {'header': None, 'year': [0, 2], 'month': [2, 4],
#                                                               'day': [4, 6], 'data': {'AP': [43, 46]}, 'format': 'txt'},
#                     'b03hist.xls': {'header': 10, 'date': 'Series ID',
#                                                                     'sheet_name': 'Data', 'format': 'xlsx'}}
#
#
#     timeseries = TimeSeries(data_sources)
#     global_df = timeseries.return_df
#     #global_df.dropna(inplace=True)
#     global_df.to_excel('example.xlsx')

