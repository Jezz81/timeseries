import datetime as dt
import pandas as pd
import os

class TimeSeries:

    def __init__(self, data_sources):
        self.__data_sources = data_sources
        self.__dfs = []
        self.__start_date = None
        self.__end_date = None
        self.__process_data_sources()

    def __process_data_sources(self):


        for ds,schema in self.__data_sources.items():
            if os.path.exists(ds):
                if os.path.isdir(ds):
                    files = os.listdir(ds)
                    for f in files:
                        full_path = os.path.join(ds,f)
                        print(full_path)
                        if os.path.isfile(full_path):
                            if schema['format'] == 'xlsx':
                                df = self.__read_xlsx_file(full_path,schema)
                                self.__dfs.append(df)
                            elif schema['format'] == 'txt':
                                df = self.__read_txt_file(full_path,schema)
                                self.__dfs.append(df)
                            else:
                                raise Exception('Format {0} is not recognised'.format(schema['format']))
                        

                if os.path.isfile(ds):
                    if schema['format'] == 'xlsx':
                        df = self.__read_xlsx_file(ds,schema)
                        self.__dfs.append(df)
                    elif schema['format'] == 'txt':
                        df = self.__read_txt_file(ds,schema)
                        self.__dfs.append(df)
                    else:
                        raise Exception('Format {0} is not recognised'.format(schema['format']))
            else:
                print('{0} not found'.format(ds))

        all_dates=[]
        for df in self.__dfs:
            all_dates.extend(df.index.values.tolist())

        start_date = min(all_dates)
        end_date = max(all_dates)
        global_df = pd.DataFrame(index=pd.date_range(start=start_date,end=end_date))

        for df in self.__dfs:
            global_df = global_df.join(df,rsuffix='_R')

        global_df.to_excel('all.xlsx')

    def __read_xlsx_file(self, filename,schema):
        print('reading xlsx file')
        header_row = schema.get('header', 0)
        sheet_name = schema.get('sheet_name','Data')
        date_col = schema['date']

        df = pd.read_excel(filename,sheet_name=sheet_name,header=header_row)
        df.to_excel('diag.xlsx')
        df.set_index(date_col,inplace=True)

        return df

    @staticmethod
    def __slice_row(row,start,end):
        return None



    def __read_txt_file(self,file_name,schema):
        header_row = schema.get('header',0)
        delimiter = schema.get('delimiter','\t')
        thousands = schema.get('thousands',',')
        date_col = schema.get('date',None)
        century = schema.get('century',None)
        year = schema.get('year',None)
        month = schema.get('month',None)
        day = schema.get('day',None)
        df = pd.read_csv(file_name, header=header_row, index_col=None, delimiter=delimiter, thousands=thousands,
                         engine='python')

        if date_col:
            df[date_col] = df[date_col].apply(self.__set_date_longform)
            df.reset_index(drop=True, inplace=True)
            df.set_index([date_col], inplace=True)
        elif century or year:
            if century:
                df['century'] = df[0]

        df.dropna(inplace=True)

        if header_row:
            df = df.rename(columns=lambda x: x.strip())
        fname_prepend = os.path.basename(file_name)
        fname_prepend = ''.join(fname_prepend.split('.')[0:-2])
        if schema.get('prepend_filename',False):
            df = df.rename(columns=lambda x: fname_prepend+x)

        df.to_excel('fname.xlsx')
        return df

    def __set_date_longform(self,date_val):
        fmts = ['%b %d, %Y', '%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%b-%d %H:%M']
        for fmt in fmts:
            try:
                return dt.datetime.strptime(date_val.strip(), fmt)
            except ValueError:
                pass

    # def __update_start_end(self,current_start,current_end):
    #
    #     if current_start:

if __name__ == '__main__':

    # data_sources = {'/home/q/Development/Downloaders/Macro':{'header':0,'date':'Date','delimiter':'\t','format':'txt'},
    #                 '/home/q/Development/Downloaders/Crypto/ADA-USD.CC.txt':{'header':0,'date':'Date','delimiter':'\t', 'format': 'txt'},
    #                 '/home/q/Development/Downloaders/b03hist.xls':{'header':10,'date':'Series ID','sheet_name':'Data','format':'xlsx'},
    #                 '/home/q/Development/Downloaders/Solar':{'header':-1,'year':[0,2],'month':[3,4],'day':[5,6],'AP':[43,46],'format':'txt'}}

    data_sources = {
                    '/home/q/Development/Downloaders/Solar':{'header':None,'year':[0,2],'month':[3,4],'day':[5,6],'AP':[43,46],'format':'txt'}}

    timeseries = TimeSeries(data_sources)

