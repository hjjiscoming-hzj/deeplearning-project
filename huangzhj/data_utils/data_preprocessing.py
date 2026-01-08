import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class DataPreprocessing(object):
    def __init__(self):
        self.name = 'dataset'

    def get_data(self): pass


class DataPreprocessing_Merak(DataPreprocessing):
    def __init__(self):
        super().__init__()
        self.name = 'Merak'
        # 读取CSV文件
        data = pd.read_csv(r"D:\DeepLearning\Datasets\Merak\dataset_1105.csv")

        # 将目标变量和特征分开
        self.X = data.iloc[:, 2:21]
        self.y = data.iloc[:, [22]]

        self.target = self.y.columns.tolist()
        print('Target: ', self.target)

    def get_data(self):
        return self.X,self.y


class DataPreprocessing_Scm(DataPreprocessing):
    def __init__(self):
        super().__init__()
        self.name = 'Scm'
        # 读取CSV文件
        data = pd.read_csv(r'D:\DeepLearning\Datasets\scm\scm20d_data.csv')
        label = pd.read_csv(r'D:\DeepLearning\Datasets\scm\scm20d_label.csv')

        # 将目标变量和特征分开
        self.X = data
        self.y = label.iloc[:,[0]]

        self.target = self.y.columns.tolist()
        print('Target: ', self.target)

    def get_data(self):
        return self.X, self.y


class DataPreprocessing_S4e12(DataPreprocessing):
    def __init__(self):
        super().__init__()
        self.name = 's4e12'
        # 读取文件并创建数据集
        data = pd.read_csv(r'D:\DeepLearning\Datasets\playground-series-s4e12/train.csv',nrows=20000)

        # 填补缺失值
        numeric_columns = ['Age', 'Annual Income', 'Number of Dependents', 'Health Score', 'Previous Claims',
                           'Vehicle Age',
                           'Credit Score', 'Insurance Duration']
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

        categorical_columns = ['Marital Status', 'Occupation', 'Customer Feedback']
        data[categorical_columns] = data[categorical_columns].fillna('unknown')

        # 将 'Policy Start Date' 列转换为 datetime 类型
        data['Policy Start Date'] = pd.to_datetime(data['Policy Start Date'])

        # 提取年、月、日、时、分、秒
        data['Year'] = data['Policy Start Date'].dt.year
        data['Month'] = data['Policy Start Date'].dt.month
        data['Day'] = data['Policy Start Date'].dt.day
        data['Hour'] = data['Policy Start Date'].dt.hour
        data['Minute'] = data['Policy Start Date'].dt.minute
        data['Second'] = data['Policy Start Date'].dt.second

        # # 遍历所有object类型的字段，查看这些字段的unique()值
        # for column in data_utils.select_dtypes(include=['object']).columns:
        #     unique_values = data_utils[column].unique()
        #     print(f"Unique values in '{column}': {unique_values}")

        # 创建教育水平的映射字典
        education_mapping = {
            'High School': 1,
            "Bachelor's": 2,
            "Master's": 3,
            'PhD': 4
        }
        policy_type = {
            'Basic': 1,
            'Comprehensive': 2,
            'Premium': 3
        }
        customer = {
            'unknown': 0,
            'Poor': 1,
            'Average': 2,
            'Good': 3

        }
        exercise = {
            'Rarely': 1,
            'Monthly': 2,
            'Weekly': 3,
            'Daily': 4
        }

        # 使用映射字典进行序列编码
        data['Education Level'] = data['Education Level'].map(education_mapping)
        data['Policy Type'] = data['Policy Type'].map(policy_type)
        data['Customer Feedback'] = data['Customer Feedback'].map(customer)
        data['Exercise Frequency'] = data['Exercise Frequency'].map(exercise)

        from sklearn.preprocessing import OneHotEncoder

        # 需要处理的列
        columns_to_encode = [
            'Gender',
            'Marital Status',
            'Occupation',
            'Location',
            'Smoking Status',
            'Property Type'
        ]

        # 初始化one-hot编码器
        encoder = OneHotEncoder(drop='first', sparse_output=False)

        # 拆分编码器拟合和转换的步奏
        data_encoded = encoder.fit_transform(data[columns_to_encode])

        # 将编码后的数据转换为DataFrame，并与原始数据拼接
        data_encoded_df = pd.DataFrame(data_encoded, columns=encoder.get_feature_names_out(columns_to_encode))

        # 删除原始编码列，并在数据集中合并新编码的列
        all_data_encoded = data.drop(columns=columns_to_encode).reset_index(drop=True)

        all_data_encoded = pd.concat([all_data_encoded, data_encoded_df], axis=1)

        # 划分好目标变量和特征
        self.X = pd.DataFrame(all_data_encoded.drop(['id', 'Premium Amount', 'Policy Start Date'], axis=1))
        self.y = pd.DataFrame(all_data_encoded['Premium Amount'])

    def get_data(self):
        return self.X, self.y


class DataPreprocessing_andro(DataPreprocessing):
    def __init__(self):
        super().__init__()
        self.name = 'andro'
        # 读取CSV文件
        data = pd.read_csv(r"D:\DeepLearning\Datasets\andromeda\andro.csv")

        # 将目标变量和特征分开
        self.X = data.iloc[:, 0:30]
        self.y = data.iloc[:, [30]]

        self.target = self.y.columns.tolist()
        print('Target: ', self.target)

    def get_data(self):
        return self.X,self.y


class DataPreprocessing_winequality(DataPreprocessing):
    def __init__(self):
        super().__init__()
        self.name = 'winequality'
        # 读取CSV文件
        data = pd.read_csv(r"D:\DeepLearning\Datasets\wine+quality\winequality-white.csv",sep=';')

        # 将目标变量和特征分开
        self.X = data.iloc[:, 0:11]
        self.y = data.iloc[:, [11]]

        self.target = self.y.columns.tolist()
        print('Target: ', self.target)

    def get_data(self):
        return self.X,self.y


class DataPreprocessing_CA(DataPreprocessing):
    def __init__(self):
        super().__init__()
        self.name = 'CA'
        # 读取CSV文件
        data = pd.read_csv(r"D:\DeepLearning\Datasets\CA\housing.csv")
        # 填补缺失值
        numeric_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                           'population','households', 'median_income', 'median_house_value']
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

        categorical_columns = ['ocean_proximity']
        data[categorical_columns] = data[categorical_columns].fillna('unknown')

        # 需要处理的列
        columns_to_encode = [
            'ocean_proximity'
        ]
        # 初始化one-hot编码器
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        # 拆分编码器拟合和转换的步奏
        data_encoded = encoder.fit_transform(data[columns_to_encode])
        # 将编码后的数据转换为DataFrame，并与原始数据拼接
        data_encoded_df = pd.DataFrame(data_encoded, columns=encoder.get_feature_names_out(columns_to_encode))
        # 删除原始编码列，并在数据集中合并新编码的列
        all_data_encoded = data.drop(columns=columns_to_encode).reset_index(drop=True)
        all_data_encoded = pd.concat([all_data_encoded, data_encoded_df], axis=1)

        # 将目标变量和特征分开
        # 划分好目标变量和特征
        self.X = pd.DataFrame(all_data_encoded.drop(['median_house_value'], axis=1))
        self.y = pd.DataFrame(all_data_encoded['median_house_value'])

        self.target = self.y.columns.tolist()
        print('Target: ', self.target)

    def get_data(self):
        return self.X,self.y


class DataPreprocessing_YE(DataPreprocessing):
    def __init__(self):
        super().__init__()
        self.name = 'YE'
        # 读取CSV文件
        data = pd.read_csv(r"D:\DeepLearning\Datasets\YE\year_prediction.csv",nrows=20000)
        # 填补缺失值
        data = data.fillna(data.mean())

        # 将目标变量和特征分开
        # 划分好目标变量和特征
        self.X = pd.DataFrame(data.drop(['label'], axis=1))
        self.y = pd.DataFrame(data['label'])

        self.target = self.y.columns.tolist()
        print('Target: ', self.target)

    def get_data(self):
        return self.X,self.y


class DataPreprocessing_Boston(DataPreprocessing):
    def __init__(self):
        super().__init__()
        self.name = 'Boston'
        # 读取CSV文件
        data = pd.read_csv(r"D:\DeepLearning\Datasets\Boston\Boston.csv")
        # 填补缺失值
        # 填补缺失值
        numeric_columns = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM',
                           'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

        categorical_columns = ['CHAS']
        data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])

        # 需要处理的列
        columns_to_encode = categorical_columns
        # 初始化one-hot编码器
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        # 拆分编码器拟合和转换的步奏
        data_encoded = encoder.fit_transform(data[columns_to_encode])
        # 将编码后的数据转换为DataFrame，并与原始数据拼接
        data_encoded_df = pd.DataFrame(data_encoded, columns=encoder.get_feature_names_out(columns_to_encode))
        # 删除原始编码列，并在数据集中合并新编码的列
        all_data_encoded = data.drop(columns=columns_to_encode).reset_index(drop=True)
        all_data_encoded = pd.concat([all_data_encoded, data_encoded_df], axis=1)
        # 将目标变量和特征分开
        # 划分好目标变量和特征
        self.X = pd.DataFrame(all_data_encoded.drop(['MEDV'], axis=1))
        self.y = pd.DataFrame(all_data_encoded['MEDV'])

        self.target = self.y.columns.tolist()
        print('Target: ', self.target)

    def get_data(self):
        return self.X,self.y


class DataPreprocessing_Bike(DataPreprocessing):
    def __init__(self):
        super().__init__()
        self.name = 'Bike'
        # 读取CSV文件
        data = pd.read_csv(r"D:\DeepLearning\Datasets\bike+sharing+dataset\hour.csv")
        # 处理缺失值（如果有的话）
        data.dropna(inplace=True)

        # 需要处理的列
        columns_to_encode = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
        # 初始化one-hot编码器
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        # 拆分编码器拟合和转换的步奏
        data_encoded = encoder.fit_transform(data[columns_to_encode])
        # 将编码后的数据转换为DataFrame，并与原始数据拼接
        data_encoded_df = pd.DataFrame(data_encoded, columns=encoder.get_feature_names_out(columns_to_encode))
        # 删除原始编码列，并在数据集中合并新编码的列
        all_data_encoded = data.drop(columns=columns_to_encode).reset_index(drop=True)
        all_data_encoded = pd.concat([all_data_encoded, data_encoded_df], axis=1)

        # 将目标变量和特征分开
        # 划分好目标变量和特征
        self.X = pd.DataFrame(all_data_encoded.drop(['cnt','instant','dteday','casual','registered'], axis=1))
        self.y = pd.DataFrame(all_data_encoded['cnt'])

        self.target = self.y.columns.tolist()
        print('Target: ', self.target)

    def get_data(self):
        return self.X,self.y

class DataPreprocessing_HS(DataPreprocessing):
    def __init__(self):
        super().__init__()
        self.name = 'HS'
        # 读取CSV文件
        data = pd.read_csv(r"D:\DeepLearning\Datasets\Merak\HS_unit_dataset_1216.csv")

        # 将目标变量和特征分开
        # 划分好目标变量和特征
        self.X = data.iloc[:, 2:22]
        self.y = pd.DataFrame(data['metal6_Peak'])

        self.target = self.y.columns.tolist()
        print('Target: ', self.target)

    def get_data(self):
        return self.X,self.y


class DataPreprocessing_HS_mos1(DataPreprocessing):
    def __init__(self):
        super().__init__()
        self.name = 'HS_mos1'
        # 读取CSV文件
        data = pd.read_csv(r"D:\DeepLearning\Datasets\Merak\HS_mos1.csv")

        # 将目标变量和特征分开
        # 划分好目标变量和特征
        self.X = data.iloc[:, 2:22]
        self.y = pd.DataFrame(data['metal6_Peak'])

        # 去掉不需要的特征列
        # columns_to_drop = ['fp_col', 'last_adv_width', 'next_adv_width']
        # self.X = self.X.drop(columns=columns_to_drop)

        self.target = self.y.columns.tolist()
        print('Target: ', self.target)

    def get_data(self):
        return self.X,self.y


class DataPreprocessing_HS_mos2(DataPreprocessing):
    def __init__(self):
        super().__init__()
        self.name = 'HS_mos2'
        # 读取CSV文件
        data = pd.read_csv(r"D:\DeepLearning\Datasets\Merak\HS_mos2.csv")

        # 将目标变量和特征分开
        # 划分好目标变量和特征
        self.X = data.iloc[:, 2:22]
        self.y = pd.DataFrame(data['metal6_Peak'])

        self.target = self.y.columns.tolist()
        print('Target: ', self.target)

    def get_data(self):
        return self.X,self.y


class DataPreprocessing_UD(DataPreprocessing):
    def __init__(self):
        super().__init__()
        self.name = 'HD'
        # 读取CSV文件
        data = pd.read_csv(r"D:\DeepLearning\Datasets\Merak\UD_unit_dataset_1105.csv")

        # 将目标变量和特征分开
        # 划分好目标变量和特征
        self.X = data.iloc[:, 2:21]
        self.y = pd.DataFrame(data['ME5_Peak'])

        self.target = self.y.columns.tolist()
        print('Target: ', self.target)

    def get_data(self):
        return self.X,self.y