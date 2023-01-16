import random
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.io as pio
import plotly.express as px
import matplotlib.pyplot as plt
from scipy import stats
from pylab import mpl
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, TruncatedSVD
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from MeanEncoder import MeanEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder, CatBoostEncoder,OneHotEncoder
from feature_selector import FeatureSelector

# from pandas.testing import assert_frame_equal

mpl.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决减号符号显示问题

class Data_Preprocess():

    def __init__(self, data_path):
        self.data_path = data_path

        if self.data_path.endswith('.csv'):
            self.data = pd.read_csv(self.data_path)
        elif self.data_path.endswith('.xlsx'):
            self.data = pd.read_excel(self.data_path)
        else:
            raise Exception('Invalid file format')

    def drop_columns(self, *args, threshold: float = 0.5):

        # drop the given columns
        if args:
            self.data.drop(list(args), axis=1, inplace=True)

        # drop columns with missing values greater than threshold
        ops = {'missing': []}
        for col in self.data.columns:
            if self.data[col].isnull().sum() / len(self.data) > threshold:
                ops['missing'].append(col)
                self.data.drop(col, axis=1, inplace=True)
        
        missing_features = ops['missing']
        print('缺失值占比大于[ {} ]的特征有[ {} ]个\n分别是：[ {} ]'.format(threshold, len(missing_features), missing_features))


    def drop_elements(self, col_elements: dict = {}):
        # drop the given elements in the given columns
        columns_list = list(col_elements.keys())

        if type(list(col_elements.values())[0]) == list:
            for column in columns_list:
                for elements in col_elements[column]:
                    if elements in self.data[column].unique():
                        idx = self.data[self.data[column] == elements].index
                        self.data.drop(index=idx, inplace=True)
                        print('element： [ {} ] \t in the column： [ {} ] \t has been deleted'.format(elements, column))
                    else:
                        raise Exception('element：“ {} ” is not in the column：{}'.format(elements, column))
        else:
            raise Exception('The value of the dictionary must be a list')
    
    def fill_na(self, numeric_fill: str = 'median', custom_fill: dict = {}):
        '''
        custom_fill: {column_1: [column_2, equal_value, sub_value1, sub_value2]}
        custom_fill是自定义的填充方式，column_1是需要填充的列，column_2是参考列，equal_value是参考列中的等值，sub_value1是等值对应的填充值，sub_value2是不等值对应的填充值
        示例：
        1.custom_fill={'语音方式': ['语音方式','VolTE', 'VOLTE', 'same']} 
        此时column1 == column2时，表示的是替换功能，将语音方式这一列中的VolTE替换为VOLTE，'same'表示不等值时不进行替换，直接填充为原值

        2.custom_fill={'是否5G网络客户':['4\\5G用户', '5G', '是', '否']}
        此时column1 != column2时，表示的是填充功能，将是否为5G网络客户这一列中，按照4\\5G这一列中的值进行填充，当4\\5G用户为5G时，是否5G网络客户填充为是，否则填充为否
        '''
        custom_columns = list(custom_fill.keys())

        for column in custom_columns:

            sub_col= custom_fill[column][0]  
            equal_val = custom_fill[column][1]
            sub_val1 = custom_fill[column][2]
            sub_val2 = custom_fill[column][3]

            if sub_col == column and sub_val2 == 'same':
                self.data[column] = self.data[sub_col].map(lambda x : sub_val1 if x == equal_val else x)
            elif sub_col == column and sub_val2 != 'same':
                self.data[column] = self.data[sub_col].map(lambda x : sub_val1 if x == equal_val else sub_val2)
            elif sub_col != column:
                self.data[column].fillna(self.data[sub_col].map(lambda x: sub_val1 if x == equal_val else sub_val2), inplace=True)

        
        #Determine which columns have missing values to fill and delete the same elements in the custom_columns list
        missing_columns = []
        for col in self.data.columns:
            if self.data[col].isnull().sum() > 0:
                missing_columns.append(col)
        
        same = set(custom_columns).intersection(set(missing_columns))
        for col in same:
            missing_columns.remove(col)
        
        #Determine the type of missing value and choose the given method to fill
        for col in missing_columns:
            if self.data[col].dtype == 'object':
                print('The type of column： [ {} ] is object'.format(col))
                self.data[col].fillna(self.data[col].mode()[0], inplace=True)

            elif self.data[col].dtype == 'int64' or self.data[col].dtype == 'float64':
                print('The type of column： [ {} ] is numeric'.format(col))
                if numeric_fill == 'median':
                    self.data[col].fillna(self.data[col].median(), inplace=True)
                elif numeric_fill == 'mean':
                    self.data[col].fillna(self.data[col].mean(), inplace=True)
                elif numeric_fill == 'mode':
                    self.data[col].fillna(self.data[col].mode()[0], inplace=True)
                else:
                    raise Exception('The method of fill_na： [ {} ] is not supported'.format(numeric_fill))

            else:
                raise Exception('The type of the column： [ {} ] is not supported'.format(self.data[col].dtype))

    #数据分箱
    def box_cut(self, cut_dic: dict = {}, cut_right=True):
        '''
        示例：
        cut_dict:{'cut_col': ['qcut', cut_ratio, cut_labels]}
        cut_col: 需要分箱的列
        cut_method: 分箱的方法，qcut和cut
        cut_ratio: qcut时，分箱的比例，cut时，分箱的个数
        cut_labels: 分箱后的标签
        '''
        for col, lst in cut_dic.items():
            cut_col = col   
            cut_method = lst[0]
            min_val = lst[1]
            max_val = lst[2]
            cut_ratio = lst[3]
            cut_labels = lst[4]

        if cut_method == 'qcut':
            self.data[cut_col] = pd.qcut(self.data[cut_col], cut_ratio, labels=cut_labels, duplicates='drop')

        elif cut_method == 'cut':
            bin_range = np.linspace(min_val, max_val, cut_ratio+1)
            self.data[cut_col] = pd.cut(self.data[cut_col], bin_range, labels= cut_labels, include_lowest=True, right=cut_right)
        else:
            raise Exception('The method of cut： [ {} ] is not supported'.format(cut_method))



    @staticmethod
    #偏态转正态，如果data_IVR[col].skew() > 0为右偏，<0为左偏，=0为正态分布
    def skew2norm(data):
        #峰度大于0，数据呈尖峰偏态分布(高度偏态)，且都为正数的数据，采用Box-Cox变换
        if data.kurtosis() > 0 and data.min() > 0:
            print('处理前的峰态系数：',data.kurtosis())
            print('处理前的偏态系数：',data.skew())
            rc_bc, bc_params = stats.boxcox(data)
            data = rc_bc
            print('处理后的峰态系数：',data.kurtosis())
            print('处理后的偏态系数：',data.skew())
            print('------------------------------------------')
        #峰度大于0，数据呈尖峰偏态分布，且有非正数据，采用对数变换
        elif data.kurtosis() > 0 and data.min() <= 0:
            print('处理前的峰态系数：',data.kurtosis())
            print('处理前的偏态系数：',data.skew())
            data = np.log10(data + 1)
            print('处理后的峰态系数：',data.kurtosis())
            print('处理后的偏态系数：',data.skew())
            print('------------------------------------------')
        #峰度小于0，数据呈平坦偏态分布（轻度偏态），且都为非负数据，采用平方根变换
        elif data.kurtosis() < 0 and data.min() >= 0:
            print('处理前的峰态系数：',data.kurtosis())
            print('处理前的偏态系数：',data.skew())
            data = np.sqrt(data)
            print('处理后的峰态系数：',data.kurtosis())
            print('处理后的偏态系数：',data.skew())
            print('------------------------------------------')
        else:
            print('不需要处理')
        
        return data

    @staticmethod
    def get_random_color():
        r1 = lambda: random.randint(0,255)
        return '#%02X%02X%02X' % (r1(),r1(),r1())
    
    @staticmethod
    def plot_scatter(data, x_col: str, y_col: str, save_path: str = None):
        plt.figure(figsize=(10,5))
        plt.scatter(data[x_col], data[y_col], color=Data_Preprocess.get_random_color())
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    @staticmethod
    def plot_heatmap(data, **kwargs):
        block = kwargs.get('block', False)
        col_num = kwargs.get('col_num', 6)
        save_path = kwargs.get('save_path', None)

        #使用淡蓝色调色板
        px.defaults.color_continuous_scale = px.colors.sequential.Blues

        #如果一次性显示整个矩阵，会看起来比较密集，因此自己截取想要的行列【0:9, 0:9】，【0:9, 9:18】......
        idx = 1
        row_num = int(data.shape[0]/col_num)+1

        if block:
            for row in range(col_num):
                for col in range(col_num):
                    fig = px.imshow(
                    data.iloc[row*row_num+1 : (row+1)*row_num, col*row_num+1 : (col+1)*row_num],
                    text_auto=True, 
                    title='相关性热力图第{}行到第{}行，第{}列到第{}列'.format(row*row_num+1, (row+1)*row_num, col*row_num+1, (col+1)*row_num), zmin=-1, zmax=1)
                    fig.show()
                    if save_path:
                        pio.write_html(fig, save_path + str(idx) + '.html')
                    idx += 1
        else:
            fig = px.imshow(data.iloc[1:, 1:], text_auto=True, title='相关性热力图', zmin=-1, zmax=1)
            fig.show()
            if save_path:
                pio.write_html(fig, save_path)

    @staticmethod
    def plot_hist(data, col, save: bool = False, target_col_lst: list = ['语音通话整体满意度', '网络覆盖与信号强度',  '语音通话清晰度', '语音通话稳定性']):
        unique_val = data[col].unique()
        count_lst = []
        for target_col in target_col_lst:
            for val in unique_val:
                count_lst.append(data[data[col] == val][target_col].value_counts())
        
            #根据unique_val的个数，自适应绘制子图
            row_num = int(len(unique_val)/6) + 1
            col_num = 6 if len(unique_val) > 6 else len(unique_val)
            fig, ax = plt.subplots(row_num, col_num, figsize=(20, 10))
            for i in range(row_num):
                for j in range(col_num):
                    if i*col_num+j < len(unique_val):
                        ax[i][j].bar(count_lst[i*col_num+j].index, count_lst[i*col_num+j].values, color=Data_Preprocess.get_random_color())
                        ax[i][j].set_title('{} = {}'.format(col, unique_val[i*col_num+j], fontsize=2))
                        #设置每个子图之间的间距
                        plt.subplots_adjust(wspace=0.5, hspace=1)

            fig.suptitle(col + '-' + target_col , fontsize=20)
            
            if save:
                save_path = './images/{}-{}.png'.format(col, target_col)
                plt.savefig(save_path)
            
            plt.show()

        
            


    #将数据转换成正态分布，查看每列的分布，并绘制每列的分布图像
    def trans2norm(self, plot: bool = False):
        for col in self.data.columns:
            if self.data[col].dtype == float or self.data[col].dtype == int:
                self.data[col] = Data_Preprocess.skew2norm(self.data[col])
                if plot:
                    plt.figure(figsize=(10,5))
                    sns.distplot(self.data[col],color=Data_Preprocess.get_random_color())
                    plt.title(col)
                    plt.show()


                
    
    def Encoding(self, col_method_dict: dict = {}):
        '''
        1.'method'：'Binary'
        示例：dict = {'col':['method', 'equal_value', 'sub_value1', 'sub_value2']}
        2.'method'：'Onehot'
        示例：dict = {'col': ['method']}
        3.'method'：'Catboost'
        示例：dict = {'col': ['method', target_col]}
        4.'method':'Target'
        示例：dict = {'col': ['method', target_col]}
        5.'method':'Mean'
        示例：dict = {'col': ['method', target_col]}

        6.'col1, col2, col3, ...': ['method']
        示例：dict = {'col1, col2, col3, ...': ['method']}
        注意，当多个列编码方式为Mean时，需要将其放在最后编码，其他编码方式放前面

        {
        '是否遇到过网络问题': ['Binary', 1, 1, 0],
        '是否4G网络客户（本地剔除物联网）,是否5G网络客户,是否实名登记用户': ['Binary', '否', 0, 1],
        '居民小区,办公室,高校,商业街,地铁,农村,高铁,其他，请注明,其他，请注明.1': ['Binary', -1, 0, 1],
        '手机没有信号,有信号无法拨通,通话过程中突然中断,通话中有杂音、听不清、断断续续,串线,通话过程中一方听不见': ['Binary', -1, 0, 1],
        '语音方式,4\\5G用户': ['Onehot'],
        '终端品牌,终端品牌类型,客户星级标识': ['Mean','语音通话整体满意度']
         }

        '''
        for col, method in col_method_dict.items():
            col_lst = col.split(',')
            print('--------------------------------------------------------')
            print('the currently encoded column is： {} '.format(col_lst))

            if len(col_lst) == 1:
                #无监督编码方式
                if method[0] == 'Binary':
                    self.data[col] = self.data[col].map(lambda x: method[2] if x == method[1] else method[3])
                    print('column：[ {} ] Binary encoding is completed'.format(col))
                elif method[0] == 'Onehot':
                    self.data = OneHotEncoder(cols=[col], use_cat_names=True).fit_transform(self.data)
                    print('column：[ {} ] Onehot encoding is completed'.format(col))
                #有监督编码方式
                elif method[0] == 'Catboost':
                    le = LabelEncoder()
                    self.data[col+'_label'] = le.fit_transform(self.data[col].astype(str))
                    catboost_encoder = CatBoostEncoder()
                    self.data[col+'_catboost'] = catboost_encoder.fit_transform(self.data[col], self.data[col+'_label'])
                    self.data.drop([col, col+'_label'], axis=1, inplace=True)
                    print('column：[ {} ] Catboost encoding is completed'.format(col))
                elif method[0] == 'Target':
                    self.data[col] = self.data[col].astype(str)
                    le = LabelEncoder()
                    self.data[col+'_label'] = le.fit_transform(self.data[col])
                    target_encoder = TargetEncoder()
                    self.data[col+'_target'] = target_encoder.fit_transform(self.data[col], self.data[col+'_label'])
                    self.data.drop([col, col+'_label'], axis=1, inplace=True)
                    print('column：[ {} ] Target encoding is completed'.format(col))
                elif method[0] == 'Mean':
                    mean_encoder = MeanEncoder(col_lst, target_type='regression')
                    self.data[col+'_mean'] = mean_encoder.fit_transform(self.data, self.data[method[1]])
                    self.data.drop([col], axis=1, inplace=True)
                    print('column：[ {} ] Mean encoding is completed'.format(col))
                else:
                    raise Exception('The method of encoding： [ {} ] is not supported'.format(method[0]))
            
            else:
                for col in col_lst:
                    if method[0] == 'Binary':
                        self.data[col] = self.data[col].map(lambda x: method[2] if x == method[1] else method[3])
                        print('column：[ {} ] Binary encoding is completed'.format(col))
                    elif method[0] == 'Onehot':
                        self.data = OneHotEncoder(cols=[col], use_cat_names=True).fit_transform(self.data)
                        print('column：[ {} ] Onehot encoding is completed'.format(col))
                        
                    elif method[0] == 'Catboost':
                        le = LabelEncoder()
                        self.data[col+'_label'] = le.fit_transform(self.data[col].astype(str))
                        catboost_encoder = CatBoostEncoder()
                        self.data[col+'_catboost'] = catboost_encoder.fit_transform(self.data[col], self.data[col+'_label'])
                        self.data.drop([col, col+'_label'], axis=1, inplace=True)
                        print('column：[ {} ] Catboost encoding is completed'.format(col))
                    elif method[0] == 'Target':
                        self.data[col] = self.data[col].astype(str)
                        le = LabelEncoder()
                        self.data[col+'_label'] = le.fit_transform(self.data[col])
                        target_encoder = TargetEncoder()
                        self.data[col+'_target'] = target_encoder.fit_transform(self.data[col], self.data[col+'_label'])
                        self.data.drop([col], axis=1, inplace=True)
                        print('column：[ {} ] Target encoding is completed'.format(col))
                    elif method[0] == 'Mean':
                        mean_encoder = MeanEncoder(col_lst, target_type='regression')
                        self.data = mean_encoder.fit_transform(self.data, self.data[method[1]])
                        self.data.drop(col_lst, axis=1, inplace=True)
                        print('column：[ {} ] Mean encoding is completed'.format(col_lst))
                        break
                    else:
                        raise Exception('The method of encoding： [ {} ] is not supported'.format(method[0]))
            print('--------------------------------------------------------')

    
    def del_collinear(self, threshold: float = 0.95, target_col: int = 1):
        # delete the high correlation features
        train = self.data.iloc[:, 5:]
        label = self.data.iloc[:, target_col]

        fs = FeatureSelector(data=train, labels=label)
        fs.identify_collinear(correlation_threshold=threshold)

        collinear_features = fs.ops['collinear']
        print('collinear features: ', collinear_features)
        train_remove = fs.remove(methods=['collinear'], keep_one_hot=True)
        self.data = pd.concat([label, train_remove], axis=1)

        print('--------------------------------------------------------')

    #实现降维
    def dim_reduction(self, method_dict: dict = {}, target_col: int = 1):
        '''
        method_dict: dict, the key is the method of dimensionality reduction, and the value is the component
        
        '''
        for method, param in method_dict.items():
            #实现PCA降维
            if method == 'PCA':
                pca = PCA(n_components=param)
                data_x = self.data.iloc[:, 5:]
                data_x = pca.fit_transform(data_x)
                data_y = self.data.iloc[:, target_col]
                data_pca = pd.concat([data_y, pd.DataFrame(data_x)], axis=1)
                print('The PCA dimensionality reduction method is completed')
                return data_pca
            #实现KPCA降维
            elif method == 'KPCA':
                kpca = KernelPCA(n_components=param, kernel='rbf')
                data_x = self.data.iloc[:, 5:]
                data_x = kpca.fit_transform(data_x)
                data_y = self.data.iloc[:, target_col]
                data_kpca = pd.concat([data_y, pd.DataFrame(data_x)], axis=1)
                print('The KPCA dimensionality reduction method is completed')
                return data_kpca
            #实现稀疏PCA降维
            elif method == 'SparsePCA':
                sparse_pca = SparsePCA(n_components=param)
                data_x = self.data.iloc[:, 5:]
                data_x = sparse_pca.fit_transform(data_x)
                data_y = self.data.iloc[:, target_col]
                data_sparse_pca = pd.concat([data_y, pd.DataFrame(data_x)], axis=1)
                print('The SparsePCA dimensionality reduction method is completed')
                return data_sparse_pca
            #实现LDA降维
            elif method == 'LDA':
                lda = LDA(n_components=param)
                data_x = self.data.iloc[:, 5:]
                data_x = lda.fit_transform(data_x, self.data.iloc[:, target_col])
                data_y = self.data.iloc[:, target_col]
                data_lda = pd.concat([data_y, pd.DataFrame(data_x)], axis=1)
                print('The LDA dimensionality reduction method is completed')
                return data_lda
            #实现SVD降维
            elif method == 'SVD':
                svd = TruncatedSVD(n_components=param)
                data_x = self.data.iloc[:, 5:]
                data_x = svd.fit_transform(data_x)
                data_y = self.data.iloc[:, target_col]
                data_svd = pd.concat([data_y, pd.DataFrame(data_x)], axis=1)
                print('The SVD dimensionality reduction method is completed')
                return data_svd
            #实现t-SNE降维
            elif method == 't-SNE':
                tsne = TSNE(n_components=param)
                data_x = self.data.iloc[:, 5:]
                data_x = tsne.fit_transform(data_x)
                data_y = self.data.iloc[:, target_col]
                data_tsne = pd.concat([data_y, pd.DataFrame(data_x)], axis=1)
                print('The t-SNE dimensionality reduction method is completed')
                return data_tsne
            #实现ISOMAP降维
            elif method == 'ISOMAP':
                isomap = Isomap(n_components=param)
                data_x = self.data.iloc[:, 5:]
                data_x = isomap.fit_transform(data_x)
                data_y = self.data.iloc[:, target_col]
                data_isomap = pd.concat([data_y, pd.DataFrame(data_x)], axis=1)
                print('The ISOMAP dimensionality reduction method is completed')
                return data_isomap
            #实现LLE降维
            elif method == 'LLE':
                lle = LocallyLinearEmbedding(n_components=param)
                data_x = self.data.iloc[:, 5:]
                data_x = lle.fit_transform(data_x)
                data_y = self.data.iloc[:, target_col]
                data_lle = pd.concat([data_y, pd.DataFrame(data_x)], axis=1)
                print('The LLE dimensionality reduction method is completed')
                return data_lle              

            else:
                raise Exception('The method of dimensionality reduction： [ {} ] is not supported'.format(method))


    
    

if __name__ == '__main__':
    data_path = './data/附件1语音业务用户满意度数据.xlsx'
    data_preprocess = Data_Preprocess(data_path)
    data_preprocess.drop_columns('其他，请注明', '其他，请注明.1', threshold=0.5)
    data_preprocess.drop_elements({'终端品牌': [0], '终端品牌类型': [0, 5320, 2015561, 2015711]})
    custom_fill = {
        '语音方式': ['语音方式','VoLTE', 'VOLTE', 'same'], 
        '是否5G网络客户':['4\\5G用户', '5G', '是', '否'],
        '是否4G网络客户（本地剔除物联网）':['4\\5G用户', '2G', '否', '是'],}
    data_preprocess.fill_na('median', custom_fill=custom_fill)
    data_preprocess.trans2norm()

    col_method_dict = {
        '是否遇到过网络问题': ['Binary', 1, 1, 0],
        '是否4G网络客户（本地剔除物联网）,是否5G网络客户,是否实名登记用户': ['Binary', '否', 0, 1],
        '居民小区,办公室,高校,商业街,地铁,农村,高铁': ['Binary', -1, 0, 1],
        '手机没有信号,有信号无法拨通,通话过程中突然中断,通话中有杂音、听不清、断断续续,串线,通话过程中一方听不见': ['Binary', -1, 0, 1],
        '语音方式,4\\5G用户': ['Onehot'],
        '终端品牌,终端品牌类型,客户星级标识': ['Catboost','语音通话整体满意度']
    }

    data_preprocess.Encoding(col_method_dict=col_method_dict)
    
    # for col in data_preprocess.data.columns:
    #     if not col == '语音通话整体满意度':
    #         data_preprocess.plot_scatter(data_preprocess.data, col, '语音通话整体满意度')
    #     else:
    #         continue

    # data_preprocess.plot_heatmap(data_preprocess.data.corr(), block = False, col_num = 6)
    # data_preprocess.del_collinear(threshold=0.95, target_col=1)
    data_preprocess.dim_reduction({'FA': [10]})


    


    # data_preprocess.data.to_csv('./data_IVR2.csv', index=False, encoding='utf-8-sig')

    # dataIVR1 = pd.read_csv('./data_IVR1.csv')
    # dataIVR2 = pd.read_csv('./data_IVR2.csv')
    # assert_frame_equal(dataIVR1, dataIVR2)




    

        
    

    
