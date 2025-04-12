import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import matplotlib
import os
import time
matplotlib.use('Agg')
start_time = time.time()
dataset='30G_data'
outpath=dataset+'_output'
if not os.path.exists(outpath):
    os.makedirs(outpath)
# 加载数据集
parquet_files = glob(dataset+'/*.parquet')
if not parquet_files:
    print("未找到任何Parquet文件，请检查路径是否正确。")
else:
    print(f"共找到 {len(parquet_files)} 个Parquet文件。")
#分快处理
processed_chunks = []
for file in parquet_files:
    print(f"正在处理文件: {file}")
    # 读取单个Parquet文件
    df1 = pd.read_parquet(file, engine='pyarrow')

    # 去重
    df1.drop_duplicates(inplace=True)
    processed_chunks.append(df1)



# 合并所有处理过的chunks，并再次去重
df = pd.concat(processed_chunks, ignore_index=True).drop_duplicates()
total_rows = df.shape[0]
print(f"数据共有 {total_rows} 行")
# dfs = [pd.read_parquet(file, engine='pyarrow') for file in parquet_files]
# df = pd.concat(dfs, ignore_index=True)
# #记录总数据
# total_rows = df.shape[0]
# print(f"数据共有 {total_rows} 行")

#3.1探索性分析
# 设置pandas的显示选项，以完整显示所有列和行
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# 打印列名
print("列名如下：")
print(df.columns.tolist())
#将前几列的数据写入csv文件中
df.head().to_csv(os.path.join(outpath,'first_five_rows.csv'), index=False,encoding='utf-8-sig')
#分析前几列数据
#打印统计摘要，这里只打印部分数据
print(df.describe())


##加载去重数据
#3.2数据预处理
# cleaned_data_file = 'cleaned_data.parquet'
# df= pd.read_parquet(cleaned_data_file, engine='pyarrow')
# total_rows=1000000000
#检查缺失值
missing_data = df.isnull().sum()
missing_ratio = (missing_data / total_rows).round(4) * 100
print("缺失值统计：\n", missing_data)
print("缺失值比例：\n", missing_ratio)

#检查异常值 IQR（Interquartile Range）方法来识别异常值
for column in ['age', 'income', 'credit_score']:  # 添加 'credit_score' 到处理列中
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"{column} 的异常值数量为: {outliers.shape[0]}")
    # 处理异常值的一种方法是将其替换为上下界值
    df[column] = df[column].clip(lower_bound, upper_bound)

# 4. 分析潜在高价值用户，并可视化结果
# 假设高价值用户的定义基于较高的收入和信用评分
# 定义高价值用户
df['is_high_value'] = (df['income'] > df['income'].quantile(0.9)) & (df['credit_score'] > df['credit_score'].quantile(0.8))

# 确认高价值用户的总数
high_value_users = df[df['is_high_value']]
print("高价值用户的总数为:", high_value_users.shape[0])

# 可视化
# 高价值用户与普通用户的年龄分布对比
plt.figure(figsize=(10, 6))
plt.hist([high_value_users['age'], df[~df['is_high_value']]['age']], bins=20, label=['High Value Users', 'Other Users'])
plt.title('Age Distribution Comparison')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend()
plt.savefig(os.path.join(outpath,'age_distribution_comparison.png'))

# 收入与信用评分的关系
plt.figure(figsize=(10, 6))
plt.scatter(high_value_users['income'], high_value_users['credit_score'], color='red', label='High Value User', alpha=0.5)
plt.scatter(df[~df['is_high_value']]['income'], df[~df['is_high_value']]['credit_score'], color='blue', label='Other User', alpha=0.5)
plt.title('Income vs Credit Score')
plt.xlabel('Income')
plt.ylabel('Credit Score')
plt.legend()
plt.savefig(os.path.join(outpath,'income_vs_credit_score.png'))



# 计算每个国家的高价值用户数量
top_countries = high_value_users.groupby('country').size().sort_values(ascending=False).head(10)


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


plt.figure(figsize=(10, 6))
top_countries.plot(kind='bar', color='steelblue')

plt.title('高价值用户最多的前10个国家')
plt.xlabel('国家')
plt.ylabel('高价值用户数量')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(outpath,'top_countries_with_high_value_users.png'))
#时间
end_time = time.time()
elapsed_time = end_time - start_time

print(f"开始时间{start_time}\n 结束时间{end_time}\n程序运行了{elapsed_time:.2f}秒")