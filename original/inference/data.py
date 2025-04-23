data = pd.read_csv(csv_path)

# 添加 'Correction' 列，判断 TC 和 MC 是否相等
data['Correction'] = (data['TC'] == data['MC']).astype(int)

# 计算 Correction 的总和
m = data['Correction'].sum()

# 总行数
total_rows = len(data)

# 计算 MC 相对于 TC 的比例
less_than = (data['MC'] < data['TC']).sum() / total_rows*100 # MC 小于 TC 的比例
greater_than = (data['MC'] > data['TC']).sum() / total_rows*100  # MC 大于 TC 的比例
equal_to = (data['MC'] == data['TC']).sum() / total_rows*100  # MC 等于 TC 的比例

# 计算 MC < TC 时，MC 总数 / TC 总数的比例
less_than_ratio = data.loc[data['MC'] < data['TC'], 'MC'].sum() / data.loc[data['MC'] < data['TC'], 'TC'].sum()*100

# 计算 MC 不等于 0 时，TC == MC 的比例
mc_not_zero = data[data['MC'] != 0]*100
tc_equal_mc_ratio = (mc_not_zero['TC'] == mc_not_zero['MC']).sum() / len(mc_not_zero)*100

# 计算 MC 的总数 / TC 的总数
mc_to_tc_ratio = (data['MC'].sum()-data['TC'].sum()) / data['TC'].sum()*100

# 打印结果
print(f"Correction 总和: {m}")
print(f"MC 小于 TC 的比例: {less_than:.4f}")
print(f"MC 大于 TC 的比例: {greater_than:.4f}")
print(f"MC 等于 TC 的比例: {equal_to:.4f}")
print(f"当 MC 小于 TC 时，MC 总数 / TC 总数的比例: {less_than_ratio:.4f}")
print(f"当 MC 不等于 0 时，TC == MC 的比例: {tc_equal_mc_ratio:.4f}")
print(f"误差率: {mc_to_tc_ratio:.4f}")

