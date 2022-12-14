import seim
from seim import prediction, analyses
import pandas as pd

data_dir = "/mnt/c/Users/ebobrow/Downloads/data/"

node_path = f"{data_dir}shapefile/node_data.shp"
pair_path = f"{data_dir}pair_data.csv"
temp_data_dir = "/home/ebobrow/data/"

node_data = pd.read_csv(f"{temp_data_dir}/node.csv")
pair_data = pd.read_csv(f"{pair_path}")
# weights_matrix = np.loadtxt(f"{temp_data_dir}weights.txt")

print(node_data.columns)
print(pair_data.columns)

exit

# node_data = node_data[node_data["ID"] < 200]
pair_data = pair_data[(pair_data["ID_ORIG"] < 200) & (pair_data["ID_DEST"] < 200)]

# node_data = node_data.to_csv(f"{temp_data_dir}/node.csv")
pair_data = pair_data.to_csv(f"{temp_data_dir}/pair.csv", index=False)

# node_path = f"{temp_data_dir}/node.csv"
pair_path = f"{temp_data_dir}/pair.csv"

flows_MSP = prediction.multi_stage_predict(prediction.tc, prediction.tc, "trip_counts", node_path, pair_path, node_path, pair_path, temp_data_dir)

print(flows_MSP)

# flows_tc = prediction.tc(coef ,node_data, pair_data, weights_matrix)