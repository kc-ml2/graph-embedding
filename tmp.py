from torch_geometric.datasets import PPI
import setting

dataset = PPI(root=setting.DATA_PATH )

print(len(dataset))