from gen_data import LSP_DATA
from Transformers import Compose, RandomCrop, RandomResized, TestResized


if __name__ == "__main__":
	training_dataset_path = 'lspet_dataset'
	val_data_path = 'lsp_dataset'

	# training_dataset_path = 'C:\Users\Wei Li\OneDrive\桌面\final_cv\CPM\preprocess\lspet_dataset'
	# val_data_path = 'C:\Users\Wei Li\OneDrive\桌面\final_cv\CPM\preprocess\lsp_dataset'


	# epoch = 0
	# while epoch < 20:
	# 	print('epoch ', epoch)
	# 	"""--------Train--------"""
	# 	# Training data
	data = LSP_DATA('lspet', training_dataset_path, 8, Compose([RandomResized(), RandomCrop(368)]))
	# data = LSP_DATA()
	# train_loader = torch.utils.data.dataloader.DataLoader(data, batch_size=8)
	for j, d in enumerate(data):
		inputs, heatmap, centermap = d
		print("inputs: ", inputs)
		print("heatmap: ", heatmap)
		print("centermap: ", centermap)


	print("data: ", data)
