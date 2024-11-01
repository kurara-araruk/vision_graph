# transformの定義
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# test_set = datasets.CIFAR10(
#     root = "/scr/data/CIFAR10",  train = False,
#     download = True, transform = transform)

# batch_size = 1
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# net = ViG(10).to(device)
# net.load_state_dict(torch.load('/scr/vision_graph/ViG/log/20241026_140511/ViG.pth'))

# def extract_edge_index(input_data):
#     net.eval()
#     with torch.no_grad():
#         edge_index_output = net(input_data, True)
#     return edge_index_output

# # テストデータから画像データを取得して推論
# for batch_data, labels in test_loader:
#     batch_data = batch_data.to(device)
#     edge_index = extract_edge_index(batch_data)  # edge_indexを取得

#     for i, edge_idx in enumerate(edge_index):
#         print(f"Edge Index {i} shape: {edge_idx.shape}")
#         print(f"Edge Index {i}: {edge_idx}")

#     break