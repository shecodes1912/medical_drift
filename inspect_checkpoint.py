import torch

path = r"third_party/PraNet/snapshots/PraNet_Res2Net/PraNet-19.pth"
state = torch.load(path, map_location="cpu", weights_only=True)

print("Checkpoint type:", type(state).__name__)
if isinstance(state, dict):
    print("Number of entries:", len(state))
    print("First 20 keys:")
    for key in list(state.keys())[:20]:
        print(" ", key)
