"""A download-free PyTorch exercise. Synthetic results are not study evidence."""
from pathlib import Path
from .common import write_json


def exercise(root):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(2026)
    torch.set_num_threads(2)
    x = torch.randn(160, 4)
    y = x.argmax(dim=1)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True,
                        generator=torch.Generator().manual_seed(2026))
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.05)
    losses = []
    for epoch in range(15):
        model.train()
        total = 0.
        for inputs, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
        losses.append(total / len(dataset))
        print(f"教学 epoch={epoch+1} loss={losses[-1]:.4f}")
    model.eval()
    probe = torch.randn(12, 4)
    with torch.inference_mode():
        before = model(probe)
    folder = Path(root) / "runs" / "pytorch-learning"
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), folder / "linear.pt")
    reloaded = torch.nn.Linear(4, 4)
    reloaded.load_state_dict(torch.load(folder / "linear.pt", weights_only=True))
    reloaded.eval()
    with torch.inference_mode():
        after = reloaded(probe)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    write_json(folder / "learning.json", {"mode": "teaching_only", "losses": losses,
        "reload_logits_identical": True, "warning": "Synthetic exercise, not AG News or backdoor results."})
    print("保存与重载通过。请解释：为什么每批先清零梯度？eval() 和 inference_mode() 分别改变什么？")
