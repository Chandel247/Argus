import os, glob, time, random, argparse
import numpy as np, torch, torch.nn as nn, torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["gan", "l1"], required=True)
p.add_argument("--epochs", type=int, required=True)
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--lr", type=float, default=3e-4)
args = p.parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
device = "cuda"

# ---------- data ----------
BASE = "/kaggle/input/datasets/naomirory2/uieb-final"
RAW, REF = BASE + "/raw-890", BASE + "/reference-890"
SPLITS = os.path.dirname(glob.glob("/kaggle/input/**/uieb_train_fit.txt", recursive=True)[0])
read = lambda f: [l.strip() for l in open(os.path.join(SPLITS, f)) if l.strip()]

class PairDataset(Dataset):
    def __init__(self, names, train):
        self.train = train
        load = lambda d, n: np.asarray(Image.open(os.path.join(d, n)).convert("RGB").resize((256, 256), Image.BICUBIC))
        self.raw = [load(RAW, n) for n in names]
        self.ref = [load(REF, n) for n in names]
    def __len__(self):
        return len(self.raw)
    def __getitem__(self, i):
        a = torch.from_numpy(self.raw[i]).permute(2, 0, 1).float() / 127.5 - 1
        b = torch.from_numpy(self.ref[i]).permute(2, 0, 1).float() / 127.5 - 1
        if self.train and random.random() < 0.5:     # same flip on both
            a, b = a.flip(-1), b.flip(-1)
        return a, b

print("loading images ...", flush=True)
train_dl = DataLoader(PairDataset(read("uieb_train_fit.txt"), True), batch_size=8, shuffle=True, num_workers=0, drop_last=True)
val_set = PairDataset(read("uieb_val.txt"), False)
val_dl = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=0)
print("train", len(train_dl.dataset), "val", len(val_set), flush=True)

# ---------- models ----------
def down(cin, cout, bn=True):
    layers = [nn.Conv2d(cin, cout, 4, 2, 1)]
    if bn: layers.append(nn.BatchNorm2d(cout))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

def up(cin, cout):
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, 2, 1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = down(3, 32, bn=False); self.d2 = down(32, 128); self.d3 = down(128, 256)
        self.d4 = down(256, 256); self.d5 = down(256, 256)
        self.u1 = up(256, 256); self.u2 = up(512, 256); self.u3 = up(512, 128); self.u4 = up(256, 32)
        self.out = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh())
    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(d1); d3 = self.d3(d2); d4 = self.d4(d3); d5 = self.d5(d4)
        u = torch.cat([self.u1(d5), d4], 1); u = torch.cat([self.u2(u), d3], 1)
        u = torch.cat([self.u3(u), d2], 1); u = torch.cat([self.u4(u), d1], 1)
        return self.out(u)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(down(6, 32, bn=False), down(32, 64), down(64, 128), down(128, 256),
                                 nn.Conv2d(256, 1, 3, 1, 1))
    def forward(self, raw, img):
        return self.net(torch.cat([raw, img], 1))

class VGGContent(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
        self.vgg = vgg.features[:32].eval()
        for q in self.vgg.parameters(): q.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def forward(self, fake, real):
        f = lambda x: self.vgg(((x + 1) / 2 - self.mean) / self.std)
        return nn.functional.mse_loss(f(fake), f(real))

def psnr(x, y):                       # x, y in -1..1; average PSNR of the batch
    x, y = (x + 1) / 2, (y + 1) / 2
    m = ((x - y) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return (10 * torch.log10(1 / m)).sum().item()

def save(obj, path):                  # atomic: a killed session cannot corrupt the file
    torch.save(obj, path + ".tmp"); os.replace(path + ".tmp", path)

# ---------- setup ----------
gan = args.mode == "gan"
os.makedirs(args.out + "/samples", exist_ok=True)
G = Generator().to(device)
optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
if gan:
    D = Discriminator().to(device)
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    content = VGGContent().to(device)
mse, l1 = nn.MSELoss(), nn.L1Loss()

start, best = 0, -1.0
last = args.out + "/last.pt"
if os.path.exists(last):              # resume
    ck = torch.load(last, map_location=device, weights_only=False)
    G.load_state_dict(ck["G"]); optG.load_state_dict(ck["optG"])
    if gan: D.load_state_dict(ck["D"]); optD.load_state_dict(ck["optD"])
    start, best = ck["epoch"] + 1, ck["best"]
    print(f"resumed: continuing from epoch {start}", flush=True)
if start == 0:
    open(args.out + "/log.csv", "w").write("epoch,loss_g,loss_d,val_psnr,seconds\n")

# ---------- train ----------
for ep in range(start, args.epochs):
    t0 = time.time(); G.train(); sg = sd = 0.0
    for raw, ref in train_dl:
        raw, ref = raw.to(device), ref.to(device)
        fake = G(raw)
        if gan:
            optD.zero_grad()
            pr, pf = D(raw, ref), D(raw, fake.detach())
            loss_d = 0.5 * (mse(pr, torch.ones_like(pr)) + mse(pf, torch.zeros_like(pf)))
            loss_d.backward(); optD.step(); sd += loss_d.item()
            pf = D(raw, fake)
            loss_g = mse(pf, torch.ones_like(pf)) + 0.7 * l1(fake, ref) + 0.3 * content(fake, ref)
        else:
            loss_g = l1(fake, ref)
        optG.zero_grad(); loss_g.backward(); optG.step(); sg += loss_g.item()

    G.eval(); total = 0.0
    with torch.no_grad():
        for i, (raw, ref) in enumerate(val_dl):
            raw, ref = raw.to(device), ref.to(device)
            out = G(raw).clamp(-1, 1)
            total += psnr(out, ref)
            if i == 0:                # raw / output / reference, first 4 val images
                save_image(torch.cat([raw[:4], out[:4], ref[:4]]) * 0.5 + 0.5, f"{args.out}/samples/ep{ep:03d}.png", nrow=4)
    val = total / len(val_set)
    if val > best:
        best = val
        save({"G": G.state_dict(), "epoch": ep, "val_psnr": val, "mode": args.mode}, args.out + "/best.pt")
    ck = {"G": G.state_dict(), "optG": optG.state_dict(), "epoch": ep, "best": best}
    if gan: ck.update(D=D.state_dict(), optD=optD.state_dict())
    save(ck, last)
    n, secs = len(train_dl), time.time() - t0
    open(args.out + "/log.csv", "a").write(f"{ep},{sg/n:.4f},{sd/n:.4f},{val:.3f},{secs:.0f}\n")
    print(f"[{args.mode}] epoch {ep}  loss_g {sg/n:.4f}  loss_d {sd/n:.4f}  val_psnr {val:.2f}  best {best:.2f}  {secs:.0f}s", flush=True)

open(args.out + "/DONE", "w").write("done")
print("finished", flush=True)
