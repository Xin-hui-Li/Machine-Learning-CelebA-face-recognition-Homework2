import os
import math
import random
import numpy as np
import pandas as pd
import warnings
import torch
import torchvision as tv
import matplotlib
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve, auc as sk_auc

matplotlib.use('Agg')
matplotlib.rcParams['figure.dpi'] = 120

SEED = 42


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Config:
    def __init__(self):
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.DATA_ROOT = "data"
        self.CELEBA_DIR = os.path.join(self.DATA_ROOT, "celeba")
        self.IMG_DIR = os.path.join(self.CELEBA_DIR, "img_align_celeba/img_align_celeba")
        self.ATTR_FILE = os.path.join(self.CELEBA_DIR, "list_attr_celeba.csv")
        self.SPLIT_FILE = os.path.join(self.CELEBA_DIR, "list_eval_partition.csv")

        self.IMG_SIZE = 112
        self.BATCH = 64
        self.NUM_WORKERS = 0

        self.PCA_N_TRAIN = 6000
        self.PCA_N_VALID = 2000
        self.DL_N_TRAIN = 10000
        self.DL_N_VALID = 2000
        self.DL_N_TEST = 2000

        self.FREEZE_EPOCHS = 2
        self.FINETUNE_EPOCHS = 4
        self.LR_FREEZE = 3e-4
        self.LR_FINETUNE = 1e-4
        self.WD = 1e-4


def assert_exists(p, hint):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing: {p}\nHint: {hint}")


def get_transforms(img_size):
    return transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])


def evaluate_binary(y_true, prob, thr=0.5):
    pred = (prob >= thr).astype(np.int32)
    return dict(
        auroc=float(roc_auc_score(y_true, prob)),
        f1=float(f1_score(y_true, pred)),
        acc=float(accuracy_score(y_true, pred))
    )


def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)


def save_misclassified_grid(model, dl, which=0, out='outputs/miscls.png', limit=1000, device='cpu'):
    model.eval()
    imgs = []
    miscls_records = []

    with torch.no_grad():
        for batch_idx, (x, attrs, y) in enumerate(dl):  
            x, attrs, y = x.to(device), attrs.to(device), y.to(device)
            # 根据模型类型调整调用方式
            if hasattr(model, 'forward') and len(model.forward.__code__.co_varnames) > 2:
                # 模型接受额外特征参数
                p = torch.sigmoid(model(x, attrs))
            else:
                # 模型只接受图像参数
                p = torch.sigmoid(model(x))
            pred = (p[:, which] >= 0.5).float()
            mask = (pred != y[:, which]).cpu()
            mismatch = x.cpu()[mask]

            if mask.any():
                batch_indices = dl.dataset.indices[batch_idx * dl.batch_size: (batch_idx + 1) * dl.batch_size] \
                    if hasattr(dl.dataset, 'indices') else list(
                    range(batch_idx * dl.batch_size, min((batch_idx + 1) * dl.batch_size, len(dl.dataset))))

                for idx, m in enumerate(mask):
                    if m:
                        data_idx = batch_indices[idx] if idx < len(batch_indices) else 0
                        fname = dl.dataset.base.dataset.items[data_idx][0] if hasattr(dl.dataset.base.dataset,
                                                                                      'items') else f'image_{data_idx}.jpg'
                        true_label = y[idx, which].item()
                        pred_prob = p[idx, which].item()
                        pred_label = pred[idx].item()

                        miscls_records.append({
                            'filename': fname,
                            'true_label': true_label,
                            'predicted_probability': pred_prob,
                            'predicted_label': pred_label
                        })

                        if len(imgs) < limit:
                            imgs.append(x[idx].cpu())


    csv_out = out.replace('.png', '.csv')
    if len(miscls_records) == 0:
        print("No misclassifications collected.")
        pd.DataFrame(columns=['filename', 'true_label', 'predicted_probability', 'predicted_label']).to_csv(csv_out,
                                                                                                           index=False)
        print(f"Empty misclassification log saved to '{csv_out}'")
    else:
        df_miscls = pd.DataFrame(miscls_records)
        df_miscls.to_csv(csv_out, index=False)
        print(f"Misclassification log saved to '{csv_out}' (total {len(miscls_records)} records)")


class CelebAFallback(Dataset):
    def __init__(self, img_dir, attr_file, split_file, split, transform=None, target_attrs=None, extra_attrs=None):
        self.items = []
        self.target_attrs = target_attrs if target_attrs else ['Smiling', 'Eyeglasses']
        self.extra_attrs = extra_attrs if extra_attrs else []
        self.all_attrs = self.target_attrs + self.extra_attrs

        with open(attr_file, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines()]

        if ',' in lines[0]:
            attr_names = lines[0].split(',')
            data_start_idx = 1
            n = len(lines) - 1
        else:
            n = int(lines[0])
            attr_names = lines[1].split()
            data_start_idx = 2

        data = []
        total_lines = min(data_start_idx + n, len(lines))
        for i in range(data_start_idx, min(data_start_idx + n, len(lines))):
            if (i - data_start_idx) % 1000 == 0 or i == total_lines - 1:
                progress = (i - data_start_idx + 1) / (total_lines - data_start_idx) * 100
                print(f"  加载属性数据: {i - data_start_idx + 1}/{total_lines - data_start_idx} ({progress:.1f}%)")

            if ',' in lines[i]:
                parts = lines[i].split(',')
            else:
                parts = lines[i].split()
            fname = parts[0]
            vals = [int(v) for v in parts[1:] if v.strip()]
            data.append((fname, vals))

        split_map = {}
        with open(split_file, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines()]

        start_idx = 0
        if lines and ('image_id' in lines[0].lower() or ',' in lines[0]):
            start_idx = 1

        total_split_lines = len(lines[start_idx:])
        for idx, line in enumerate(lines[start_idx:]):
            if idx % 1000 == 0 or idx == total_split_lines - 1:
                progress = (idx + 1) / total_split_lines * 100
                print(f"  加载分割数据: {idx + 1}/{total_split_lines} ({progress:.1f}%)")

            if ',' in line:
                parts = line.split(',')
            else:
                parts = line.strip().split()

            if len(parts) >= 2:
                fname = parts[0]
                sp = parts[-1]
                try:
                    split_map[fname] = int(sp)
                except ValueError:
                    continue

        sp_code = {'train': 0, 'valid': 1, 'test': 2}[split]
        items = [(fname, vals) for (fname, vals) in data if split_map.get(fname, -1) == sp_code]


        self.attr_names = attr_names
        self.target_indices = [attr_names.index(a) for a in self.target_attrs]
        self.extra_indices = [attr_names.index(a) for a in self.extra_attrs]

        self.items = items
        self.transform = transform
        self.img_dir = img_dir

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fname, vals = self.items[i]
        path = os.path.join(self.img_dir, fname)
        from PIL import Image
        im = Image.open(path).convert('RGB')
        if self.transform:
            im = self.transform(im)


        valid_target_indices = [j for j in self.target_indices if j < len(vals)]
        targets = np.zeros(len(self.target_attrs), dtype=np.float32)
        for idx, j in enumerate(valid_target_indices):
            if idx < len(targets):  # 额外安全检查
                targets[idx] = 1.0 if vals[j] > 0 else 0.0


        valid_extra_indices = [j for j in self.extra_indices if j < len(vals)]
        extras = np.zeros(len(self.extra_attrs), dtype=np.float32)
        for idx, j in enumerate(valid_extra_indices):
            if idx < len(extras):  # 额外安全检查
                extras[idx] = 1.0 if vals[j] > 0 else 0.0

        return im, torch.from_numpy(extras), torch.from_numpy(targets)



def load_data(config, target_attrs=None, extra_attrs=None):
    tfm = get_transforms(config.IMG_SIZE)
    target_attrs = target_attrs if target_attrs else ['Smiling', 'Eyeglasses']
    try:
        train = tv.datasets.CelebA(root=config.DATA_ROOT, split='train', target_type='attr', transform=tfm,
                                   download=False)
        valid = tv.datasets.CelebA(root=config.DATA_ROOT, split='valid', target_type='attr', transform=tfm,
                                   download=False)
        test = tv.datasets.CelebA(root=config.DATA_ROOT, split='test', target_type='attr', transform=tfm,
                                  download=False)
        _ = train[0]
        ATTRS = train.attr_names
        IDX_SMILE = ATTRS.index('Smiling')
        IDX_EYES = ATTRS.index('Eyeglasses')
        EXTRA_IDXS = [ATTRS.index(a) for a in extra_attrs] if extra_attrs else []
        print('Using torchvision CelebA (local).')
        return train, valid, test, ATTRS, IDX_SMILE, IDX_EYES, EXTRA_IDXS, True
    except Exception as e:
        warnings.warn('torchvision CelebA failed, switching to fallback loader. Reason: ' + str(e))
        tr = CelebAFallback(config.IMG_DIR, config.ATTR_FILE, config.SPLIT_FILE, 'train', transform=tfm,
                            target_attrs=target_attrs, extra_attrs=extra_attrs)
        va = CelebAFallback(config.IMG_DIR, config.ATTR_FILE, config.SPLIT_FILE, 'valid', transform=tfm,
                            target_attrs=target_attrs, extra_attrs=extra_attrs)
        te = CelebAFallback(config.IMG_DIR, config.ATTR_FILE, config.SPLIT_FILE, 'test', transform=tfm,
                            target_attrs=target_attrs, extra_attrs=extra_attrs)
        ATTRS = target_attrs + (extra_attrs if extra_attrs else [])
        IDX_SMILE, IDX_EYES = 0, 1
        EXTRA_IDXS = list(range(2, 2 + len(extra_attrs))) if extra_attrs else []
        print('Using fallback loader.')
        return tr, va, te, ATTRS, IDX_SMILE, IDX_EYES, EXTRA_IDXS, False


def rand_subset(base, n, seed=SEED):
    idx = np.random.default_rng(seed).choice(len(base), size=min(n, len(base)), replace=False)
    return Subset(base, idx)


def to_np(ds, attr_idx, n_max):
    X, Y = [], []
    m = min(n_max, len(ds))
    for i in range(m):
        # 根据数据集类型调整获取方式
        if isinstance(ds[i], tuple) and len(ds[i]) == 3:  # 包含额外特征的情况
            img, _, attr = ds[i]
        else:  # 原始情况
            img, attr = ds[i]
        X.append(img.view(-1).numpy().astype('float32'))
        Y.append(int(attr[attr_idx].item()) if torch.is_tensor(attr) else int(attr[attr_idx]))
    return np.stack(X, 0), np.array(Y, dtype=np.int64)



class Wrap(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        return self.base[i]


# 定义带额外特征的ResNet模型
class ResNetWithFeatures(nn.Module):
    def __init__(self, num_extra_features, num_outputs=2):
        super().__init__()
        self.resnet = resnet18(weights='IMAGENET1K_V1')
        self.resnet_fc_in = self.resnet.fc.in_features

        # 移除原始fc层
        self.resnet.fc = nn.Identity()

        # 额外特征处理层
        self.extra_features_layer = nn.Sequential(
            nn.Linear(num_extra_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 合并特征后的分类器
        combined_features = self.resnet_fc_in + 32
        self.classifier = nn.Linear(combined_features, num_outputs)

    def forward(self, x, extra_features):
        img_features = self.resnet(x)
        extra_features_processed = self.extra_features_layer(extra_features)
        combined = torch.cat([img_features, extra_features_processed], dim=1)
        return self.classifier(combined)



def train_evaluate_resnet18(config, train, valid, test, use_extra_features=False, extra_feature_count=0):
    tr_ds = Wrap(rand_subset(train, config.DL_N_TRAIN))
    va_ds = Wrap(rand_subset(valid, config.DL_N_VALID))
    te_ds = Wrap(rand_subset(test, config.DL_N_TEST))

    Ltr = DataLoader(tr_ds, batch_size=config.BATCH, shuffle=True, num_workers=config.NUM_WORKERS)
    Lva = DataLoader(va_ds, batch_size=config.BATCH * 2, shuffle=False, num_workers=config.NUM_WORKERS)
    Lte = DataLoader(te_ds, batch_size=config.BATCH * 2, shuffle=False, num_workers=config.NUM_WORKERS)


    if use_extra_features and extra_feature_count > 0:
        model = ResNetWithFeatures(extra_feature_count)
    else:
        model = resnet18(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(config.DEVICE)

    lossf = nn.BCEWithLogitsLoss()


    training_history = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'val_auroc_smile': [],
        'val_auroc_eyes': []
    }

    def epoch(dl, train_flag, opt=None):
        model.train(train_flag)
        all_p, all_y, tot = [], [], 0.0
        for batch in dl:
            if len(batch) == 3:  # 图像, 额外特征, 目标
                x, attrs, y = batch
                x, attrs, y = x.to(config.DEVICE), attrs.to(config.DEVICE), y.to(config.DEVICE)
            else:  # 图像, 目标
                x, y = batch
                x, y = x.to(config.DEVICE), y.to(config.DEVICE)
                attrs = None

            with torch.set_grad_enabled(train_flag):
                if use_extra_features and attrs is not None:
                    logits = model(x, attrs)
                else:
                    logits = model(x)
                loss = lossf(logits, y)

            if train_flag:
                opt.zero_grad()
                loss.backward()
                opt.step()

            tot += float(loss.item()) * x.size(0)
            all_p.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_y.append(y.detach().cpu().numpy())

        P = np.concatenate(all_p, 0)
        Y = np.concatenate(all_y, 0)
        M = {}
        for i, name in enumerate(['Smiling', 'Eyeglasses']):
            pred = (P[:, i] >= 0.5).astype(int)
            M[name] = dict(
                auroc=float(roc_auc_score(Y[:, i], P[:, i])),
                f1=float(f1_score(Y[:, i], pred)),
                acc=float(accuracy_score(Y[:, i], pred))
            )
        return tot / len(dl.dataset), P, Y, M

    # 冻结训练
    for m in [model.resnet.layer1, model.resnet.layer2, model.resnet.layer3,
              model.resnet.layer4] if use_extra_features else \
            [model.layer1, model.layer2, model.layer3, model.layer4]:
        for p in m.parameters():
            p.requires_grad = False

    if use_extra_features:
        # 只训练额外特征层和分类器
        trainable_params = list(model.extra_features_layer.parameters()) + list(model.classifier.parameters())
    else:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    opt = torch.optim.AdamW(
        trainable_params,
        lr=config.LR_FREEZE,
        weight_decay=config.WD
    )

    print('Stage-1 freeze')
    for e in range(config.FREEZE_EPOCHS):
        tr_loss, _, _, _ = epoch(Ltr, True, opt)
        va_loss, Pv, Yv, Mv = epoch(Lva, False)
        print(
            f'  {e + 1}/{config.FREEZE_EPOCHS} train={tr_loss:.4f} val={va_loss:.4f} AUROC(smile)={Mv['Smiling']['auroc']:.3f}')
        # 记录训练历史
        training_history['epochs'].append(f'freeze_{e + 1}')
        training_history['train_loss'].append(tr_loss)
        training_history['val_loss'].append(va_loss)
        training_history['val_auroc_smile'].append(Mv['Smiling']['auroc'])
        training_history['val_auroc_eyes'].append(Mv['Eyeglasses']['auroc'])

    # 微调
    for p in model.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR_FINETUNE,
        weight_decay=config.WD
    )

    print('Stage-2 finetune')
    for e in range(config.FINETUNE_EPOCHS):
        tr_loss, _, _, _ = epoch(Ltr, True, opt)
        va_loss, Pv, Yv, Mv = epoch(Lva, False)
        print(
            f'  {e + 1}/{config.FINETUNE_EPOCHS} train={tr_loss:.4f} val={va_loss:.4f} AUROC(smile)={Mv['Smiling']['auroc']:.3f}')
        # 记录训练历史
        training_history['epochs'].append(f'finetune_{e + 1}')
        training_history['train_loss'].append(tr_loss)
        training_history['val_loss'].append(va_loss)
        training_history['val_auroc_smile'].append(Mv['Smiling']['auroc'])
        training_history['val_auroc_eyes'].append(Mv['Eyeglasses']['auroc'])

    # 测试
    te_loss, Pt, Yt, Mt = epoch(Lte, False)

    model_name = f"ResNet18+Features" if use_extra_features else "ResNet18"
    df_deep = pd.DataFrame([
        [model_name, 'Smiling', 'test', Mt['Smiling']['auroc'], Mt['Smiling']['f1'], Mt['Smiling']['acc']],
        [model_name, 'Eyeglasses', 'test', Mt['Eyeglasses']['auroc'], Mt['Eyeglasses']['f1'], Mt['Eyeglasses']['acc']],
    ], columns=['Model', 'Task', 'Split', 'AUROC', 'F1', 'ACC'])

    # 保存详细的训练历史
    history_df = pd.DataFrame(training_history)
    history_file = f'outputs/training_history_{"with_features" if use_extra_features else "base"}.csv'
    history_df.to_csv(history_file, index=False)
    print(f"Training history saved to '{history_file}'")

    # 保存详细的测试结果，包括所有评估指标
    test_results = []
    for attr_idx, attr_name in enumerate(['Smiling', 'Eyeglasses']):
        # 计算更多评估指标
        pred = (Pt[:, attr_idx] >= 0.5).astype(int)
        true = Yt[:, attr_idx]

        from sklearn.metrics import precision_score, recall_score, confusion_matrix
        precision = precision_score(true, pred)
        recall = recall_score(true, pred)
        tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        test_results.append({
            'model': model_name,
            'attribute': attr_name,
            'auroc': Mt[attr_name]['auroc'],
            'f1': Mt[attr_name]['f1'],
            'accuracy': Mt[attr_name]['acc'],
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'true_positive': tp,
            'false_positive': fp,
            'true_negative': tn,
            'false_negative': fn
        })

    test_results_df = pd.DataFrame(test_results)
    test_results_file = f'outputs/detailed_test_results_{"with_features" if use_extra_features else "base"}.csv'
    test_results_df.to_csv(test_results_file, index=False)
    print(f"Detailed test results saved to '{test_results_file}'")

    # 保存预测概率和真实标签，便于后续分析
    predictions_data = {
        'true_smiling': Yt[:, 0],
        'pred_prob_smiling': Pt[:, 0],
        'true_eyeglasses': Yt[:, 1],
        'pred_prob_eyeglasses': Pt[:, 1]
    }
    predictions_df = pd.DataFrame(predictions_data)
    predictions_file = f'outputs/predictions_{"with_features" if use_extra_features else "base"}.csv'
    predictions_df.to_csv(predictions_file, index=False)
    print(f"Predictions saved to '{predictions_file}'")

    return model, Ltr, Lva, Lte, Pv, Yv, df_deep, Pt, Yt, Mt


def main():
    set_seeds()
    config = Config()
    os.makedirs("outputs", exist_ok=True)

    # 验证文件存在
    assert_exists(config.CELEBA_DIR, "Create this folder and put CelebA files inside.")
    assert_exists(config.IMG_DIR, "Put the extracted 'img_align_celeba' folder here.")
    assert_exists(config.ATTR_FILE, "Place 'list_attr_celeba.txt' here.")
    assert_exists(config.SPLIT_FILE, "Place 'list_eval_partition.txt' here.")

    num_imgs = len([f for f in os.listdir(config.IMG_DIR) if f.lower().endswith('.jpg')])
    print(" Found directory:", config.CELEBA_DIR)
    print(" Images:", num_imgs, " (OK if > 1000; full set ~200k)")
    print(" Attr file:", config.ATTR_FILE)
    print(" Split file:", config.SPLIT_FILE)

    # 定义要评估的额外特征
    attributes_to_evaluate = ['Male', 'Young', 'Attractive', 'Bald', 'Heavy_Makeup',
                              'Mouth_Slightly_Open', 'Narrow_Eyes', 'Wearing_Hat']
    target_attrs = ['Smiling', 'Eyeglasses']

    # 1. 加载不带额外特征的数据并训练基准模型
    print("\nLoading base data (without extra features)...")
    train_base, valid_base, test_base, ATTRS_BASE, IDX_SMILE, IDX_EYES, _, _ = load_data(
        config, target_attrs=target_attrs, extra_attrs=None)

    # 2. 加载带额外特征的数据
    print("\nLoading data with extra features...")
    train_extra, valid_extra, test_extra, ATTRS_EXTRA, _, _, _, _ = load_data(
        config, target_attrs=target_attrs, extra_attrs=attributes_to_evaluate)

    print(f"Dataset sizes (base): train={len(train_base)}, valid={len(valid_base)}, test={len(test_base)}")
    print(
        f"Dataset sizes (with extra features): train={len(train_extra)}, valid={len(valid_extra)}, test={len(test_extra)}")
    print(f"Target attributes: {target_attrs}")
    print(f"Extra attributes to evaluate: {attributes_to_evaluate}")

    # 3. 传统基线：PCA + Logistic Regression (仅使用图像)
    print("\nRunning PCA + Logistic Regression (baseline)...")
    tr_small = rand_subset(train_base, config.PCA_N_TRAIN)
    va_small = rand_subset(valid_base, config.PCA_N_VALID)

    print("  Processing 'Smiling' attribute...")
    Xtr, ytr = to_np(tr_small, IDX_SMILE, config.PCA_N_TRAIN)
    Xva, yva = to_np(va_small, IDX_SMILE, config.PCA_N_VALID)

    pca = PCA(n_components=200, whiten=True, random_state=SEED).fit(Xtr)
    Ztr, Zva = pca.transform(Xtr), pca.transform(Xva)

    lr = LogisticRegression(max_iter=2000, n_jobs=-1).fit(Ztr, ytr)
    proba_smile_val = lr.predict_proba(Zva)[:, 1]
    metrics_lr_smile = evaluate_binary(yva, proba_smile_val)

    print("  Processing 'Eyeglasses' attribute...")
    Xtr2, ytr2 = to_np(tr_small, IDX_EYES, config.PCA_N_TRAIN)
    Xva2, yva2 = to_np(va_small, IDX_EYES, config.PCA_N_VALID)
    Ztr2, Zva2 = pca.transform(Xtr2), pca.transform(Xva2)

    lr2 = LogisticRegression(max_iter=2000, n_jobs=-1).fit(Ztr2, ytr2)
    proba_eyes_val = lr2.predict_proba(Zva2)[:, 1]
    metrics_lr_eyes = evaluate_binary(yva2, proba_eyes_val)

    df_lr = pd.DataFrame([
        ['PCA+LR', 'Smiling', 'valid', metrics_lr_smile['auroc'], metrics_lr_smile['f1'], metrics_lr_smile['acc']],
        ['PCA+LR', 'Eyeglasses', 'valid', metrics_lr_eyes['auroc'], metrics_lr_eyes['f1'], metrics_lr_eyes['acc']],
    ], columns=['Model', 'Task', 'Split', 'AUROC', 'F1', 'ACC'])

    print("PCA+LR results:")
    print(df_lr)

    # 保存PCA+LR的详细结果
    lr_results = []
    for attr_name, y_true, y_pred_prob in [
        ('Smiling', yva, proba_smile_val),
        ('Eyeglasses', yva2, proba_eyes_val)
    ]:
        y_pred = (y_pred_prob >= 0.5).astype(int)
        from sklearn.metrics import precision_score, recall_score, confusion_matrix
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        lr_results.append({
            'model': 'PCA+LR',
            'attribute': attr_name,
            'auroc': roc_auc_score(y_true, y_pred_prob),
            'f1': f1_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'true_positive': tp,
            'false_positive': fp,
            'true_negative': tn,
            'false_negative': fn
        })

    lr_results_df = pd.DataFrame(lr_results)
    lr_results_df.to_csv('outputs/detailed_lr_results.csv', index=False)
    print(f"Detailed PCA+LR results saved to 'outputs/detailed_lr_results.csv'")

    # 4. 训练基准ResNet18 (不带额外特征)
    print("\nRunning baseline ResNet18 (without extra features)...")
    model_base, Ltr_base, Lva_base, Lte_base, Pv_base, Yv_base, df_deep_base, Pt_base, Yt_base, Mt_base = train_evaluate_resnet18(
        config, train_base, valid_base, test_base, use_extra_features=False)

    print("Baseline ResNet18 results:")
    print(df_deep_base)

    # 5. 训练带额外特征的ResNet18
    print("\nRunning ResNet18 with extra features...")
    model_extra, Ltr_extra, Lva_extra, Lte_extra, Pv_extra, Yv_extra, df_deep_extra, Pt_extra, Yt_extra, Mt_extra = train_evaluate_resnet18(
        config, train_extra, valid_extra, test_extra,
        use_extra_features=True,
        extra_feature_count=len(attributes_to_evaluate))

    print("ResNet18 with extra features results:")
    print(df_deep_extra)

    # 合并结果并保存
    results = pd.concat([df_lr, df_deep_base, df_deep_extra], ignore_index=True)
    results.to_csv('outputs/results_with_features.csv', index=False)
    print("\nCombined results saved to 'outputs/results_with_features.csv'")
    print(results)

    # 绘制额外的对比图表
    print("\nGenerating additional comparison charts...")

    # 模型性能对比图
    models = ['PCA+LR', 'ResNet18', 'ResNet18+Features']
    smiling_auroc = [
        lr_results_df[lr_results_df['attribute'] == 'Smiling']['auroc'].values[0],
        df_deep_base[df_deep_base['Task'] == 'Smiling']['AUROC'].values[0],
        df_deep_extra[df_deep_extra['Task'] == 'Smiling']['AUROC'].values[0]
    ]
    eyes_auroc = [
        lr_results_df[lr_results_df['attribute'] == 'Eyeglasses']['auroc'].values[0],
        df_deep_base[df_deep_base['Task'] == 'Eyeglasses']['AUROC'].values[0],
        df_deep_extra[df_deep_extra['Task'] == 'Eyeglasses']['AUROC'].values[0]
    ]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.35

    plt.bar(x - width / 2, smiling_auroc, width, label='Smiling')
    plt.bar(x + width / 2, eyes_auroc, width, label='Eyeglasses')
    plt.xlabel('Model')
    plt.ylabel('AUROC')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models)
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/model_comparison.png', dpi=200)
    plt.close()
    print("Model comparison chart saved.")

    # 绘制ROC曲线对比
    print("\nGenerating ROC curves...")
    # 微笑属性ROC对比
    fpr1, tpr1, _ = roc_curve(yva, proba_smile_val)
    auc1 = sk_auc(fpr1, tpr1)
    fpr2, tpr2, _ = roc_curve(Yv_base[:, 0], Pv_base[:, 0])
    auc2 = sk_auc(fpr2, tpr2)
    fpr3, tpr3, _ = roc_curve(Yv_extra[:, 0], Pv_extra[:, 0])
    auc3 = sk_auc(fpr3, tpr3)

    plt.figure()
    plt.plot(fpr1, tpr1, label=f'PCA+LR AUC={auc1:.3f}')
    plt.plot(fpr2, tpr2, label=f'ResNet18 AUC={auc2:.3f}')
    plt.plot(fpr3, tpr3, label=f'ResNet18+Features AUC={auc3:.3f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC on Smiling (Valid)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('outputs/roc_smiling_comparison.png', dpi=200)
    plt.close()

    # 眼镜属性ROC对比
    fpr1e, tpr1e, _ = roc_curve(yva2, proba_eyes_val)
    auc1e = sk_auc(fpr1e, tpr1e)
    fpr2e, tpr2e, _ = roc_curve(Yv_base[:, 1], Pv_base[:, 1])
    auc2e = sk_auc(fpr2e, tpr2e)
    fpr3e, tpr3e, _ = roc_curve(Yv_extra[:, 1], Pv_extra[:, 1])
    auc3e = sk_auc(fpr3e, tpr3e)

    plt.figure()
    plt.plot(fpr1e, tpr1e, label=f'PCA+LR AUC={auc1e:.3f}')
    plt.plot(fpr2e, tpr2e, label=f'ResNet18 AUC={auc2e:.3f}')
    plt.plot(fpr3e, tpr3e, label=f'ResNet18+Features AUC={auc3e:.3f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC on Eyeglasses (Valid)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('outputs/roc_eyeglasses_comparison.png', dpi=200)
    plt.close()

    print("ROC curves saved.")

    # 保存错误分类的图像
    print("\nSaving misclassified images...")
    save_misclassified_grid(model_base, Lte_base, which=0, out='outputs/miscls_base_smiling.png', limit=1000,
                            device=config.DEVICE)
    save_misclassified_grid(model_base, Lte_base, which=1, out='outputs/miscls_base_eyeglasses.png', limit=1000,
                            device=config.DEVICE)
    save_misclassified_grid(model_extra, Lte_extra, which=0, out='outputs/miscls_extra_smiling.png', limit=1000,
                            device=config.DEVICE)
    save_misclassified_grid(model_extra, Lte_extra, which=1, out='outputs/miscls_extra_eyeglasses.png', limit=1000,
                            device=config.DEVICE)
    print("Misclassified images saved.")


    print("\nGenerating final results summary...")
    summary_data = []

    # 添加PCA+LR结果
    for _, row in lr_results_df.iterrows():
        summary_data.append({
            'Model': 'PCA+LR',
            'Attribute': row['attribute'],
            'AUROC': row['auroc'],
            'F1': row['f1'],
            'Accuracy': row['accuracy'],
            'Precision': row['precision'],
            'Recall': row['recall'],
            'Specificity': row['specificity']
        })

    # 添加基础ResNet18结果
    for _, row in df_deep_base.iterrows():
        attr_name = row['Task']
        detailed_row = pd.read_csv('outputs/detailed_test_results_base.csv')
        detailed_row = detailed_row[detailed_row['attribute'] == attr_name].iloc[0]
        summary_data.append({
            'Model': 'ResNet18',
            'Attribute': attr_name,
            'AUROC': row['AUROC'],
            'F1': row['F1'],
            'Accuracy': row['ACC'],
            'Precision': detailed_row['precision'],
            'Recall': detailed_row['recall'],
            'Specificity': detailed_row['specificity']
        })

    # 添加带额外特征的ResNet18结果
    for _, row in df_deep_extra.iterrows():
        attr_name = row['Task']
        detailed_row = pd.read_csv('outputs/detailed_test_results_with_features.csv')
        detailed_row = detailed_row[detailed_row['attribute'] == attr_name].iloc[0]
        summary_data.append({
            'Model': 'ResNet18+Features',
            'Attribute': attr_name,
            'AUROC': row['AUROC'],
            'F1': row['F1'],
            'Accuracy': row['ACC'],
            'Precision': detailed_row['precision'],
            'Recall': detailed_row['recall'],
            'Specificity': detailed_row['specificity']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('outputs/results_summary.csv', index=False)
    print("Results summary saved to 'outputs/results_summary.csv'")


if __name__ == "__main__":
    import datetime

    start_time = datetime.datetime.now()
    print(f"程序开始执行时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    main()

    end_time = datetime.datetime.now()
    print(f"程序结束执行时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    execution_time = end_time - start_time
    print(f"程序总运行时间: {execution_time}")

