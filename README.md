# ldm-image-geneartor

[LatentDiffusion](https://arxiv.org/abs/2112.10752)をベースにした生成モデルです。

# Usage

1. このリポジトリをクローンします
```sh
git clone https://github.com/uthree/ldm-image-generator/
```

2. ディレクトリを移動
```sh
cd ldm-image-generator
```

3. VAEを訓練する。
```sh
python3 train_vae.py <Dataset Directory Path> -e <num_epoch> -d <Device>
```

4. Diffusion Modelを訓練する。
```sh
python3 train_ldm.py <Dataset Directory Path> -e <num_epoch> -d <Device>
```

5. 推論する
```sh
python3 sample.py
```