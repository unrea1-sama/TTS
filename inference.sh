python3 inference.py \
--glow_hparams configs/myself.json \
--glow_ckpt logs/myself/G_1322.pth \
--input ../biaobei/biaobei_pingyin_glow-test.json \
--vocoder_ckpt hifi-gan/ckpt/g_02500000 \
--vocoder_config hifi-gan/ckpt/config.json