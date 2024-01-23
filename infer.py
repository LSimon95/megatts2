from models.megatts2 import Megatts

if __name__ == '__main__':
    megatts = Megatts(
        g_ckpt='generator.ckpt',
        g_config='configs/config_gan.yaml',
        plm_ckpt='plm.ckpt',
        plm_config='configs/config_plm.yaml',
        adm_ckpt='adm.ckpt',
        adm_config='configs/config_adm.yaml',
        symbol_table='/root/autodl-tmp/megatts2/data/ds/unique_text_tokens.k2symbols'
    )

    megatts.eval()

    megatts(
        '/root/autodl-tmp/megatts2/data/test',
        '小明明天我们出去玩吧'
    )