import sys

sys.path.append('/data/MambaCD/')

import argparse
import os
import time
import numpy as np

from changedetection.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from changedetection.datasets.make_data_loader import ChangeDetectionDatset, make_data_loader
from changedetection.utils_func.metrics import Evaluator
from changedetection.models.ChangeMambaBCD import ChangeMambaBCD
import changedetection.utils_func.lovasz_loss as L


class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        from thop import profile
        from thop import clever_format

        self.train_data_loader = make_data_loader(args)
        self.evaluator = Evaluator(num_class=2)

        self.deep_model = ChangeMambaBCD(
            pretrained=args.pretrained_weight_path,
            patch_size=config.MODEL.VSSM.PATCH_SIZE,
            in_chans=config.MODEL.VSSM.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VSSM.DEPTHS,
            dims=config.MODEL.VSSM.EMBED_DIM,
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto"
                         else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        ).cuda()

        # FLOPs & Params
        input1 = torch.randn(1, 3, 256, 256).cuda()
        input2 = torch.randn(1, 3, 256, 256).cuda()
        flops, params = profile(self.deep_model, inputs=(input1, input2))
        flops, params = clever_format([flops, params], "%.3f")
        print(f"Model FLOPs: {flops}, Parameters: {params}")

        self.model_save_path = os.path.join(
            args.model_param_path, args.dataset,
            args.model_type + '_' + str(time.time())
        )
        os.makedirs(self.model_save_path, exist_ok=True)

        if args.resume is not None:
            checkpoint = torch.load(args.resume)
            state_dict = self.deep_model.state_dict()
            state_dict.update({k: v for k, v in checkpoint.items() if k in state_dict})
            self.deep_model.load_state_dict(state_dict)

        self.optim = optim.AdamW(
            self.deep_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        # ===== EarlyStop & Save 指标 =====
        self.patience = 50
        self.best_f1 = 0.0  # EarlyStop = F1
        self.best_kc = 0.0  # Save = Kappa
        self.counter = 0
        self.early_stop = False

    def training(self):
        best_round = []

        torch.cuda.empty_cache()
        elem_num = len(self.train_data_loader)
        train_enumerator = enumerate(self.train_data_loader)

        for _ in tqdm(range(elem_num)):
            itera, data = next(train_enumerator)
            pre_change_imgs, post_change_imgs, labels, _ = data

            pre_change_imgs = pre_change_imgs.cuda().float()
            post_change_imgs = post_change_imgs.cuda()
            labels = labels.cuda().long()

            output = self.deep_model(pre_change_imgs, post_change_imgs)

            self.optim.zero_grad()
            ce_loss = F.cross_entropy(output, labels, ignore_index=255)
            lovasz = L.lovasz_softmax(F.softmax(output, dim=1), labels, ignore=255)
            loss = ce_loss + 0.75 * lovasz
            loss.backward()
            self.optim.step()

            if (itera + 1) % 10 == 0:
                print(f'iter is {itera + 1}, overall loss is {loss}')

            if (itera + 1) % 500 == 0:
                self.deep_model.eval()
                rec, pre, oa, f1, iou, kc = self.validation()

                # ===== 打印指标 =====
                print(f"Validation metrics at iter {itera + 1}: "
                      f"Rec={rec:.4f}, Pre={pre:.4f}, OA={oa:.4f}, "
                      f"F1={f1:.4f}, IoU={iou:.4f}, Kappa={kc:.4f}")

                # ===== Save = Kappa =====
                if kc > self.best_kc:
                    self.best_kc = kc
                    torch.save(
                        self.deep_model.state_dict(),
                        os.path.join(self.model_save_path, f'{itera + 1}_model.pth')
                    )
                    best_round = [rec, pre, oa, f1, iou, kc]

                # ===== EarlyStop = F1 =====
                if f1 > self.best_f1:
                    self.best_f1 = f1
                    self.counter = 0
                else:
                    self.counter += 1
                    print(f'No F1 improvement for {self.counter} validations')
                    if self.counter >= self.patience:
                        print(f'Early stopping at iteration {itera + 1}')
                        self.early_stop = True

                self.deep_model.train()

            if self.early_stop:
                break

        print('The accuracy of the best round is ', best_round)

    def validation(self):
        print('---------starting evaluation-----------')
        self.evaluator.reset()

        torch.cuda.empty_cache()  # ⭐提前清理

        dataset = ChangeDetectionDatset(
            self.args.test_dataset_path,
            self.args.test_data_name_list,
            256, None, 'test'
        )
        loader = DataLoader(dataset, batch_size=1, num_workers=0)

        with torch.no_grad():
            for data in loader:
                pre, post, labels, _ = data
                pre = pre.cuda().float()
                post = post.cuda()
                labels = labels.cuda().long()

                with torch.cuda.amp.autocast():  # ⭐减少显存
                    output = self.deep_model(pre, post)

                pred = torch.argmax(output, dim=1)

                self.evaluator.add_batch(
                    labels.cpu().numpy(),
                    pred.cpu().numpy()
                )

                # ⭐释放显存（关键）
                del output, pred, pre, post, labels
                torch.cuda.empty_cache()

        print('---------evaluation completed----------')

        return (
            self.evaluator.Pixel_Recall_Rate(),
            self.evaluator.Pixel_Precision_Rate(),
            self.evaluator.Pixel_Accuracy(),
            self.evaluator.Pixel_F1_score(),
            self.evaluator.Intersection_over_Union(),
            self.evaluator.Kappa_coefficient()
        )


def main():
    parser = argparse.ArgumentParser(description="Training on CDD/LEVIR-CD+/SYSU-CD dataset")
    parser.add_argument('--cfg', type=str,
                        default='/data/MambaCD/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str, default='/data/MambaCD/ManbaCD/pretrained_weight/MambaBCD_Tiny_CDD_F1_0.8316.pth')
    parser.add_argument('--dataset', type=str, default='SYSU')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--train_dataset_path', type=str, default='/data/datasets/SYSU/train/')
    parser.add_argument('--train_data_list_path', type=str, default='/data/datasets/SYSU/train.txt')
    parser.add_argument('--test_dataset_path', type=str, default='/data/datasets/SYSU/test/')
    parser.add_argument('--test_data_list_path', type=str, default='/data/datasets/SYSU/test.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=60000)
    parser.add_argument('--model_type', type=str, default='DIEFEN')
    parser.add_argument('--model_param_path', type=str, default='../saved_models')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    args = parser.parse_args()
    with open(args.train_data_list_path, "r") as f:
        args.train_data_name_list = [data_name.strip() for data_name in f]

    with open(args.test_data_list_path, "r") as f:
        args.test_data_name_list = [data_name.strip() for data_name in f]

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()