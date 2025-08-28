import os
import torch
from utils.data_prefetcher import DataPrefetcher
from utils.mixup import mixup_data
from transformers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
import warnings
from utils.evaluate import get_all_embeddings, do_metric, do_validation_loss

warnings.filterwarnings("ignore")


class Trainer:
    def __init__(self, model, train_iter, train_set, val_iter, val_set, loss_func_list, num_epoches, accuracy_calculator, save_dir,
                 optimizer=None, loss_optimizer=None, mining_func=None, writer=None, knn_model=None,
                 mixup_enable=False, xbm_enable=False, xbm_start_iteration=0, xbm_close=0, validate_loss=True,
                 ref_includes_query=False, print_step=10, save_checkpoint_frequency=1):
        """
        初始化Trainer类
        
        Args:
            model: 训练模型
            train_iter: 训练数据迭代器
            train_set: 训练数据集
            val_iter: 验证数据迭代器
            val_set: 验证数据集
            loss_func_list: 损失函数列表
            num_epoches: 训练轮数
            accuracy_calculator: 准确率计算器
            save_dir: 模型保存目录
            optimizer: 优化器
            loss_optimizer: 损练优化器
            mining_func: 挖掘函数
            writer: TensorBoard写入器
            knn_model: KNN模型
            mixup_enable: 是否启用mixup
            xbm_enable: 是否启用xbm
            xbm_start_iteration: xbm开始迭代次数
            xbm_close: xbm关闭轮数
            validate_loss: 是否验证损失
            ref_includes_query: 是否包含查询
            print_step: 打印步长
            save_checkpoint_frequency: 保存检查点频率
        """
        self.model = model
        self.train_iter = train_iter
        self.train_set = train_set
        self.val_iter = val_iter
        self.val_set = val_set
        self.loss_func_list = loss_func_list
        self.num_epoches = num_epoches
        self.accuracy_calculator = accuracy_calculator
        self.save_dir = save_dir
        self.optimizer = optimizer
        self.loss_optimizer = loss_optimizer
        self.mining_func = mining_func
        self.writer = writer
        self.knn_model = knn_model
        self.mixup_enable = mixup_enable
        self.xbm_enable = xbm_enable
        self.xbm_start_iteration = xbm_start_iteration
        self.xbm_close = xbm_close
        self.validate_loss = validate_loss
        self.ref_includes_query = ref_includes_query
        self.print_step = print_step
        self.save_checkpoint_frequency = save_checkpoint_frequency
        
        # 确保保存目录存在
        self.weights_dir = os.path.join(self.save_dir, "weights")
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
            
        self.lr_decay_list = []
        
    def _setup_lr_scheduler(self):
        """
        设置学习率调度器
        """
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=10,
            num_training_steps=len(self.train_iter) * self.num_epoches
        )
        
    def train(self):
        """
        执行训练过程
        """
        self._setup_lr_scheduler()
        
        print("xbm enable: {}, xbm iteration: {}, mixup enable: {}".format(
            self.xbm_enable, self.xbm_start_iteration, self.mixup_enable))
        print(self.loss_func_list[0], self.loss_func_list[1], self.mining_func)

        loss_func = self.loss_func_list[0]
        for epoch in range(self.num_epoches):
            self.model.train()
            train_loss_sum = 0.0
            n = 0

            # 进行训练数据预加载，提升数据加载效率
            prefetcher = DataPrefetcher(self.train_iter)
            X, y = prefetcher.next()
            batch_idx = 0
            while X is not None:
                if self.xbm_enable:
                    # xbm_start_iteration轮迭代之后，使用 xbm 跨批量增强
                    if epoch * len(self.train_iter) + batch_idx >= self.xbm_start_iteration:
                        loss_func = self.loss_func_list[1]
                else:
                    loss_func = self.loss_func_list[0]

                # 提前结束 xbm 跨批量增强
                if epoch > self.num_epoches - self.xbm_close:
                    self.xbm_enable = False

                X = X.cuda()
                y = y.cuda()
                # print("train: {}, {}".format(X.shape, y.shape))
                # print(y)

                if self.mixup_enable:
                    # 使用 mixup 数据增强
                    X, y_a, y_b, lam = mixup_data(X, y, 0.1, True)
                    X, y_a, y_b = map(torch.tensor, (X, y_a, y_b))
                    X = X.cuda()
                    y_a = y_a.cuda()
                    y_b = y_b.cuda()

                    y_pred = self.model(X)
                    # y_pred = torch.nn.functional.normalize(y_pred, p=2, dim=1)
                    loss = lam * loss_func(y_pred, y_a) + (1 - lam) * loss_func(y_pred, y_b)
                else:
                    y_pred = self.model(X)
                    # y_pred = torch.nn.functional.normalize(y_pred, p=2, dim=1)
                    if self.mining_func:
                        # 在 batch 中进行难训练样本挖掘，提升效果
                        indices_tuple = self.mining_func(y_pred, y)
                        loss = loss_func(y_pred, y, indices_tuple)
                    else:
                        loss = loss_func(y_pred, y)

                loss = torch.mean(loss)

                self.optimizer.zero_grad()
                if self.loss_optimizer:
                    self.loss_optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.loss_optimizer:
                    self.loss_optimizer.step()

                train_loss_sum += loss.item()
                n += y.shape[0]

                if batch_idx % self.print_step == 0:
                    print("Epoch {} Iteration {}, Loss = {}".format(epoch, batch_idx, loss))
                    self.writer.add_scalar("loss/iter", loss.item(), epoch * len(self.train_iter) + batch_idx)

                self.model.train()
                X, y = prefetcher.next()

                self.lr_decay_list.append(self.optimizer.state_dict()["param_groups"][0]["lr"])
                self.writer.add_scalar("learning rate", self.lr_decay_list[-1], epoch * len(self.train_iter) + batch_idx)
                self.lr_scheduler.step()

                batch_idx += 1

            print("Epoch {}, Train Loss = {}".format(epoch, train_loss_sum / n))
            self.writer.add_scalar("loss/train", train_loss_sum / n, epoch)

            with torch.no_grad():
                if self.validate_loss:
                    val_loss = do_validation_loss(self.val_iter, self.model, self.loss_func_list[0], self.mining_func, self.print_step)
                    print("Epoch {}, Val Loss ={}".format(epoch, val_loss))
                    self.writer.add_scalar("loss/val", val_loss, epoch)

                if self.ref_includes_query is False:
                    acc = do_metric(self.train_set, self.val_set, self.model, self.accuracy_calculator, self.knn_model,
                                    ref_includes_query=self.ref_includes_query)
                else:
                    acc = do_metric(self.val_set, self.val_set, self.model, self.accuracy_calculator, self.knn_model,
                                    ref_includes_query=self.ref_includes_query)
                precision_at_1 = acc["precision_at_1"]
                r_precision = acc["r_precision"]
                r_map = acc["mean_average_precision_at_r"]
                knn_score = acc["knn_score"]

                self.writer.add_scalar("accuracy/precision@1", precision_at_1, epoch)
                self.writer.add_scalar("accuracy/r_precision", r_precision, epoch)
                self.writer.add_scalar("accuracy/mean_average_precision_at_r", r_map, epoch)
                self.writer.add_scalar("accuracy/knn_score", knn_score, epoch)

                # 根据保存频率参数决定是否保存检查点
                if self.save_checkpoint_frequency >= 1 and (epoch + 1) % self.save_checkpoint_frequency == 0:
                    best_model_name = os.path.join(self.weights_dir, "model-acc-%03d-%.04f-%.04f-%.04f.pth" % (
                        epoch, r_map, precision_at_1, r_precision))
                    torch.save(self.model.module.state_dict(), best_model_name)